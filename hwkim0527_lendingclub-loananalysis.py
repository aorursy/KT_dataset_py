# This Python 3 environment
# here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import time

# Classifier Libraries
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import collections
from tensorflow import feature_column

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.utils import check_X_y
import sklearn.utils
from sklearn.metrics import classification_report, roc_curve, auc, brier_score_loss

# Other Libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings
warnings.filterwarnings("ignore")

import os

# converter for percent values
p2f = lambda s: np.NaN if not s else float(s.strip().strip("%")) / 100

df = pd.read_csv("/kaggle/input/lending-club/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv", low_memory=False,
                   converters={"int_rate": p2f, "revol_util": p2f})

print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
dropna_cols = ["funded_amnt"]
len_prev = len(df)
df.dropna(subset=dropna_cols, inplace=True)
n_dropped = len_prev-len(df)

dt_tmp=pd.to_datetime(df['issue_d'])
df['year']=pd.to_datetime(dt_tmp).dt.strftime("%Y").dropna(axis=0)
df['year'].astype('int')


# Drop irrelevant columns
df.drop(['id', 'member_id', 'emp_title', 'desc', 'zip_code', 'title'], axis=1, inplace=True)

print(f"Dropped {n_dropped} rows ({100*n_dropped/len_prev:.2f}%) with NaN values, {len(df)} rows remaining.")

df 
print(df["loan_status"].unique())

df["defaulted"] = df["loan_status"].map(lambda x: 1 if x in ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off", "In Grace Period",
                                                             "Late (16-30 days)", "Late (31-120 days)"]
                                                 else 0 if x in ["Fully Paid", 'Does not meet the credit policy. Status:Fully Paid', 'Current']
                                                 else -1)
len_prev = len(df)
df=df[df['defaulted']!= -1]
#df.query("defaulted != -1", inplace=True)

n_dropped = len_prev - len(df)
print(f"Dropped {n_dropped} rows ({100*n_dropped/len_prev:.2f}%) with invalid loan_status, {len(df)} rows remaining.")
f, ax = plt.subplots(1,2, figsize=(16,8))

colors = ["#3791D7", "#D72626"]
labels ="Normal", "Defaulted"

plt.suptitle('Information on Loan Conditions', fontsize=20)

df["defaulted"].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', ax=ax[0], shadow=True, colors=colors, 
                                             labels=labels, fontsize=12, startangle=70)


# ax[0].set_title('State of Loan', fontsize=16)
ax[0].set_ylabel('% of Condition of Loans', fontsize=14)

# sns.countplot('loan_condition', data=df, ax=ax[1], palette=colors)
# ax[1].set_title('Condition of Loans', fontsize=20)
# ax[1].set_xticklabels(['Good', 'Bad'], rotation='horizontal')
palette = ["#3791D7", "#E01E1B"]

sns.barplot(x="year", y="funded_amnt", hue="defaulted", data=df, palette=palette, estimator=lambda x: len(x) / len(df) * 100)
ax[1].set(ylabel="(%)")
df['addr_state'].unique()

# Make a list with each of the regions by state.

west = ['CA', 'OR', 'UT','WA', 'CO', 'NV', 'AK', 'MT', 'HI', 'WY', 'ID']
south_west = ['AZ', 'TX', 'NM', 'OK']
south_east = ['GA', 'NC', 'VA', 'FL', 'KY', 'SC', 'LA', 'AL', 'WV', 'DC', 'AR', 'DE', 'MS', 'TN' ]
mid_west = ['IL', 'MO', 'MN', 'OH', 'WI', 'KS', 'MI', 'SD', 'IA', 'NE', 'IN', 'ND']
north_east = ['CT', 'NY', 'PA', 'NJ', 'RI','MA', 'MD', 'VT', 'NH', 'ME']



df['region'] = np.nan

def finding_regions(state):
    if state in west:
        return 'West'
    elif state in south_west:
        return 'SouthWest'
    elif state in south_east:
        return 'SouthEast'
    elif state in mid_west:
        return 'MidWest'
    elif state in north_east:
        return 'NorthEast'
    


df['region'] = df['addr_state'].apply(finding_regions)
# This code will take the current date and transform it into a year-month format
df['complete_date'] = pd.to_datetime(df['issue_d'])

group_dates = df.groupby(['complete_date', 'region'], as_index=False).sum()

group_dates['issue_d'] = [month.to_period('M') for 
                          month in group_dates['complete_date']]

group_dates = group_dates.groupby(['issue_d', 'region'], as_index=False).sum()
group_dates = group_dates.groupby(['issue_d', 'region'], as_index=False).sum()
group_dates['funded_amnt'] = group_dates['funded_amnt']/1000


df_dates = pd.DataFrame(data=group_dates[['issue_d','region','funded_amnt']])


plt.style.use('dark_background')
cmap = plt.cm.Set3

by_issued_amount = df_dates.groupby(['issue_d', 'region']).funded_amnt.sum()
by_issued_amount.unstack().plot(stacked=False, colormap=cmap, grid=False, legend=True, figsize=(15,6))

plt.title('Loans issued by Region', fontsize=16)
employment_length = ['10+ years', '< 1 year', '1 year', '3 years', '8 years', '9 years',
                    '4 years', '5 years', '6 years', '2 years', '7 years', 'n/a']

# Create a new column and convert emp_length to integers.

lst = [df]
df['emp_length_int'] = np.nan

for col in lst:
    col.loc[col['emp_length'] == '10+ years', "emp_length_int"] = 10
    col.loc[col['emp_length'] == '9 years', "emp_length_int"] = 9
    col.loc[col['emp_length'] == '8 years', "emp_length_int"] = 8
    col.loc[col['emp_length'] == '7 years', "emp_length_int"] = 7
    col.loc[col['emp_length'] == '6 years', "emp_length_int"] = 6
    col.loc[col['emp_length'] == '5 years', "emp_length_int"] = 5
    col.loc[col['emp_length'] == '4 years', "emp_length_int"] = 4
    col.loc[col['emp_length'] == '3 years', "emp_length_int"] = 3
    col.loc[col['emp_length'] == '2 years', "emp_length_int"] = 2
    col.loc[col['emp_length'] == '1 year', "emp_length_int"] = 1
    col.loc[col['emp_length'] == '< 1 year', "emp_length_int"] = 0.5
    col.loc[col['emp_length'] == 'n/a', "emp_length_int"] = 0
    
# Loan issued by Region and by Credit Score grade
# Change the colormap for tomorrow!

sns.set_style('whitegrid')

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
cmap = plt.cm.inferno

by_interest_rate = df.groupby(['year', 'region']).int_rate.mean()
by_interest_rate.unstack().plot(kind='area', stacked=True, colormap=cmap, grid=False, legend=False, ax=ax1, figsize=(16,12))
ax1.set_title('Average Interest Rate by Region', fontsize=14)


by_employment_length = df.groupby(['year', 'region']).emp_length_int.mean()
by_employment_length.unstack().plot(kind='area', stacked=True, colormap=cmap, grid=False, legend=False, ax=ax2, figsize=(16,12))
ax2.set_title('Average Employment Length by Region', fontsize=14)
# plt.xlabel('Year of Issuance', fontsize=14)

by_dti = df.groupby(['year', 'region']).dti.mean()
by_dti.unstack().plot(kind='area', stacked=True, colormap=cmap, grid=False, legend=False, ax=ax3, figsize=(16,12))
ax3.set_title('Average Debt-to-Income by Region', fontsize=14)

by_income = df.groupby(['year', 'region']).annual_inc.mean()
by_income.unstack().plot(kind='area', stacked=True, colormap=cmap, grid=False, ax=ax4, figsize=(16,12))
ax4.set_title('Average Annual Income by Region', fontsize=14)
ax4.legend(bbox_to_anchor=(-1.0, -0.5, 1.8, 0.1), loc=10,prop={'size':12},
           ncol=5, mode="expand", borderaxespad=0.)
df_year_sum=df.groupby(by=['year'], as_index=False).agg({'funded_amnt':np.sum})

# 연도별, 대출금액 추이 

plt.figure(figsize=(12,8))
sns.barplot('year', 'funded_amnt', data=df_year_sum, palette='tab10')
plt.title('Issuance of Loans', fontsize=16)
plt.xlabel('Year', fontsize=14)
plt.ylabel('Amount of loans by year', fontsize=14)
# Let's visualize how many loans were issued by creditscore
f, ((ax1, ax2)) = plt.subplots(1, 2)
cmap = plt.cm.coolwarm

by_credit_score = df.groupby(['year', 'grade']).funded_amnt.mean()
by_credit_score.unstack().plot(legend=False, ax=ax1, figsize=(14, 4), colormap=cmap)
ax1.set_title('Loans issued by Credit Score', fontsize=14)
    
    
by_inc = df.groupby(['year', 'grade']).int_rate.mean()
by_inc.unstack().plot(ax=ax2, figsize=(14, 4), colormap=cmap)
ax2.set_title('Interest Rates by Credit Score', fontsize=14)

ax2.legend(bbox_to_anchor=(-1.0, -0.3, 1.7, 0.1), loc=5, prop={'size':12},
           ncol=7, mode="expand", borderaxespad=0.)
fig = plt.figure(figsize=(16,12))

ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(212)

cmap = plt.cm.coolwarm_r

loans_by_region = df.groupby(['grade', 'defaulted']).size()
loans_by_region.unstack().plot(kind='bar', stacked=True, colormap=cmap, ax=ax1, grid=False)
ax1.set_title('Type of Loans by Grade', fontsize=14)


loans_by_grade = df.groupby(['sub_grade', 'defaulted']).size()
loans_by_grade.unstack().plot(kind='bar', stacked=True, colormap=cmap, ax=ax2, grid=False)
ax2.set_title('Type of Loans by Sub-Grade', fontsize=14)

by_interest = df.groupby(['year', 'defaulted']).int_rate.mean()
by_interest.unstack().plot(ax=ax3, colormap=cmap)
ax3.set_title('Average Interest rate by Loan Condition', fontsize=14)
ax3.set_ylabel('Interest Rate (%)', fontsize=12)
plt.figure(figsize=(18,18))

# Create a dataframe for bad loans
bad_df = df.loc[df['defaulted'] == 1]

plt.subplot(211)
g = sns.boxplot(x='home_ownership', y='funded_amnt', hue='defaulted',
               data=bad_df, color='r')

g.set_xticklabels(g.get_xticklabels(),rotation=45)
g.set_xlabel("Type of Home Ownership", fontsize=12)
g.set_ylabel("Loan Amount", fontsize=12)
g.set_title("Distribution of Defaulted Loan \n by Home Ownership", fontsize=16)


plt.subplots_adjust(hspace = 0.6, top = 0.8)

plt.show()
# Handling Missing Numeric Values

# Transform Missing Values for numeric dataframe
dropna_cols = ["avg_cur_bal", "bc_util", "loan_status", "dti", "inq_last_6mths"]
df.dropna(subset=dropna_cols, inplace=True)

# Nevertheless check what these variables mean tomorrow in the morning.
for col in ("funded_amnt", 'dti', 'annual_inc_joint', 'il_util', 'mths_since_rcnt_il', 'open_acc_6m', 'open_acc_6m', 'open_il_12m', 'emp_length_int',
           'open_il_24m', 'inq_last_12m', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl',
           'mths_since_last_record', 'mths_since_last_major_derog', 'mths_since_last_delinq', 'total_bal_il', 'tot_coll_amt',
           'tot_cur_bal', 'total_rev_hi_lim', 'revol_util', 'collections_12_mths_ex_med', 'open_acc', 'inq_last_6mths',
           'verification_status_joint', 'acc_now_delinq'):
    df[col] = df[col].fillna(0)
    
# # Get the mode of next payment date and last payment date and the last date credit amount was pulled   
df["next_pymnt_d"] = df.groupby("region")["next_pymnt_d"].transform(lambda x: x.fillna(x.mode))
df["last_pymnt_d"] = df.groupby("region")["last_pymnt_d"].transform(lambda x: x.fillna(x.mode))
df["last_credit_pull_d"] = df.groupby("region")["last_credit_pull_d"].transform(lambda x: x.fillna(x.mode))
df["earliest_cr_line"] = df.groupby("region")["earliest_cr_line"].transform(lambda x: x.fillna(x.mode))

# # Get the mode on the number of accounts in which the client is delinquent
df["pub_rec"] = df.groupby("region")["pub_rec"].transform(lambda x: x.fillna(x.median()))

# # Get the mean of the annual income depending in the region the client is located.
df["annual_inc"] = df.groupby("region")["annual_inc"].transform(lambda x: x.fillna(x.mean()))

# Get the mode of the  total number of credit lines the borrower has 
df["total_acc"] = df.groupby("region")["total_acc"].transform(lambda x: x.fillna(x.median()))

# Mode of credit delinquencies in the past two years.
df["delinq_2yrs"] = df.groupby("region")["delinq_2yrs"].transform(lambda x: x.fillna(x.mean()))

df["hardship_flag01"] = df["hardship_flag"].map(lambda x: 1 if x == "Y" else 0)
df["joint_application_flag01"] = df["application_type"].map(lambda x: 0 if x=="Individual" else 1)
df["listed_as_whole_flag01"] = df["initial_list_status"].map(lambda x: 1 if x=="w" else 1)

individual_indices = df["joint_application_flag01"] == 0
df.loc[individual_indices, "annual_inc_joint"] = df[individual_indices]["annual_inc"]
df.loc[individual_indices, "dti_joint"] = df[individual_indices]["dti"]
df.loc[individual_indices, "verification_status_joint"] = df[individual_indices]["verification_status"]
df.loc[individual_indices, "revol_bal_joint"] = df[individual_indices]["revol_bal"]

dropna_cols = ["annual_inc_joint", "dti_joint", "verification_status_joint", "revol_bal_joint"]
df.dropna(subset=dropna_cols, inplace=True)


# Our goal is to predict the loan status, given by the boolean "defaulted" column
y = df["defaulted"].values

# We define the columns used as features to train our model
# continuous valued columns or 

x_columns_cont = ["funded_amnt", "annual_inc", "annual_inc_joint", 
                "dti", "dti_joint","fico_range_low", "fico_range_high", "inq_last_6mths", "mort_acc",
                "open_acc", "pub_rec", "pub_rec_bankruptcies", "revol_bal", "revol_bal_joint", "revol_util"]


# binary categorical (0/1) columns
x_columns_bin = ['hardship_flag01', "joint_application_flag01"]

# columns with categorical values that need to be one hot encoded

x_columns_cat = ["grade", "sub_grade","term", "purpose", "hardship_flag","addr_state"]  # "verification_status_joint",


ct = ColumnTransformer(transformers=[
    ("identity", FunctionTransformer(func=lambda x: x, validate=False), x_columns_cont + x_columns_bin),
    ("onehot", OneHotEncoder(sparse=False, handle_unknown="ignore"), x_columns_cat),
])

# columns with categorical values that need to be one hot encoded

X_cols = x_columns_cont + x_columns_bin + x_columns_cat

df=df[X_cols]

X = ct.fit_transform(df)

X_col_labels = x_columns_cont + x_columns_bin + list(ct.named_transformers_["onehot"].get_feature_names())

# check for nans/infs and other stuff
X, y = check_X_y(X, y)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

ndef_train = np.sum(y == 1)
print(f"{ndef_train} ({100*ndef_train/len(y):.2f}%) defaulted in training data set")

train_ndefault = np.sum(y)
X_train_rest_nondefault = X[y == 0]
X_train_rest_balanced_nondefault = X_train_rest_nondefault[
    sklearn.utils.random.sample_without_replacement(len(X_train_rest_nondefault),
                                                    train_ndefault, random_state=42)]

X_train_rest_balanced = np.concatenate([X[y == 1], X_train_rest_balanced_nondefault])

y_train_balanced = np.concatenate([np.ones(train_ndefault),np.zeros(train_ndefault)])
# Now the classes are balanced:
print(f"{np.sum(y_train_balanced == 1)} ({100*np.sum(y==1)/len(y_train_balanced):.2f}%) defaulted in balanced training data set")

# define a function to correct probabilities coming from models trained on the balanced dataset
beta = ndef_train / (len(y) - ndef_train) # ratio of defaults to non-defaults
# because of numerical errors the probability could be slightly above 1, so we clip the value
correct_balanced_probabilities = lambda probs: np.clip(beta * probs / ((beta - 1) * probs + 1), 0, 1)

del X_train_rest_nondefault, X_train_rest_balanced_nondefault

train_b_x, test_b_x, train_b_y, test_b_y = train_test_split(X_train_rest_balanced, y_train_balanced, test_size=0.1)
#train_ub_x, test_ub_x, train_ub_y, test_ub_y = train_test_split(X, y, test_size=0.025)

del X_train_rest_balanced, y_train_balanced, X, y
import keras
from keras import backend as K
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.callbacks import LearningRateScheduler

def get_model():
    model = tf.keras.Sequential([
    layers.Dense(1024, input_shape=(train_b_x.shape[1],), use_bias=False, activation='relu'),
    layers.BatchNormalization(scale=False),
    layers.Dense(1024, activation='relu', use_bias=False),
    layers.BatchNormalization(scale=False),
    layers.Dropout(0.3),
    layers.Dense(512, activation='relu', use_bias=False),
    layers.BatchNormalization(scale=False),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu', use_bias=False),
    layers.BatchNormalization(scale=False),
    layers.Dropout(0.3),
    layers.Dense(512, activation='relu', use_bias=False),
    layers.BatchNormalization(scale=False),
    layers.Dropout(0.3),
    layers.Dense(1024, activation='relu', use_bias=False),
    layers.BatchNormalization(scale=False),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu', use_bias=False),
    layers.BatchNormalization(scale=False),
    layers.Dense(1, activation='sigmoid')])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.05, decay=0.005),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


model = get_model()

checkpoint_path = "/kaggle/working/cp100.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

#체크포인트 콜백 만들기
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.summary()

#model.load_weights(checkpoint_path)
model.fit(train_b_x, train_b_y, epochs=1, callbacks = [cp_callback]) 


model.load_weights(checkpoint_path)
model.fit(train_b_x, train_b_y, epochs=1, callbacks = [cp_callback]) 
loss, accuracy = model.evaluate(test_b_x, test_b_y)
print("정확도", accuracy)
import itertools

# Create a confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=14)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt

predictions = model.predict(test_b_x )

y_predict=np.where(predictions>=0.5, 1, 0)

predictions_cm = confusion_matrix(test_b_y  , y_predict)
actual_cm = confusion_matrix(test_b_y  , test_b_y  )

print(classification_report(test_b_y, y_predict))

labels = ['Non Defaulted', 'Dafaulted']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predictions_cm, labels, title="Prediction \n Confusion Matrix", cmap=plt.cm.Reds)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)
# Train a random forest classifier
rf_model = RandomForestClassifier(n_estimators=100, oob_score=True, class_weight="balanced", n_jobs=-1, verbose=True)
rf_model.fit(train_b_x , train_b_y )

# print some statistics
y_predict = rf_model.predict_proba(test_b_x )[:,1]
y_predict = y_predict > 0.5

predictions_cm = confusion_matrix(test_b_y  , y_predict)
actual_cm = confusion_matrix(test_b_y  , test_b_y  )

print(classification_report(test_b_y, y_predict))

labels = ['Non Defaulted', 'Dafaulted']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predictions_cm, labels, title="Prediction \n Confusion Matrix", cmap=plt.cm.Reds)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)


rf_model_feature_importances = rf_model.feature_importances_
del rf_model


# Train a linear SVM classifier on a subset of the unbalanced dataset using "balanced" class weights
X_train_rest_sample, y_train_sample = sklearn.utils.resample(train_b_x , train_b_y , n_samples=2000, random_state=42)

print(f"{np.sum(y_train_sample == 1)} ({100*np.sum(y_train_sample==1)/len(y_train_sample):.2f}%) defaulted in subsampled training data set")

svm_model = svm.SVC(kernel="linear", class_weight="balanced", probability=True)
svm_model.fit(X_train_rest_sample, y_train_sample)

# predict
y_predict = svm_model.predict_proba(test_b_x )[:,1]
y_predict=np.where(y_predict>0.5, 1, 0)

print(classification_report(test_b_y, y_predict))

labels = ['Non Defaulted', 'Dafaulted']

fig = plt.figure(figsize=(16,8))

fig.add_subplot(221)
plot_confusion_matrix(predictions_cm, labels, title="Prediction \n Confusion Matrix", cmap=plt.cm.Reds)

fig.add_subplot(222)
plot_confusion_matrix(actual_cm, labels, title="Confusion Matrix \n (with 100% accuracy)", cmap=plt.cm.Greens)

del svm_model

