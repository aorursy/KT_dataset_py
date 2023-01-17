import pandas as pd
import numpy as np
import os
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

warnings.filterwarnings('ignore')
os.listdir('../input')
# read in application train and application test files
app_train = pd.read_csv('../input/application_train.csv')
app_test = pd.read_csv('../input/application_test.csv')
# How many rows and columns are there? train has the extra TARGET variable over test!
print(app_train.shape)
print(app_test.shape)
# What are the column types?
plt.figure(figsize = (10,6))
ax = app_train.dtypes.value_counts().plot(kind = 'bar', rot = 0)
ax.set_title('Count of Column Data Types')
ax.set_xlabel('Data Type')
ax.set_yticks([])
ax.grid(False)

# each rectangle is a bar
rects = ax.patches

# Make some labels.
labels = app_train.dtypes.value_counts().tolist()

# loop through each rectangle and put label
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width() / 2, height - 5, label,
            ha='center', va='bottom')
# define a function to get the missing values
def missing_check(df):
    '''Given a dataframe this determines the missing values and plots them'''
    missing_df = df.isnull().sum().reset_index()
    missing_df.columns = ['variable', 'missing_values']
    missing_df['Perc_Missing'] = missing_df['missing_values']*100/len(df)
    missing_df.sort_values('Perc_Missing', ascending = False, inplace = True)
    missing_df = missing_df.loc[missing_df['Perc_Missing']>0, :]
    if len(missing_df) == 0:
        return "No columns with missing values"
    else:
        missing_df['Perc_total'] = 100
        return missing_df
# visualize the top most columns with missing entries
abc = missing_check(app_train).reset_index(drop = True)

plt.figure(figsize = (10, 10))
plt.barh(abc.loc[:15, 'variable'], abc.loc[:15, 'Perc_total'], label = "Total rows")
plt.barh(abc.loc[:15,'variable'], abc.loc[:15,'Perc_Missing'], label = "Missing rows")
plt.legend()
abc['Column_Type'] = abc['variable'].map(lambda x: app_train[x].dtype)
abc.head()
plt.figure(figsize = (10,6))
plt.style.use('seaborn-darkgrid')
ax = abc['Column_Type'].value_counts().plot(kind = 'bar')
ax.grid(False)
ax.set_title('Column Type count with missing values')
ax.set_xlabel('Column Type')
ax.set_ylabel('Count of columns')
ax.set_yticks([])

rects = ax.patches
labels = abc['Column_Type'].value_counts().tolist()

for rect, label in zip(rects, labels):
    height = rect.get_height()
    width = rect.get_width()
    ax.text(rect.get_x() + width/2, height - 3, label)
# First lets select the object type columns from app_train
obj_cols = app_train.select_dtypes('object').columns.tolist()
print(obj_cols)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
count_le = 0

for col in obj_cols:
    if len(app_train[col].unique()) <= 2:
        le.fit(app_train[col])
        app_train[col] = le.transform(app_train[col])
        app_test[col] = le.transform(app_test[col])
        
        count_le += 1
        
print("Columns with label encoding:", count_le)
# get_dummies takes all categorical variables and creates a new column for each level
app_train_encoded = pd.get_dummies(app_train)
app_test_encoded = pd.get_dummies(app_test)
print(app_train_encoded.shape)
print(app_test_encoded.shape)
# align the dataframes - This removes the target variable too, so lets store it separately
train_labels = app_train_encoded['TARGET']

app_train_encoded, app_test_encoded = app_train_encoded.align(app_test_encoded,
                                                             join = 'inner',
                                                             axis = 1)

print(app_train_encoded.shape)
print(app_test_encoded.shape)
# attach train_labels to the app_train_encoded
app_train_encoded['TARGET'] = train_labels

# correlations
correlations = app_train_encoded.corr()['TARGET']
top_5_positive = correlations.sort_values(ascending = False)[:5]
top_5_negative = correlations.sort_values(ascending = True)[:5]

print(top_5_positive)
print(top_5_negative)
# What does the DAYS_BIRTH variable look like? This is the difference between the loan
# application date and birthdate of the applicant
plt.figure(figsize = (10,6))
app_train_encoded['DAYS_BIRTH'].plot.hist(bins = 25, edgecolor = 'k', rot = 0)
plt.title('DAYS_BIRTH Histogram')
plt.xlabel('DAYS_BIRTH')
# Lets convert this to positive values so that it makes more sense
app_train_encoded['DAYS_BIRTH'] = abs(app_train_encoded['DAYS_BIRTH'])
app_test_encoded['DAYS_BIRTH'] = abs(app_test_encoded['DAYS_BIRTH'])

plt.figure(figsize = (10,6))
app_train_encoded['DAYS_BIRTH'].plot.hist(bins = 25, edgecolor = 'k', rot = 0)
app_train_encoded['DAYS_BIRTH'].corr(app_train_encoded['TARGET'])
# age versus the outcome variable
plt.figure(figsize = (10, 8))

sns.kdeplot(app_train_encoded.loc[app_train_encoded['TARGET'] == 0, 'DAYS_BIRTH'],
           label = "TARGET = 0")
sns.kdeplot(app_train_encoded.loc[app_train_encoded['TARGET'] == 1, 'DAYS_BIRTH'],
           label = "TARGET = 1")
plt.title('DAYS_BIRTH vs TARGET')
plt.xlabel('DAYS_BIRTH')
plt.ylabel('Density')
correl = app_train_encoded[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'TARGET']].corr()

plt.figure(figsize = (10,6))
sns.heatmap(correl, cmap = plt.cm.RdYlBu_r, annot = True, vmin = -0.6, vmax = 0.8)
plt.yticks(rotation = 'horizontal')
plt.title('Correlation Heatmap')
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = 'median')

if 'TARGET' in app_train_encoded:
    app_train_encoded.drop(columns = ['TARGET'], inplace = True)
    

# we need to remove the SK_ID_CURR variable before we do the scaling since these do not 
# have to be scaled
train_id = app_train_encoded['SK_ID_CURR']
test_id = app_test_encoded['SK_ID_CURR']

# Fit on train data and then transform test data as well!
app_train_encoded.drop(columns = ['SK_ID_CURR'], inplace = True)
app_test_encoded.drop(columns = ['SK_ID_CURR'], inplace = True)

# get column names from the dataframe since imputer converts these to matrices
features = app_train_encoded.columns.tolist()

# impute the missing values
app_train_enc_imput_med = imputer.fit_transform(app_train_encoded)
app_test_enc_imput_med = imputer.transform(app_test_encoded)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0,1))

app_train_enc_imput_med = scaler.fit_transform(app_train_enc_imput_med)
app_test_enc_imput_med = scaler.transform(app_test_enc_imput_med)

print("Test data:", app_test_enc_imput_med.shape)
print("Train data shape:", app_train_enc_imput_med.shape)
# Lets break our train data into training and validation datasets - 0.7 and 0.3
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(app_train_enc_imput_med, 
                                                   train_labels,
                                                   test_size = 0.3,
                                                   random_state = 2)

print("Train data shape:", X_train.shape)
print("Test data shape:", X_test.shape)
print("train labels shape:", y_train.shape)
print("test labels shape:", y_test.shape)
# Fit the model on training data!
from sklearn.linear_model import LogisticRegression

# Logistic regression model
log_reg = LogisticRegression(C =  0.0001)

log_reg.fit(X_train, y_train)
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(log_reg, random_state = 2).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = features)
log_reg_coeff = pd.Series(log_reg.coef_.tolist()[0], index = features).sort_values()
print("Top 5 negative:\n",log_reg_coeff[:5])
print("\nTop 5 positive:\n",log_reg_coeff[-5:])
# predict the probability of each class in the test data and extract the probability for
# class = 1
log_predictions = log_reg.predict_proba(X_test)[:,1]
log_predictions[:5]
# Whats the baseline accuracy for this problem
y_test.value_counts()/len(y_test)
# lets calculate the AUCROC metric
from sklearn import metrics
logit_accuracy = metrics.roc_auc_score(y_test, log_predictions)
print("Logistic Regression Accuracy: {0:.2f}".format(logit_accuracy))
# 10 fold cross validation setup
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits = 10, random_state = 2).split(X = app_train_enc_imput_med,
                                                               y = train_labels)

accuracies = []

# train the logistic regression model on each and get auc-roc scores
for train, holdout in kfold:
    log_reg.fit(app_train_enc_imput_med[train,:], train_labels[train])
    predictions = log_reg.predict_proba(app_train_enc_imput_med[holdout,:])[:,1]
    accuracy = metrics.roc_auc_score(train_labels[holdout], predictions)
    accuracies.append(accuracy)
# Scatter plot of the accuracies achieved through cross validation
plt.figure(figsize = (6, 6))
plt.scatter(range(1,11), accuracies)
plt.title('ROC AUC score with Cross Validation')
plt.xlabel('Fold Number')
plt.ylabel('Area under the ROC curve')

# draw a line for the mean auc-roc score
plt.axhline(y = np.mean(accuracies), color = 'red', linewidth = 1)
plt.text(x = 6, y = 0.695, 
         s = "Average Accuracy:{0:.3f}".format(np.mean(accuracies)),
        color = 'red')
# train the random Forest Model
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators = 100,
                                 n_jobs = 1,
                                 random_state = 2)

# 5 fold cross validation
kfold = StratifiedKFold(n_splits = 5, random_state = 2).split(X = app_train_enc_imput_med,
                                                              y = train_labels)

# Calculate the auc-roc scores for each train, holdout score
accuracies = []
for train, holdout in kfold:
    rf_model.fit(app_train_enc_imput_med[train, :], train_labels[train])
    predictions = rf_model.predict_proba(app_train_enc_imput_med[holdout, :])[:,1]
    accuracies.append(metrics.roc_auc_score(train_labels[holdout], predictions))
# Lets draw a scatter plot of the AUC-ROC scores for each fold!
plt.figure(figsize = (6,6))
plt.scatter(range(1,6), accuracies)
plt.title('Cross Validation AUC-ROC with Random Forest')
plt.xlabel('Fold Number')
plt.ylabel('ROC-AUC Score')

plt.axhline(y = np.mean(accuracies), color = 'red', linewidth = 1)
plt.text(x = 3.5, y = 0.715,
         s = "Avg AUC-ROC score:{0:.2f}".format(np.mean(accuracies)),
        color = 'red')
plt.show()
# convert the test and train matrices to a dataframe
app_train_part1 = pd.DataFrame(app_train_enc_imput_med,
                              columns = features)

app_test_part1 = pd.DataFrame(app_test_enc_imput_med,
                              columns = features)

# Append the SK_ID_CURR variable
app_train_part1 = pd.concat([app_train_part1, train_id], axis = 1)
app_test_part1 = pd.concat([app_test_part1, test_id], axis = 1)
# write to a csv file
app_train_part1.to_csv("app_train_part1.csv", index = False)
app_test_part1.to_csv("app_test_part1.csv", index = False)
train_labels.to_csv("train_labels.csv", index = False)
