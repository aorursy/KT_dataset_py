import pandas as pd 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pylab as plt 
from pandas_profiling import ProfileReport
import os 
df_train = pd.read_csv('../input/credit-score/devC_data/arpu_train.csv')
df_test = pd.read_csv('../input/credit-score/devC_data/arpu_test.csv')
test_submisstion  = pd.read_csv('../input/submisstion-test/test_submission.csv')
# loan = pd.read_csv('/kaggle/input/devC_data/loan.csv')
# recharge = pd.read_csv('/kaggle/input/devC_data/recharge.csv')
# sample_submission = pd.read_csv('/kaggle/input/devC_data/sample_submission.csv')
# temp = pd.read_csv('/kaggle/input/devC_data/temp.csv')
result = pd.concat([df_train, test_submisstion], axis=1, join='inner')
df_train = result.loc[:,:'COL_27d']
df_train
df_train.describe().T
print(" ==========> COL_13   ")
print((df_train.COL_13.median,df_train.COL_13.mean))
print(" ==========> COL_14  ")
print((df_train.COL_14.median,df_train.COL_14.mean))
print(" ==========> COL_15 ")
print((df_train.COL_15.median,df_train.COL_15.mean))
print(" ==========> COL_16 ")
print((df_train.COL_16.median,df_train.COL_16.mean))
print(" ==========> COL_17 ")
print((df_train.COL_17.median,df_train.COL_17.mean))
print(" ==========> COL_18 ")
print((df_train.COL_18.median,df_train.COL_18.mean))
df_train.COL_13.fillna(value='B240008', inplace = True)
df_train.COL_14.fillna(value= np.float64(df_train.COL_14.median()), inplace = True)
df_train.COL_15.fillna(value='B240',    inplace = True)
df_train.COL_16.fillna(value='M',       inplace = True)
df_train.COL_17.fillna(value= np.float64(df_train.COL_17.mean()) , inplace = True)
df_train.COL_16.fillna(value= np.float64(df_train.COL_18.mean())     , inplace = True)
df_test.describe().T
df_test
print(" ==========> COL_13   ")
print((df_test.COL_13.median,df_test.COL_13.mean))
print(" ==========> COL_14  ")
print((df_test.COL_14.median,df_test.COL_14.mean))
print(" ==========> COL_15 ")
print((df_test.COL_15.median,df_test.COL_15.mean))
print(" ==========> COL_16 ")
print((df_test.COL_16.median,df_test.COL_16.mean))
print(" ==========> COL_17 ")
print((df_test.COL_17.median,df_test.COL_17.mean))
print(" ==========> COL_18 ")
print((df_test.COL_18.median,df_test.COL_18.mean))
df_test.COL_13.fillna(value='B075006', inplace = True)
df_test.COL_14.fillna(value= np.float64(df_train.COL_14.median()), inplace = True)
df_test.COL_15.fillna(value='B075',    inplace = True)
df_test.COL_16.fillna(value='F',       inplace = True)
df_test.COL_17.fillna(value= np.float64(df_test.COL_17.mean()) , inplace = True)
df_test.COL_16.fillna(value= np.float64(df_test.COL_18.mean()) , inplace = True)
print(" ==========> COL_18   ")
print((df_train.COL_18.median,df_train.COL_18.mean))
print(" ==========> COL_19  ")
print((df_train.COL_19.median,df_train.COL_19.mean))
print(" ==========> COL_20 ")
print((df_train.COL_20.median,df_train.COL_20.mean))
print(" ==========> COL_21 ")
print((df_train.COL_21.median,df_train.COL_21.mean))
print(" ==========> COL_22 ")
print((df_train.COL_22.median,df_train.COL_22.mean))
print(" ==========> COL_27c ")
print((df_train.COL_27c.median,df_train.COL_27c.mean))
df_train.COL_18.fillna(value= np.float64(df_train.COL_18.mean()) ,inplace=True)
df_train.COL_19.fillna(value= np.float64(df_train.COL_19.mean()) ,inplace=True)
df_train.COL_20.fillna(value=np.float64(df_train.COL_20.mean()),inplace=True)
df_train.COL_21.fillna(value= 0.0,inplace = True)
df_train.COL_22.fillna(value= 5500.0,inplace = True)
df_train.COL_27c.fillna(value=np.float64(df_train.COL_27c.mean()),inplace = True)
print(" ==========> COL_18   ")
print((df_test.COL_18.median,df_train.COL_18.mean))
print(" ==========> COL_19  ")
print((df_test.COL_19.median,df_train.COL_19.mean))
print(" ==========> COL_20 ")
print((df_test.COL_20.median,df_train.COL_20.mean))
print(" ==========> COL_21 ")
print((df_test.COL_21.median,df_train.COL_21.mean))
print(" ==========> COL_22 ")
print((df_test.COL_22.median,df_train.COL_22.mean))
print(" ==========> COL_27c ")
print((df_test.COL_27c.median,df_train.COL_27c.mean))
df_test.COL_18.fillna(value= np.float64(df_test.COL_18.mean()) ,inplace=True)
df_test.COL_19.fillna(value= np.float64(df_test.COL_19.mean()) ,inplace=True)
df_test.COL_20.fillna(value=np.float64(df_test.COL_20.mean()),inplace=True)
df_test.COL_21.fillna(value= 0.0,inplace = True)
df_test.COL_22.fillna(value= 5500.0,inplace = True)
df_test.COL_27c.fillna(value=np.float64(df_test.COL_27c.mean()),inplace = True)
df_train.isnull().sum()
df_test.COL_27a.fillna(value=np.float64(df_test.COL_27a.mean()),inplace = True)
df_test.COL_27b.fillna(value=np.float64(df_test.COL_27b.mean()),inplace = True)
df_test.COL_27d.fillna(value=np.float64(df_test.COL_27d.mean()),inplace = True)

df_train.COL_27a.fillna(value=np.float64(df_train.COL_27a.mean()),inplace = True)
df_train.COL_27b.fillna(value=np.float64(df_train.COL_27b.mean()),inplace = True)
df_train.COL_27d.fillna(value=np.float64(df_train.COL_27d.mean()),inplace = True)


df_train.isnull().sum().values.any(),df_test.isnull().sum().values.any()
df_train.to_csv('train_handle.csv',index=False, index_label=False)
df_test.to_csv('test_handle.csv',index=False, index_label=False)
df_train = pd.read_csv('train_handle.csv')
df_test  = pd.read_csv('test_handle.csv')
df_train.head()
df_test.head()
from sklearn import preprocessing

# Create x, where x the 'scores' column's values as floats
col_17 = df_train[['COL_17']].values.astype(np.float64)
col_18 = df_train[['COL_18']].values.astype(np.float64)
col_19 = df_train[['COL_19']].values.astype(np.float64)
col_20 = df_train[['COL_20']].values.astype(np.float64)
col_21 = df_train[['COL_21']].values.astype(np.float64)
col_22 = df_train[['COL_22']].values.astype(np.float64)
col_27a = df_train[['COL_27a']].values.astype(np.float64)
col_27b = df_train[['COL_27b']].values.astype(np.float64)
col_27c = df_train[['COL_27c']].values.astype(np.float64)
col_27d = df_train[['COL_27d']].values.astype(np.float64)

# Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor

col_17_scaled  = min_max_scaler.fit_transform(col_17)
col_18_scaled  = min_max_scaler.fit_transform(col_18)
col_19_scaled  = min_max_scaler.fit_transform(col_19)
col_20_scaled  = min_max_scaler.fit_transform(col_20)
col_21_scaled  = min_max_scaler.fit_transform(col_21)
col_22_scaled  = min_max_scaler.fit_transform(col_22)
col_27a_scaled = min_max_scaler.fit_transform(col_27a)
col_27b_scaled = min_max_scaler.fit_transform(col_27b)
col_27c_scaled = min_max_scaler.fit_transform(col_27c)
col_27d_scaled = min_max_scaler.fit_transform(col_27d)


df_train['COL_17']  = pd.DataFrame(col_17_scaled)
df_train['COL_18']  = pd.DataFrame(col_18_scaled)
df_train['COL_19']  = pd.DataFrame(col_19_scaled)
df_train['COL_20']  = pd.DataFrame(col_20_scaled)
df_train['COL_21']  = pd.DataFrame(col_21_scaled)
df_train['COL_22']  = pd.DataFrame(col_22_scaled)
df_train['COL_27a'] = pd.DataFrame(col_27a_scaled)
df_train['COL_27b'] = pd.DataFrame(col_27b_scaled)
df_train['COL_27c'] = pd.DataFrame(col_27c_scaled)
df_train['COL_27d'] = pd.DataFrame(col_27d_scaled)
from sklearn import preprocessing

# Create x, where x the 'scores' column's values as floats
col_17 = df_test[['COL_17']].values.astype(np.float64)
col_18 = df_test[['COL_18']].values.astype(np.float64)
col_19 = df_test[['COL_19']].values.astype(np.float64)
col_20 = df_test[['COL_20']].values.astype(np.float64)
col_21 = df_test[['COL_21']].values.astype(np.float64)
col_22 = df_test[['COL_22']].values.astype(np.float64)
col_27a = df_test[['COL_27a']].values.astype(np.float64)
col_27b = df_test[['COL_27b']].values.astype(np.float64)
col_27c = df_test[['COL_27c']].values.astype(np.float64)
col_27d = df_test[['COL_27d']].values.astype(np.float64)

# Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor

col_17_scaled  = min_max_scaler.fit_transform(col_17)
col_18_scaled  = min_max_scaler.fit_transform(col_18)
col_19_scaled  = min_max_scaler.fit_transform(col_19)
col_20_scaled  = min_max_scaler.fit_transform(col_20)
col_21_scaled  = min_max_scaler.fit_transform(col_21)
col_22_scaled  = min_max_scaler.fit_transform(col_22)
col_27a_scaled = min_max_scaler.fit_transform(col_27a)
col_27b_scaled = min_max_scaler.fit_transform(col_27b)
col_27c_scaled = min_max_scaler.fit_transform(col_27c)
col_27d_scaled = min_max_scaler.fit_transform(col_27d)


df_test['COL_17']  = pd.DataFrame(col_17_scaled)
df_test['COL_18']  = pd.DataFrame(col_18_scaled)
df_test['COL_19']  = pd.DataFrame(col_19_scaled)
df_test['COL_20']  = pd.DataFrame(col_20_scaled)
df_test['COL_21']  = pd.DataFrame(col_21_scaled)
df_test['COL_22']  = pd.DataFrame(col_22_scaled)
df_test['COL_27a'] = pd.DataFrame(col_27a_scaled)
df_test['COL_27b'] = pd.DataFrame(col_27b_scaled)
df_test['COL_27c'] = pd.DataFrame(col_27c_scaled)
df_test['COL_27d'] = pd.DataFrame(col_27d_scaled)
from sklearn.preprocessing import OneHotEncoder 

ohc = OneHotEncoder()

ohc_train = ohc.fit_transform(df_train.COL_15.values.reshape(-1,1)).toarray()
ohc_test  =  ohc.fit_transform(df_test.COL_15.values.reshape(-1,1)).toarray()

df_train_OneHot = pd.DataFrame(ohc_train, columns= ['City_' + str(ohc.categories_[0][i])
                                                   for i in range(len(ohc.categories_[0]))])
df_test_OneHot  = pd.DataFrame(ohc_test , columns= ['City_' + str(ohc.categories_[0][i])
                                                   for i in range(len(ohc.categories_[0]))])

df_train = pd.concat([df_train,df_train_OneHot], axis = 1)
df_test  = pd.concat([df_test ,df_test_OneHot] , axis = 1)

df_train = df_train.drop(columns = ['COL_15'], axis = 1)
df_test = df_test.drop(columns = ['COL_15'], axis = 1)
from sklearn.preprocessing import LabelEncoder

df_train['COL_16'] = LabelEncoder().fit_transform(df_train.COL_16)
df_test['COL_16']  = LabelEncoder().fit_transform(df_test.COL_16)

from category_encoders import woe
new_data = woe.WOEEncoder(df_train,return_df=True, handle_missing=True,handle_unknown=True)
df_data = new_data.fit_transform(X=df_train.COL_13, y=df_train.label)
df_train.COL_13 = df_data.COL_13
df_train.head()
df_test['COL_13'] = df_train.groupby(by=df_test.msisdn)['COL_13'].head()
df_test.COL_13.isnull().sum()
df_train['COL_14'] = 2020 - df_train.COL_14
df_test['COL_14']  = 2020 -  df_test.COL_14
df_train.label.astype(float)
df_train.T.head()
df_train.T.tail()
df_train.head()
df_test.dropna(inplace=True, how = 'any')
df_test.isnull().sum()
df_train.to_csv('data_train_encoder.csv',index=False, index_label=False)
df_test.to_csv('data_test_encoder.csv',index=False, index_label=False)
df_train = pd.read_csv('data_train_encoder.csv')
df_test = pd.read_csv('data_train_encoder.csv')

from matplotlib.gridspec import GridSpec

print(" train dataset ................. ")

plt.figure(figsize=(25, 25)) # Set figsize

columns = ['COL_13','COL_14','COL_17','COL_18','COL_19','COL_20','COL_22','COL_27a','COL_27b','COL_27c','COL_27d']
gses = GridSpec(6,2)

column_name = columns
for i, gs in enumerate(gses):
    if i == 11: 
        break
    ax = plt.subplot(gs)
    sns.distplot(df_train[column_name[i]])
plt.show()
from matplotlib.gridspec import GridSpec

print(" test dataset ")

plt.figure(figsize=(25, 25)) # Set figsize

columns = ['COL_14','COL_17','COL_18','COL_19','COL_20','COL_22','COL_27a','COL_27b','COL_27c','COL_27d']
gses = GridSpec(5,2)

column_name = columns
for i, gs in enumerate(gses):
    ax = plt.subplot(gs)
    sns.distplot(df_test[column_name[i]])
plt.show()
# Check ratio between classes
percentage_Good = (df_train['label'] == 1).sum() / df_train.shape[0] * 100
percentage_Bad = (df_train['label'] == 0).sum() / df_train.shape[0] * 100

print ('Percentage Good credit score: ', percentage_Good)
print ('Percentage Bad credit score: ', percentage_Bad)
fig = plt.figure(figsize=(7,7)) # Set figsize
# Your code here
sns.countplot(data=df_train, x='label')

plt.show()
# Original data
X = df_train.drop(columns='label')
y = df_train['label']

print ('X shape:', X.shape)
print ('y shape:', y.shape)
# import train_test_split
# Your code here
from sklearn.model_selection import train_test_split
# Whole dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

print("Number customer training dataset: ", X_train.shape)
print("Number customer testing dataset: ", X_test.shape)
print("Number customer training label dataset: ", y_train.shape)
print("Number customer testing label dataset: ", y_test.shape)

print("Total number of customer: ", len(X_train)+len(X_test))
sub_training_data = pd.concat ([X_train,y_train],axis = 1)
sub_training_data['label'].value_counts()
sub_training_data
# Fraud/non-fraud data
# Select row which "Class" is 1 and save in fraud_data
good_data = sub_training_data[sub_training_data['label'] == 1]
# Select row which "Class" is 0 and save in non_fraud_data
bad_data = sub_training_data[sub_training_data['label'] == 0]

# Number of fraud, non-fraud transactions
number_records_good = good_data.shape[0]
number_records_bad = bad_data.shape[0]

# Using sample function on data frame to randomly select number_records_fraud from non_fraud_data data frame
under_sample_bad_customer = bad_data.sample(number_records_good)
# **concat** under_sample_non_fraud and fraud_data to form under_sample_data
under_sample_data = pd.concat([under_sample_bad_customer, good_data], axis=0)

# Showing ratio
print("Percentage of normal transactions: ", under_sample_bad_customer.shape[0] / under_sample_data.shape[0])
print("Percentage of fraud transactions: ", good_data.shape[0] / under_sample_data.shape[0])
print("Total number of transactions in resampled data: ", under_sample_data.shape[0])

# Assigning X,y for Under-sampled Data
X_train_undersample = under_sample_data.drop(columns=['label'])
y_train_undersample = under_sample_data['label']

# Plot countplot
plt.figure(figsize=(7,7))
# Make a count plot to show ratio between 2 class on "Class" column
sns.countplot(data=under_sample_data, x='label')
plt.show()
# Fraud/non-fraud data
# Select row which "Class" is 1 and save in fraud_data
good_customer = sub_training_data[sub_training_data['label'] == 1]
# Select row which "Class" is 0 and save in non_fraud_data
bad_customer   = sub_training_data[sub_training_data['label'] == 0]

# Number of fraud, non-fraud transactions
number_records_good = good_customer.shape[0]
number_records_bad = bad_customer.shape[0]
# print(" before use over-sampling ",fraud_data.shape, non_fraud_data.shape)
# Using sample on fraud_data with replacement "replace = True",  since we take a larger sample than population

over_sample_good_customer = good_customer.sample(replace = True, n=number_records_bad)
new_data = over_sample_good_customer
# print(' using over sample fraud ',over_sample_fraud.shape)
# **concat** over_sample_fraud and non_fraud_data to form under_sample_data
over_sample_data = pd.concat([over_sample_good_customer, bad_customer], axis=0)

# Showing ratio

print("Percentage of normal transactions: ", bad_customer.shape[0]/over_sample_data.shape[0])
print("Percentage of fraud transactions: ", over_sample_good_customer.shape[0]/over_sample_data.shape[0])
print("Total number of transactions in resampled data: ", over_sample_data.shape[0])

# Assigning X, y for over-sampled dataset
X_train_oversample = over_sample_data.drop(columns=['label'])
y_train_oversample = over_sample_data['label']

# Plot countplot
plt.figure(figsize=(7,7))
# Make a count plot to show ratio between 2 class on "Class" column
sns.countplot(data=over_sample_data, x='label')
plt.show()
over_sample_data[over_sample_data['label']==0].shape,over_sample_data[over_sample_data['label']==1].shape
over_sample_data
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB


lr = LogisticRegression()
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()
bnb = BernoulliNB()
gnb = GaussianNB()
svm = LinearSVC()


models = [lr, dtc, rfc, gnb, bnb, svm]
models_name = ["Logistic Regression", "Decision Tree", "Random Forest", "Bernoulli NB", "Gaussian NB", "SVM"]
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import recall_score, precision_score

# We create an utils function, that take a trained model as argument and print out confusion matrix
# classification report base on X and y
def evaluate_model(estimator, X, y, description):
    # Note: We should test on the original test set
    prediction = estimator.predict(X)
#     print('Confusion matrix:\n', confusion_matrix(y, prediction))
#     print('Classification report:\n', classification_report(y, prediction))
#     print('Testing set information:\n', "Your code here")

    # Set print optionscapture the most fraudulent trcapture the most fraudulent transactions.ansactions.
    np.set_printoptions(precision=2)
    model_name = type(estimator).__name__
    return {'name': model_name, 
            'recall': recall_score(y, prediction),
            'precision': precision_score(y, prediction),
           'description': description}
# Now we will test on origin dataset (X_train_sub, y_train_sub)
# We loop for models
# For each model, we train with train_sub dataset
# and use evaluate_model function to test with test set
X_train_sub = sub_training_data.drop(columns='label')
X_train_sub = X_train_sub.drop(columns='msisdn')
X_test_data = X_test.drop(columns='msisdn')
y_train_sub = sub_training_data['label']
print(X_train_sub.shape,y_train_sub.shape,X_test_data.shape,y_test.shape)
scores_origin = []
for idx, model in enumerate(models):
    print("Model: {}".format(models_name[idx]))
    # Your code here
    model.fit(X_train_sub, y_train_sub)
    scores_origin.append(evaluate_model(model, X_test_data, y_test, 'origin'))
    
    print("=======================================")
# Now we will test on Undersampled dataset (X_train_undersample, y_train_undersample)
# We loop for models
# For each model, we train with train_undersample dataset
# and use evaluate_model function to test with test set
X_train_undersample_non_key = X_train_undersample.drop(columns = 'msisdn')
scores_under = []
for idx, model in enumerate(models):
    print("Model: {}".format(models_name[idx]))
    # Your code here
    model.fit(X_train_undersample_non_key, y_train_undersample)
    scores_under.append(evaluate_model(model, X_test_data, y_test, 'under'))
    print("=======================================")
df = pd.DataFrame(scores_origin)
df.head()
# Now we will test on Oversampled dataset (X_train_oversample, y_train_oversample)
# We loop for models
# For each model, we train with train_oversample dataset
# and use evaluate_model function to test with test set
X_train_oversample_non_key = X_train_oversample.drop(columns = 'msisdn')
scores_over = []
for idx, model in enumerate(models):
    print("Model: {}".format(models_name[idx]))
    model.fit(X_train_oversample_non_key, y_train_oversample)
    scores_over.append(evaluate_model(model, X_test_data, y_test, 'oversample'))

    print("=======================================")
# Uncomment this code if you didn't install imblearn
# !conda install -c conda-forge imbalanced-learn -y
X_train_data =X_train.drop(columns = 'msisdn')
from imblearn.over_sampling import SMOTE
sm = SMOTE()
X_res, y_res = sm.fit_resample(X_train_data, y_train)
from imblearn.over_sampling import ADASYN 
adasyn = ADASYN()
X_ada, y_ada = adasyn.fit_resample(X_train_data, y_train)
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler()
X_rus, y_rus = rus.fit_resample(X_train_data, y_train)
from imblearn.combine import SMOTEENN
smenn = SMOTEENN()
X_smenn, y_smenn = smenn.fit_resample(X_train_data, y_train)
scores = []

for idx, model in enumerate(models):
    print("Model: {}".format(models_name[idx]))
    model.fit(X_res, y_res)
    scores.append(evaluate_model(model, X_test_data, y_test, 'smote'))
    # Your code here
    print("=======================================")

for idx, model in enumerate(models):
    print("Model: {}".format(models_name[idx]))
    model.fit(X_ada, y_ada)
    scores.append(evaluate_model(model, X_test_data, y_test, 'ada'))
    # Your code here
    print("=======================================")

for idx, model in enumerate(models):
    print("Model: {}".format(models_name[idx]))
    model.fit(X_rus, y_rus)
    scores.append(evaluate_model(model, X_test_data, y_test, 'rus'))
    # Your code here
    print("=======================================")    

for idx, model in enumerate(models):
    print("Model: {}".format(models_name[idx]))
    model.fit(X_smenn, y_smenn)
    scores.append(evaluate_model(model, X_test_data, y_test, 'smoteenn'))
    # Your code here
    print("=======================================")    
df_imb = pd.DataFrame(scores)
df_under = pd.DataFrame(scores_under)
df_over  = pd.DataFrame(scores_over)
df_origin = pd.DataFrame(scores_origin)

df_all = pd.concat([df_imb, df_under, df_over, df_origin])
df_all
df_all.sort_values('recall', inplace=True)
for label, df in df_all.groupby('name'):
    df.plot(x='description', kind='barh', title=label, figsize=(8, 4), xlim=(0,1))
# Import Pipeline and GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
step = []
# append a step 'cls' with value is LogisticRegression to step variable
# Your code here
step.append(['cls', LogisticRegression()])
# Create Pipeline with defined step
ppl = Pipeline(step)
# Define params grid, gridsearch cv go throuh each item in params_grid
# For each item, it changes param of Pipeline base on "key" of item
params_grid = {
    "cls": models
}
from sklearn.metrics import roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix
# Define metrics to evaluate model
scorers = {
    'recall_score': make_scorer(recall_score),
    'precision_score': make_scorer(precision_score),
    'accuracy_score': make_scorer(accuracy_score),
    'auc': make_scorer(auc)
}
# Create GridSearchCV with Pipeline as estimator and params_grid
gridcv = GridSearchCV(ppl, param_grid=params_grid, scoring=scorers, refit='recall_score', return_train_score=True, verbose=5)
# train model as a normal model with fit
gridcv.fit(X_train_oversample_non_key, y_train_oversample)
results = pd.DataFrame(gridcv.cv_results_)
results = results.sort_values(by='mean_test_precision_score', ascending=False)
results[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'param_cls']].round(3).head()
results[['split0_test_auc', 'split1_test_auc', 'split2_test_auc',
       'split3_test_auc', 'split4_test_auc', 'mean_test_auc', 'std_test_auc',
       'rank_test_auc', 'split0_train_auc', 'split1_train_auc',
       'split2_train_auc', 'split3_train_auc', 'split4_train_auc',
       'mean_train_auc', 'std_train_auc']]
df_test
answer = gridcv.predict(df_test.drop(columns = ['msisdn','label']))
ans = pd.DataFrame(data=answer, columns=["label"])
submission = pd.concat([df_test.msisdn, ans], axis = 1)
submission[submission.label==1]
submission.to_csv('submission.csv', index= False, index_label=False)
