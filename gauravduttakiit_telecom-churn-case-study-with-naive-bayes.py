# Suppressing Warnings

import warnings

warnings.filterwarnings('ignore')
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Importing Pandas and NumPy

import pandas as pd, numpy as np, seaborn as sns,matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
# Importing all datasets

churn_data = pd.read_csv("/kaggle/input/telecom-churn-data-sets/churn_data.csv")

churn_data.head()
customer_data = pd.read_csv("/kaggle/input/telecom-churn-data-sets/customer_data.csv")

customer_data.head()
internet_data = pd.read_csv("/kaggle/input/telecom-churn-data-sets/internet_data.csv")

internet_data.head()
# Merging on 'customerID'

df_1 = pd.merge(churn_data, customer_data, how='inner', on='customerID')
# Final dataframe with all predictor variables

telecom = pd.merge(df_1, internet_data, how='inner', on='customerID')
# Let's see the head of our master dataset

telecom.head()
# Let's check the dimensions of the dataframe

telecom.shape
# let's look at the statistical aspects of the dataframe

telecom.describe()
# Let's see the type of each column

telecom.info()
#The varaible was imported as a string we need to convert it to float

# telecom['TotalCharges'] = telecom['TotalCharges'].astype(float) 

telecom.TotalCharges = pd.to_numeric(telecom.TotalCharges, errors='coerce')
telecom.info()


plt.figure(figsize=(20,40))

plt.subplot(10,2,1)

ax = sns.distplot(telecom['tenure'], hist=True, kde=False, 

             bins=int(180/5), color = 'darkblue', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})

ax.set_ylabel('# of Customers')

ax.set_xlabel('Tenure (months)')

plt.subplot(10,2,2)

ax = sns.countplot(x='PhoneService', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,3)

ax =sns.countplot(x='Contract', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,3)

ax =sns.countplot(x='Contract', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,4)

ax =sns.countplot(x='PaperlessBilling', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,5)

ax =sns.countplot(x='PaymentMethod', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,6)

ax =sns.countplot(x='Churn', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,7)

ax =sns.countplot(x='gender', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,8)

ax =sns.countplot(x='SeniorCitizen', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,9)

ax =sns.countplot(x='Partner', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,10)

ax =sns.countplot(x='Dependents', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,11)

ax =sns.countplot(x='MultipleLines', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,12)

ax =sns.countplot(x='InternetService', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,13)

ax =sns.countplot(x='OnlineSecurity', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,14)

ax =sns.countplot(x='OnlineBackup', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,15)

ax =sns.countplot(x='DeviceProtection', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,16)

ax =sns.countplot(x='TechSupport', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,17)

ax =sns.countplot(x='StreamingTV', data=telecom)

ax.set_ylabel('# of Customers')



plt.subplot(10,2,18)

ax =sns.countplot(x='StreamingMovies', data=telecom)

ax.set_ylabel('# of Customers')

plt.subplot(10,2,19)

ax = sns.distplot(telecom['MonthlyCharges'], hist=True, kde=False, 

             bins=int(180/5), color = 'darkblue', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})

ax.set_ylabel('# of Customers')

ax.set_xlabel('MonthlyCharges')

plt.subplot(10,2,20)

ax = sns.distplot(telecom['TotalCharges'], hist=True, kde=False, 

             bins=int(180/5), color = 'darkblue', 

             hist_kws={'edgecolor':'black'},

             kde_kws={'linewidth': 4})

ax.set_ylabel('# of Customers')

ax.set_xlabel('TotalCharges');
sns.pairplot(telecom)

plt.show()
plt.figure(figsize=(25, 10))

plt.subplot(1,3,1)

sns.boxplot(x = 'tenure', y = 'Churn', data=telecom)

plt.subplot(1,3,2)

sns.boxplot(x = 'MonthlyCharges', y = 'Churn', data=telecom)

plt.subplot(1,3,3)

sns.boxplot(x = 'TotalCharges', y = 'Churn', data=telecom)

plt.show()
# List of variables to map



varlist =  ['PhoneService', 'PaperlessBilling', 'Churn', 'Partner', 'Dependents']



# Defining the map function

def binary_map(x):

    return x.map({'Yes': 1, "No": 0})



# Applying the function to the housing list

telecom[varlist] = telecom[varlist].apply(binary_map)
telecom.head()
# Creating a dummy variable for some of the categorical variables and dropping the first one.

dummy1 = pd.get_dummies(telecom[['Contract', 'PaymentMethod', 'gender', 'InternetService']], drop_first=True)



# Adding the results to the master dataframe

telecom = pd.concat([telecom, dummy1], axis=1)
telecom.head()
# Creating dummy variables for the remaining categorical variables and dropping the level with big names.



# Creating dummy variables for the variable 'MultipleLines'

ml = pd.get_dummies(telecom['MultipleLines'], prefix='MultipleLines')

# Dropping MultipleLines_No phone service column

ml1 = ml.drop(['MultipleLines_No phone service'], 1)

#Adding the results to the master dataframe

telecom = pd.concat([telecom,ml1], axis=1)



# Creating dummy variables for the variable 'OnlineSecurity'.

os = pd.get_dummies(telecom['OnlineSecurity'], prefix='OnlineSecurity')

os1 = os.drop(['OnlineSecurity_No internet service'], 1)

# Adding the results to the master dataframe

telecom = pd.concat([telecom,os1], axis=1)



# Creating dummy variables for the variable 'OnlineBackup'.

ob = pd.get_dummies(telecom['OnlineBackup'], prefix='OnlineBackup')

ob1 = ob.drop(['OnlineBackup_No internet service'], 1)

# Adding the results to the master dataframe

telecom = pd.concat([telecom,ob1], axis=1)



# Creating dummy variables for the variable 'DeviceProtection'. 

dp = pd.get_dummies(telecom['DeviceProtection'], prefix='DeviceProtection')

dp1 = dp.drop(['DeviceProtection_No internet service'], 1)

# Adding the results to the master dataframe

telecom = pd.concat([telecom,dp1], axis=1)



# Creating dummy variables for the variable 'TechSupport'. 

ts = pd.get_dummies(telecom['TechSupport'], prefix='TechSupport')

ts1 = ts.drop(['TechSupport_No internet service'], 1)

# Adding the results to the master dataframe

telecom = pd.concat([telecom,ts1], axis=1)



# Creating dummy variables for the variable 'StreamingTV'.

st =pd.get_dummies(telecom['StreamingTV'], prefix='StreamingTV')

st1 = st.drop(['StreamingTV_No internet service'], 1)

# Adding the results to the master dataframe

telecom = pd.concat([telecom,st1], axis=1)



# Creating dummy variables for the variable 'StreamingMovies'. 

sm = pd.get_dummies(telecom['StreamingMovies'], prefix='StreamingMovies')

sm1 = sm.drop(['StreamingMovies_No internet service'], 1)

# Adding the results to the master dataframe

telecom = pd.concat([telecom,sm1], axis=1)
telecom.head()
# We have created dummies for the below variables, so we can drop them

telecom = telecom.drop(['Contract','PaymentMethod','gender','MultipleLines','InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',

       'TechSupport', 'StreamingTV', 'StreamingMovies'], 1)
# Checking for outliers in the continuous variables

num_telecom = telecom[['tenure','MonthlyCharges','SeniorCitizen','TotalCharges']]
# Checking outliers at 25%, 50%, 75%, 90%, 95% and 99%

num_telecom.describe(percentiles=[.25, .5, .75, .90, .95, .99])
# Adding up the missing values (column-wise)

telecom.isnull().sum()
print('No. of Null Records for TotalCharges:',telecom.TotalCharges.isnull().sum())
print('No. of Records for TotalCharges:',len(telecom))
print('No. of non Records for TotalCharges:',len(telecom)-telecom.TotalCharges.isnull().sum())
# Checking the percentage of missing values

round(100*(telecom.isnull().sum()/len(telecom.index)), 2)
telecom = telecom.dropna()

telecom = telecom.reset_index(drop=True)



# Checking percentage of missing values after removing the missing values

round(100*(telecom.isnull().sum()/len(telecom.index)), 2)
from sklearn.model_selection import train_test_split
# Putting feature variable to X

X = telecom.drop(['Churn','customerID'], axis=1)



X.head()
# Putting response variable to y

y = telecom['Churn']



y.head()
# Splitting the data into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



X_train[['tenure','MonthlyCharges','TotalCharges']] = scaler.fit_transform(X_train[['tenure','MonthlyCharges','TotalCharges']])



X_train.head()
X_test[['tenure','MonthlyCharges','TotalCharges']] = scaler.transform(X_test[['tenure','MonthlyCharges','TotalCharges']])



X_test.head()
### Checking the Churn Rate

churn = (sum(telecom['Churn'])/len(telecom['Churn'].index))*100

churn
# Importing matplotlib and seaborn

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Let's see the correlation matrix 

plt.figure(figsize = (25,25))        # Size of the figure

sns.heatmap(telecom.corr(),annot = True,cmap="tab20c")

plt.show()
plt.figure(figsize=(10,8))

telecom.corr()['Churn'].sort_values(ascending = False).plot(kind='bar');
X_test = X_test.drop(['MultipleLines_No','OnlineSecurity_No','OnlineBackup_No','DeviceProtection_No','TechSupport_No',

                       'StreamingTV_No','StreamingMovies_No'], 1)

X_train = X_train.drop(['MultipleLines_No','OnlineSecurity_No','OnlineBackup_No','DeviceProtection_No','TechSupport_No',

                         'StreamingTV_No','StreamingMovies_No'], 1)
plt.figure(figsize = (25,25))

sns.heatmap(X_train.corr(),annot = True,cmap="tab20c")

plt.show()
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

model = GaussianNB()
# fit the model with the training data

model.fit(X_train,y_train)
# predict the target on the train dataset

predict_train = model.predict(X_train)

predict_train
trainaccuracy = accuracy_score(y_train,predict_train)

print('accuracy_score on train dataset : ', trainaccuracy)
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif.tail()
features_to_remove = vif.loc[vif['VIF'] >= 4.99,'Features'].values

features_to_remove = list(features_to_remove)

print(features_to_remove)
X_train = X_train.drop(columns=features_to_remove, axis = 1)

X_train.head()
X_test = X_test.drop(columns=features_to_remove, axis = 1)

X_test.head()
# fit the model with the training data

model.fit(X_train,y_train)
# predict the target on the train dataset

predict_train = model.predict(X_train)

predict_train
trainaccuracy = accuracy_score(y_train,predict_train)

print('accuracy_score on train dataset : ', trainaccuracy)
# Check for the VIF values of the feature variables. 

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
from sklearn import metrics

# Confusion matrix 

confusion = metrics.confusion_matrix(y_train, predict_train )

print(confusion)

TP = confusion[1,1] # true positive 

TN = confusion[0,0] # true negatives

FP = confusion[0,1] # false positives

FN = confusion[1,0] # false negatives
# Let's see the sensitivity of our model

trainsensitivity= TP / float(TP+FN)

trainsensitivity
# Let us calculate specificity

trainspecificity= TN / float(TN+FP)

trainspecificity
# Calculate false postive rate - predicting churn when customer does not have churned

print(FP/ float(TN+FP))
# Positive predictive value 

print (TP / float(TP+FP))
def draw_roc( actual, probs ):

    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,

                                              drop_intermediate = False )

    auc_score = metrics.roc_auc_score( actual, probs )

    plt.figure(figsize=(5, 5))

    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()



    return None
draw_roc(y_train,predict_train)
#Looking at the confusion matrix again
from sklearn.metrics import precision_score, recall_score

precision_score(y_train,predict_train)
recall_score(y_train,predict_train)
# predict the target on the test dataset

predict_test = model.predict(X_test)

print('Target on test data\n\n',predict_test)
confusion2 = metrics.confusion_matrix(y_test, predict_test )

print(confusion2)
# Let's check the overall accuracy.

testaccuracy= accuracy_score(y_test,predict_test)

testaccuracy
# Let's see the sensitivity of our lmodel

testsensitivity=TP / float(TP+FN)

testsensitivity
# Let us calculate specificity

testspecificity= TN / float(TN+FP)

testspecificity
# Let us compare the values obtained for Train & Test:

print("Train Data Accuracy    :{} %".format(round((trainaccuracy*100),2)))

print("Train Data Sensitivity :{} %".format(round((trainsensitivity*100),2)))

print("Train Data Specificity :{} %".format(round((trainspecificity*100),2)))

print("Test Data Accuracy     :{} %".format(round((testaccuracy*100),2)))

print("Test Data Sensitivity  :{} %".format(round((testsensitivity*100),2)))

print("Test Data Specificity  :{} %".format(round((testspecificity*100),2)))