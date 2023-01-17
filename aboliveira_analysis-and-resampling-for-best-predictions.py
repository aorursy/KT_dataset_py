import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Warnings
import warnings
warnings.filterwarnings('ignore')

sns.set(style='darkgrid')
plt.rcParams["patch.force_edgecolor"] = True
df = pd.read_csv(r"../input/WA_Fn-UseC_-Telco-Customer-Churn.csv")
# First 5 Rows:
df.head(5)
print(df.info())
# Check for Null Values:
df.isnull().any()

# Null values in TotalCharges must be dealt with:

df['TotalCharges_new']= pd.to_numeric(df.TotalCharges,errors='coerce_numeric')
TotalCharges_Missing=[488,753,936,1082,1340,3331,3826,4380,5218,6670,6754]
df.loc[pd.isnull(df.TotalCharges_new),'TotalCharges_new']=TotalCharges_Missing
df.TotalCharges=df.TotalCharges_new
df.drop('TotalCharges_new',axis=1,inplace=True)

# Converting 'TotalCharges' column from object type to float type

df['TotalCharges'] = df['TotalCharges'].convert_objects(convert_numeric=True)
# Adding a feature for the total amount of services

df['TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)
sns.heatmap(df.apply(lambda x: pd.factorize(x)[0]).corr(), cmap='Blues')
plt.show()
plt.style.use(['seaborn-dark','seaborn-talk'])

fig, ax = plt.subplots(1,2,figsize=(16,6))

df['Churn'].value_counts().plot.pie(explode=[0,0.08], ax=ax[0], autopct='%1.2f%%', shadow=True, 
                                    fontsize=14, startangle=30, colors=["#3791D7", "#D72626"])
ax[0].set_title('Total Churn Percentage')

sns.countplot('Churn', data=df, ax=ax[1], palette=["#3791D7", "#D72626"])
ax[1].set_title('Total Number of Churn Customers')
ax[1].set_ylabel(' ')

plt.show()
plt.style.use(['seaborn-dark','seaborn-talk'])

fig, ax = plt.subplots(1,2,figsize=(16,6))

sns.boxplot(x='Churn', y='TotalCharges', data=df, ax=ax[0], palette=["#3791D7", "#D72626"])
ax[0].set_title('Total Charges')
ax[0].set_ylabel('Total Charges ($)')
ax[0].set_label('Churn')

sns.boxplot(x='Churn', y='MonthlyCharges', data=df, ax=ax[1], palette=["#3791D7", "#D72626"])
ax[1].set_title('Monthly Charges')
ax[1].set_ylabel('Monthly Charges ($)')
ax[1].set_label('Churn')

plt.show()
plt.style.use(['bmh','seaborn-talk'])
plt.figure(figsize=(14,6))

sns.countplot(x='TotalServices', hue='Churn', data=df)
plt.title('Number of Customers per Number of Services')
plt.xlabel('Number of Online Services')
plt.ylabel('Number of Customers')

plt.show()
cols = ['PhoneService', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

for i in cols:
    plt.figure(figsize=(14,4))
    sns.countplot(x=i, hue='Churn', data=df)
    plt.ylabel('Number of Customers')
    plt.show()
plt.style.use(['seaborn-dark','seaborn-talk'])
fig, ax = plt.subplots(1,2,figsize=(16,6))

sns.countplot(x='Contract', data=df, hue='Churn', ax=ax[0])
ax[0].set_title('Number of Customers per Contract Type')
ax[0].set_xlabel('Contract Type')
ax[0].set_ylabel('Number of Customers')

sns.countplot(x='PaymentMethod', data=df, hue='Churn', ax=ax[1])
ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation = 30)
ax[1].set_title('Number of Customers per Payment Method')
ax[1].set_xlabel('Payment Method')
ax[1].set_ylabel('Number of Customers')

plt.show()
plt.figure(figsize=(14,6))

sns.boxplot(x='Churn', y='tenure', data=df)
plt.title('Tenure by Churn')
plt.xlabel('Churn')
plt.ylabel('Tenure')

plt.show()
for i in ['gender', 'SeniorCitizen', 'Partner', 'Dependents']:
    plt.figure(figsize=(14,6))
    sns.countplot(x=i, data=df, hue='Churn')
for i in ['OnlineSecurity','OnlineBackup','DeviceProtection',
          'TechSupport','StreamingTV','StreamingMovies']:
    df[i] = df[i].apply(lambda x: 'No' if x=='No internet service' else x)
    
df.MultipleLines=df.MultipleLines.apply(lambda x: 'No' if x=='No phone service' else x)
    
# For variables with only two classifications:

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in [e for e in df.columns if len(df[e].unique())==2]:
    df[i] = le.fit_transform(df[i])
# For variables with more than two classifications:

df = pd.get_dummies(df, columns = [i for i in df.columns if df[i].dtypes=='object'], drop_first=True)
from sklearn.utils import resample

df_majority = df[df.Churn==0]
df_minority = df[df.Churn==1]

df_minority_upsampled = resample(df_minority,
                                replace=True,
                                n_samples=5000,
                                random_state=123)

df_upsampled = pd.concat([df_majority, df_minority_upsampled])

print('Churn Count in Original Data: \n', df.Churn.value_counts(), '\n')
print('New Churn Count: \n', df_upsampled.Churn.value_counts())
from sklearn.model_selection import train_test_split

# Separate input features (X) and target variable (y)
y = df_upsampled.Churn
X = df_upsampled.drop('Churn', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=123, test_size=0.30)
from sklearn.metrics import classification_report, precision_score, accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

classifier_list = [ LogisticRegression(),
                    KNeighborsClassifier(),
                    GaussianNB(priors=None),
                    RandomForestClassifier()]

for clf in classifier_list:
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    precision = precision_score(y_test, predictions) 
    accuracy = accuracy_score(y_test, predictions)
    
    
# Precision_score = tp / (tp + fp)
# Accuracy_score = (# of correctly assigned rows) / (All rows)

    print(clf, '\n \n',classification_report(y_test, predictions), 
          '\n \nPrecision Score: ' , precision,
          '\nAccuracy Score: ', accuracy,
          '\n\n----------------------------------------------------------------\n\n')
