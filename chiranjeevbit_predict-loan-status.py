import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from pylab import plot, show, subplot, specgram, imshow, savefig
from sklearn import preprocessing
from sklearn import cross_validation, metrics
from sklearn.preprocessing import Normalizer
from sklearn.cross_validation import cross_val_score
%matplotlib inline
data_train = pd.read_csv("../input/train_u6lujuX_CVtuZ9i.csv") 

data_test = pd.read_csv("../input/test_Y3wMUE5_7gLdaTN.csv") 
data_train.shape
data_test.shape
data_train.head(5)
data_train.describe()
data_test.head()
data_test.describe()
data_train.isnull().sum()
test.isnull().sum()
train = pd.read_csv("../input/train_u6lujuX_CVtuZ9i.csv") 
test = pd.read_csv("../input/test_Y3wMUE5_7gLdaTN.csv") 
targets = train.Loan_Status

train.drop('Loan_Status', 1, inplace=True)
combined = train.append(test)
combined.reset_index(inplace=True)
combined.drop(['index', 'Loan_ID'], inplace=True, axis=1)
combined.head(5)
combined.shape
combined.describe()
# To check how many columns have missing values - this can be repeated to see the progress made
def show_missing():
    missing = combined.columns[combined.isnull().any()].tolist()
    return missing

#from this we can find the total missing data in each columns

combined[show_missing()].isnull().sum()
print (combined['Property_Area'].value_counts())
print (combined['Education'].value_counts())
print (combined['Gender'].value_counts())
print (combined['Dependents'].value_counts())
print (combined['Married'].value_counts())
print (combined['Self_Employed'].value_counts())
print (combined['Credit_History'].value_counts())
#filling data with approperiate measure of central tendency
combined['Gender'].fillna('Male', inplace=True)
combined['Married'].fillna('Yes', inplace=True)

combined['Self_Employed'].fillna('Yes', inplace=True)

combined['Credit_History'].fillna(1, inplace=True)

combined['LoanAmount'].fillna(combined['LoanAmount'].median(), inplace=True)
#combined['Loan_Amount_Term'].fillna(combined['Loan_Amount_Term'].mean(), inplace=True)

combined.isnull().sum()
combined['ApplicantIncome'].hist()
combined['LoanAmount'].hist()
combined['Loan_Amount_Term'].hist()
ax = combined.groupby('Gender').ApplicantIncome.mean().plot(kind='bar')
ax.set_xlabel("Gender")
ax.set_ylabel("mean ApplicantIncom")
ax = combined.groupby('Education').ApplicantIncome.mean().plot(kind='bar')
ax.set_xlabel("Education(1=Graduate)")
ax.set_ylabel("mean ApplicantIncom")
ax = combined.groupby('Married').ApplicantIncome.mean().plot(kind='bar')
ax.set_xlabel("Married(1=yes)")
ax.set_ylabel("mean ApplicantIncom")

temp = pd.crosstab(data_train['Credit_History'], data_train['Loan_Status'])
temp.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
temp3 = pd.crosstab(data_train['Dependents'], data_train['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
temp3 = pd.crosstab(data_train['Gender'], data_train['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
temp3 = pd.crosstab(data_train['Education'], data_train['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
temp3 = pd.crosstab(data_train['Property_Area'], data_train['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
combined['Gender'] = combined['Gender'].map({'Male':1,'Female':0})
combined['Married'] = combined['Married'].map({'Yes':1,'No':0})
combined['Education'] = combined['Education'].map({'Graduate':1,'Not Graduate':0})
combined['Self_Employed'] = combined['Self_Employed'].map({'Yes':1,'No':0})
combined['Singleton'] = combined['Dependents'].map(lambda d: 1 if d=='1' else 0)
combined['Small_Family'] = combined['Dependents'].map(lambda d: 1 if d=='2' else 0)
combined['Large_Family'] = combined['Dependents'].map(lambda d: 1 if d=='3+' else 0)
combined.drop(['Dependents'], axis=1, inplace=True)
combined['Total_Income'] = combined['ApplicantIncome'] + combined['CoapplicantIncome']
combined.drop(['ApplicantIncome','CoapplicantIncome'], axis=1, inplace=True)

combined['Income_Ratio'] = combined['Total_Income'] / combined['LoanAmount']

combined['Loan_Amount_Term'].value_counts()
approved_term = data_train[data_train['Loan_Status']=='Y']['Loan_Amount_Term'].value_counts()
unapproved_term = data_train[data_train['Loan_Status']=='N']['Loan_Amount_Term'].value_counts()
df = pd.DataFrame([approved_term,unapproved_term])
df.index = ['Approved','Unapproved']
df.plot(kind='bar', stacked=True, figsize=(15,8))
combined['Very_Short_Term'] = combined['Loan_Amount_Term'].map(lambda t: 1 if t<=60 else 0)
combined['Short_Term'] = combined['Loan_Amount_Term'].map(lambda t: 1 if t>60 and t<180 else 0)
combined['Long_Term'] = combined['Loan_Amount_Term'].map(lambda t: 1 if t>=180 and t<=300  else 0)
combined['Very_Long_Term'] = combined['Loan_Amount_Term'].map(lambda t: 1 if t>300 else 0)
combined.drop('Loan_Amount_Term', axis=1, inplace=True)

combined['Credit_History_Bad'] = combined['Credit_History'].map(lambda c: 1 if c==0 else 0)
combined['Credit_History_Good'] = combined['Credit_History'].map(lambda c: 1 if c==1 else 0)
combined['Credit_History_Unknown'] = combined['Credit_History'].map(lambda c: 1 if c==2 else 0)
combined.drop('Credit_History', axis=1, inplace=True)
property_dummies = pd.get_dummies(combined['Property_Area'], prefix='Property')
combined = pd.concat([combined, property_dummies], axis=1)
combined.drop('Property_Area', axis=1, inplace=True)
combined[60:70]
def feature_scaling(dataframe):
    dataframe -= dataframe.min()
    dataframe /= dataframe.max()
    return dataframe
combined['LoanAmount'] = feature_scaling(combined['LoanAmount'])
combined['Total_Income'] = feature_scaling(combined['Total_Income'])
combined['Income_Ratio'] = feature_scaling(combined['Income_Ratio'])
combined.head()
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
#function for computing score
def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)
#recovering train test &target
global combined, data_train
targets = data_train['Loan_Status'].map({'Y':1,'N':0})
train = combined.head(614)
test = combined.iloc[614:]

clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train, targets)
features = pd.DataFrame()
features['Feature'] = train.columns
features['Importance'] = clf.feature_importances_
features.sort_values(by=['Importance'], ascending=False, inplace=True)
features.set_index('Feature', inplace=True)
features.plot(kind='bar', figsize=(20, 10))
model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(train)
train_reduced.shape
test_reduced = model.transform(test)
test_reduced.shape
parameters  = {'bootstrap': False,
              'min_samples_leaf': 3,
              'n_estimators': 50,
              'min_samples_split': 10,
              'max_features': 'sqrt',
              'max_depth': 6}

model = RandomForestClassifier(**parameters)
model.fit(train, targets)
compute_score(model, train, targets, scoring='accuracy')
#saving output as output.csv
output = model.predict(test).astype(int)
df_output = pd.DataFrame()
aux = pd.read_csv('../input/test_Y3wMUE5_7gLdaTN.csv')
df_output['Loan_ID'] = aux['Loan_ID']
df_output['Loan_Status'] = np.vectorize(lambda s: 'Y' if s==1 else 'N')(output)
df_output[['Loan_ID','Loan_Status']].to_csv('output.csv',index=False)
####Prediction model########
#Train-Test split
from sklearn.model_selection import train_test_split
datatrain, datatest, labeltrain, labeltest = train_test_split(train, targets, test_size = 0.2, random_state = 42)
labeltrain.shape
#Logistic Regression
from sklearn.linear_model import LogisticRegression
logis = LogisticRegression()
logis.fit(datatrain, labeltrain)
logis_score_train = logis.score(datatrain, labeltrain)
print("Training score: ",logis_score_train)
logis_score_test = logis.score(datatest, labeltest)
print("Testing score: ",logis_score_test)
#saving output as output.csv of decision tree
output2 = logis.predict(test).astype(int)
df_output2 = pd.DataFrame()
aux = pd.read_csv('../input/test_Y3wMUE5_7gLdaTN.csv')
df_output2['Loan_ID'] = aux['Loan_ID']
df_output2['Loan_Status'] = np.vectorize(lambda s: 'Y' if s==1 else 'N')(output2)
df_output2[['Loan_ID','Loan_Status']].to_csv('output2.csv',index=False)

#decision tree
from sklearn.ensemble import RandomForestClassifier
dt = RandomForestClassifier()
dt.fit(datatrain, labeltrain)
dt_score_train = dt.score(datatrain, labeltrain)
print("Training score: ",dt_score_train)
dt_score_test = dt.score(datatest, labeltest)
print("Testing score: ",dt_score_test)
#saving output as output.csv of decision tree
#output2 = dt.predict(test).astype(int)
#df_output2 = pd.DataFrame()
#aux = pd.read_csv('../input/test_Y3wMUE5_7gLdaTN.csv')
#df_output2['Loan_ID'] = aux['Loan_ID']
#df_output2['Loan_Status'] = np.vectorize(lambda s: 'Y' if s==1 else 'N')(output2)
#df_output2[['Loan_ID','Loan_Status']].to_csv('output2.csv',index=False)

#random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(datatrain, labeltrain)
rfc_score_train = rfc.score(datatrain, labeltrain)
print("Training score: ",rfc_score_train)
rfc_score_test = rfc.score(datatest, labeltest)
print("Testing score: ",rfc_score_test)
#Model comparison
models = pd.DataFrame({
        'Model'          : ['Logistic Regression',  'Decision Tree', 'Random Forest'],
        'Training_Score' : [logis_score_train,  dt_score_train, rfc_score_train],
        'Testing_Score'  : [logis_score_test, dt_score_test, rfc_score_test]
    })
models.sort_values(by='Testing_Score', ascending=False)


