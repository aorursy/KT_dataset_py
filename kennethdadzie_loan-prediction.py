# Ignore harmless warnings

import warnings

warnings.filterwarnings("ignore")



#Importing libraries for data analysis and cleaning

import numpy as np

import pandas as pd



#importing visualisation libraries for data visualisation

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px



#load datasets

train = pd.read_csv("../input/analytics-vidhya-loan-prediction/train.csv")

test = pd.read_csv("../input/analytics-vidhya-loan-prediction/test.csv")



#per describtion, loan amount is in 1000's

train['LoanAmount'] = train['LoanAmount'] *1000

test['LoanAmount'] = test['LoanAmount'] *1000
#Observing the first five rows of the dataset for training

train.head()
# Function to calculate missing values by column#

def missing_values_table(df):

        # Total missing values

        mis_val = df.isnull().sum()

        

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Print some summary information

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        # Return the dataframe with missing information

        return mis_val_table_ren_columns
missing_values_table(train)
#check for duplicated data

train.duplicated().sum()
#checking the distribution of the target column (Loan Status)

plt.figure(figsize=(8,5))

plt.title('Loan Status Count')

sns.countplot(data=train,x='Loan_Status');
#checking the distribution of the target column (Loan Status)

px.pie(data_frame=train,names='Loan_Status',title='Distribution of Loan Status')
#Checking the statistical info of the dataset(train)

train.describe()
#Check for the distribution of loan amount

train['LoanAmount'].plot(kind='hist',figsize=(13,8),bins=50,edgecolor='k',

                         title='Distribution of Loan Amount').autoscale(axis='x',tight=True)
train.boxplot(column='LoanAmount',figsize=(4,7));
#Check for the distribution of loan amount

train['ApplicantIncome'].plot(kind='hist',figsize=(13,8),bins=50,edgecolor='k',

                              title='Applicant Income Distribution').autoscale(axis='x',tight=True)
#median of loan amounts applied for

train['ApplicantIncome'].median()
#Check for the distribution of loan amount

train['CoapplicantIncome'].plot(kind='hist',figsize=(13,8),bins=50,edgecolor='k',

                                title='Distribution of Coapplicant Income').autoscale(axis='x',tight=True)
#Median of coapplicant income

train['CoapplicantIncome'].median()
pd.DataFrame(train.groupby(['Property_Area','Loan_Status'])['Loan_Status'].count())
plt.figure(figsize=(8,5))

sns.countplot(data=train,x='Property_Area',hue='Loan_Status');
pd.DataFrame(train.groupby(['Property_Area','Education'])['Loan_Status'].count())
a = pd.crosstab(train['Property_Area'],[train['Education'],train['Loan_Status']])

a.plot(kind='bar',stacked=True,figsize=(12,7),legend=True,title='Loan status based on education and property type').legend(loc=3, bbox_to_anchor=(1.0,0.1));
#Checking the barchart distribution of loan status in relation with educational status

pd.crosstab(train['Education'],train['Loan_Status']).plot(kind='bar',figsize=(12,6));
pd.DataFrame(train.groupby(['Education','Credit_History','Loan_Status'])['Loan_Status'].count())
b = pd.crosstab(train['Education'],[train['Credit_History'],train['Loan_Status']])

b.plot(kind='bar',stacked=True,figsize=(10,5),title='Distribution of education in relation with credit history and loan status');
#Grouping loan_status by credit history

pd.DataFrame(train.groupby(['Credit_History','Loan_Status'])['Loan_Status'].count())
#Graphical representaion

c = pd.crosstab(train['Credit_History'],train['Loan_Status'])

c.plot(kind='bar',figsize=(10,6));
#Grouping loan status by education,self employed and credit history

pd.DataFrame(train.groupby(['Education','Self_Employed','Credit_History'])['Loan_Status'].count())
#Barchart represnting eduaction,self-employment,and credit historys impact on loan status

d = pd.crosstab(train['Education'],[train['Self_Employed'],train['Credit_History'],train['Loan_Status']])

d.plot(kind='bar',stacked=True,figsize=(10,6),legend=True);
f = pd.crosstab(train['Gender'],[train['Loan_Status'],train['Credit_History']])

f.plot(kind='bar',stacked=True,figsize=(12,6));
#Grouping loan status by Gender, and credit history

pd.DataFrame(train.groupby(['Gender','Credit_History','Loan_Status'])['Loan_Status'].count())
print('Total count of male applicants are', len(train[train['Gender'] == 'Male']))

print('Total count of female applicants are', len(train[train['Gender'] == 'Female']))
sns.countplot(data=train,x='Married',hue='Loan_Status');
round(100 * train.groupby('Married')['Loan_Status'].value_counts(normalize=True))
print('Total count of married applicants are', len(train[train['Married'] == 'Yes']))

print('Total count of single applicants are', len(train[train['Married'] == 'No']))
plt.figure(figsize=(12,5))

sns.scatterplot(data=train,x='ApplicantIncome',y='LoanAmount');
plt.figure(figsize=(12,5))

sns.scatterplot(data=train,x='CoapplicantIncome',y='LoanAmount');
px.scatter(train,x='ApplicantIncome',y='LoanAmount',color='Loan_Status',title='LOAN STATUS BASED ON APPLICANT INCOME')
px.scatter(train,x='ApplicantIncome',y='LoanAmount',color='Credit_History',facet_col='Loan_Status',

           color_continuous_scale=["red", "green", "blue"],title='LOAN STATUS BASED ON APPLICANTINCOME AND CREDIT HISTORY')
px.scatter(train,y='LoanAmount',x='CoapplicantIncome',color='Loan_Status',title='LOAN STATUS BASED ON COAPPLICANT INCOME')
px.scatter(train,x='CoapplicantIncome',y='LoanAmount',color='Credit_History',facet_col='Loan_Status',

           color_continuous_scale=["red", "green", "blue"],title='LOAN STATUS BASED ON COAPPLICANTINCOME AND CREDIT HISTORY')
train['Dependents'].value_counts()
temp = train.dropna()
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

temp['Loan_Status'] = encoder.fit_transform(temp['Loan_Status'])

temp['Property_Area'] = encoder.fit_transform(temp['Property_Area'])

temp['Education'] = encoder.fit_transform(temp['Education'])
temp['Dependents'] = encoder.fit_transform(temp['Dependents'])

temp['Married'] = encoder.fit_transform(temp['Married'])

temp['Self_Employed'] = encoder.fit_transform(temp['Self_Employed'])

temp['Gender'] = encoder.fit_transform(temp['Gender'])
#Correlations in data

plt.figure(figsize=(12,7))

ax = sns.heatmap(temp.corr(),cmap='coolwarm',annot=True,vmax=1,vmin=-1);

# fix for mpl bug that cuts off top/bottom of seaborn viz

b, t = plt.ylim() # discover the values for bottom and top

b += 0.5 # Add 0.5 to the bottom

t -= 0.5 # Subtract 0.5 from the top

plt.ylim(b, t) # update the ylim(bottom, top) values

plt.show() 
missing_values_table(train)
#Save loanID

train_ID= train['Loan_ID']

train = train.drop('Loan_ID',axis=1)
#filling n/a with the most occuring for genders

train['Gender'] = train['Gender'].fillna(train['Gender'].mode()[0])
#filling values for married

train['Married'] = train['Married'].fillna(train['Married'].mode()[0])
#filling values for dependents

train['Dependents'] = train['Dependents'].fillna(train['Dependents'].mode()[0])
#filling values for self_employed

train['Self_Employed'] = train['Self_Employed'].fillna(train['Self_Employed'].mode()[0])
avg_loans = train.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)

avg_loans
avg_loans.columns
def values(x):

    return avg_loans.loc[x['Self_Employed'],x['Education']]
train['LoanAmount'] = train['LoanAmount'].fillna(train[train['LoanAmount'].isnull()].apply(values, axis=1))
#filling based on the most frequent

train['Loan_Amount_Term'] = train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0])
from sklearn.preprocessing import LabelBinarizer

encoder = LabelBinarizer()

train['Loan_Status'] = encoder.fit_transform(train['Loan_Status'])
train[train['Credit_History'].isnull()].head()
#filling credit_history where loan status was approved

train['Credit_History'] = np.where(((train['Credit_History'].isnull()) & (train['Loan_Status'] ==1)),

                                   1,train['Credit_History'])



#filling credit_history based on where loan status was declined

train['Credit_History'] = np.where(((train['Credit_History'].isnull()) & (train['Loan_Status'] ==0)),

                                   0,train['Credit_History'])
#Log transfromations

train['LoanAmount'] = np.log1p(train['LoanAmount'])
#Log transforming features

train['ApplicantIncome'] = np.log1p(train['ApplicantIncome'])

train['CoapplicantIncome'] = np.log1p(train['CoapplicantIncome'])
#coapplicant income and applicant income both serves as determinants for loan status

#log transformation



train['total_income'] = train['ApplicantIncome'] + train['CoapplicantIncome']

train['total_income'] = np.log1p(train['total_income'])
#Log transformation 

train['Ratio of LoanAmt :Total_Income'] = train['LoanAmount'] / train['total_income']

train['Ratio of LoanAmt :Total_Income'] = np.log1p(train['Ratio of LoanAmt :Total_Income'])
#checking the categorical variables in the dataset

train.select_dtypes(['object']).columns
#One hot encoding variables

Dummies = pd.get_dummies(train[['Gender','Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']],drop_first=True)



#Dropping the columns which got one hot-encoded

train = train.drop(['Gender','Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'],axis=1)



#Combining the one-hot encoded variables into the actual dataset to make it as one

train = pd.concat([train,Dummies],axis=1)
#Viewing the combined dataset with one-hot enocoded variables

train.head()
#Defining the variables X and y Where; 



#X are the features for training 

X = train.drop('Loan_Status',axis=1)



#y is the target(Loan_Status) to be predicted

y = train['Loan_Status']
#Splitting the train data into train and test purposes.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
#Scaling features. 

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
model_LR = LogisticRegression()

model_LR.fit(X_train,y_train)
pred_LR = model_LR.predict(X_test)

print(classification_report(y_test,pred_LR))

print('\n')

print(confusion_matrix(y_test,pred_LR))
print(accuracy_score(pred_LR,y_test))
model_LR2 = LogisticRegression()

tuned_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] ,

              'penalty':['l1','l2']

                   }



LR = GridSearchCV(estimator=model_LR2,

                  param_grid=tuned_parameters,

                  cv=10,

                 scoring='accuracy',n_jobs=-1)



LR.fit(X_train,y_train)
LR.best_estimator_
pred_LR2 = LR.predict(X_test)

print(classification_report(y_test,pred_LR2))

print('\n')

print(confusion_matrix(y_test,pred_LR2))
print(accuracy_score(pred_LR2,y_test))
import xgboost
params = {

    'learning_rate'   : [0.05,0.3,0.10,0.15,0.20],

    'max_depth'       : [3,4,5,6,8,10],

    'gamma'           : [0.0,0.1,0.2,0.3,0.4],

    'n_estimators'    : range(100,1000,100),

    'colsample_bytree': [0.3,0.4,0.5,0.7]

}





model_xg2 = xgboost.XGBClassifier()

xgb_rand_cv = RandomizedSearchCV(estimator=model_xg2,

                             param_distributions=params,n_iter=5,

                            scoring='accuracy',cv=5,n_jobs=-1)



xgb_rand_cv.fit(X_train,y_train)
pred_xgb = xgb_rand_cv.predict(X_test)

print(classification_report(y_test,pred_xgb))

print('\n')

print(confusion_matrix(y_test,pred_xgb))
print(accuracy_score(pred_xgb,y_test))
from sklearn.svm import SVC

model_svc = SVC()

model_svc.fit(X_train,y_train)
pred_svc = model_svc.predict(X_test)

print(classification_report(y_test,pred_svc))

print('\n')

print(confusion_matrix(y_test,pred_svc))
print(accuracy_score(pred_svc,y_test))
# Applying Grid Search to find the best model and the best parameters

model_svc2 = SVC()

parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},

              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

svm = RandomizedSearchCV(estimator = model_svc2,

                         param_distributions=parameters,

                           scoring = 'accuracy',

                           cv = 5,

                           n_jobs = -1)

svm.fit(X_train, y_train)
svm.best_estimator_
pred_svc2 = svm.predict(X_test)

print(classification_report(y_test,pred_svc2))

print('\n')

print(confusion_matrix(y_test,pred_svc2))
print(accuracy_score(pred_svc2,y_test))
from sklearn.ensemble import RandomForestClassifier

model_RR = RandomForestClassifier()

model_RR.fit(X_train,y_train)
pred_rr = model_RR.predict(X_test)

print(classification_report(y_test,pred_rr))

print('\n')

print(confusion_matrix(y_test,pred_rr))
print(accuracy_score(pred_rr,y_test))
model_RR2 = RandomForestClassifier()



tuned_parameters = {'min_samples_leaf': range(2,100,10), 

                    'n_estimators' : range(100,550,50),

                    'max_features':['auto','sqrt','log2'],

                    'max_depth' : range(0,100,10)

                    }



rr = RandomizedSearchCV(estimator = model_RR2,

                        param_distributions= tuned_parameters,

                           scoring = 'accuracy',

                           cv = 5,

                           n_jobs = -1)



rr.fit(X_train,y_train)
rr.best_params_
pred_rr2 = rr.predict(X_test)

print(classification_report(y_test,pred_rr2))

print('\n')

print(confusion_matrix(y_test,pred_rr2))
print(accuracy_score(y_test,pred_rr2))
X = train.drop('Loan_Status', axis = 1).values

y = train['Loan_Status'].values
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
X_train.shape
y_train.shape
#Importing libraries for neural network

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation,Dropout

from tensorflow.keras.constraints import max_norm
#Creating a neural network

model = Sequential()



# input layer

model.add(Dense(40, activation='relu'))

model.add(Dropout(0.2))



# hidden layer

model.add(Dense(20, activation='relu'))

model.add(Dropout(0.2))



# hidden layer

model.add(Dense(10, activation='relu'))

model.add(Dropout(0.2))



# hidden layer

model.add(Dense(5, activation='relu'))

model.add(Dropout(0.2))



# output layer

model.add(Dense(units=1,activation='sigmoid'))



# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping



early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
model.fit(x=X_train, 

          y=y_train, 

          epochs=600,

          validation_data=(X_test, y_test), verbose=1,

          callbacks=[early_stop]

          )
metrics = pd.DataFrame(model.history.history)

metrics[['loss','val_loss']].plot()
metrics[['accuracy','val_accuracy']].plot()
pred_ANN = model.predict_classes(X_test)
from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,pred_ANN))

print('\n')

print(confusion_matrix(y_test,pred_ANN))
print(accuracy_score(pred_ANN,y_test))
#Training all features and target column on train dataset for final use

X = train.drop('Loan_Status',axis=1)  #------> features

y = train['Loan_Status'] #-------> target(loan_status prediction)
#scaling train features

full_scaler = StandardScaler()

X = full_scaler.fit_transform(X)
#shape of train (features and target)

X.shape,y.shape
LR.fit(X,y)
#Checking first 5 rows of the test data

test.head()
#Checking for null values

test.isnull().sum()
#filling null_values

test['Gender'] = test['Gender'].fillna(test['Gender'].mode()[0])

test['Dependents'] = test['Dependents'].fillna(test['Dependents'].mode()[0])

test['Self_Employed'] = test['Self_Employed'].fillna(test['Self_Employed'].mode()[0])
#fiiling loan amt based on the defined function used on train

test['LoanAmount'] = test['LoanAmount'].fillna(test[test['LoanAmount'].isnull()].apply(values, axis=1))
#filling based on the most frequent

test['Loan_Amount_Term'] = test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0])
test['Credit_History'].value_counts()

#filling based on the most frequent

test['Credit_History'] = test['Credit_History'].fillna(test['Credit_History'].mode()[0])
#Log_transformations

test['ApplicantIncome'] = np.log1p(test['ApplicantIncome'])

test['CoapplicantIncome'] = np.log1p(test['CoapplicantIncome'])

test['LoanAmount'] = np.log1p(test['LoanAmount'])

test['total_income'] = test['ApplicantIncome'] + test['CoapplicantIncome']

test['total_income'] = np.log1p(test['total_income'])

test['Ratio of LoanAmt:Total_income'] = test['LoanAmount']/test['total_income']

test['Ratio of LoanAmt:Total_income'] = np.log1p(test['Ratio of LoanAmt:Total_income'])
test.select_dtypes(['object']).columns
#Save loanID

test_ID= test['Loan_ID']

test = test.drop('Loan_ID',axis=1)
dummies = pd.get_dummies(test[['Gender','Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']],drop_first=True)

test = test.drop(['Gender','Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'],axis=1)

test = pd.concat([test,dummies],axis=1)
#scaling test features

scaled_test = full_scaler.transform(test)
result = LR.predict(scaled_test)
#reassigning target names Y and N

result = np.where(result ==1, 'Y', 'N')

result = pd.Series(result,name='Loan_Status')
test_predictions = pd.concat([test_ID,result],axis=1)

test_predictions['Loan_Status'].value_counts()
test_predictions['Loan_Status'].value_counts()
test_predictions.to_csv('submission_1.csv')