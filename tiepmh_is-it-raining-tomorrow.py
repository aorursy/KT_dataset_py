import pandas as pd

df = pd.read_csv('../input/weatherAUS.csv') #read csv-type data using pandas

df = df.drop( ['RISK_MM'] , axis=1 ) #drop RISK_MM because it is leaked information
df.info() #look at the description of the data

#There are 7 object-type columns and the remaining columns are float64-type
#A rule of thumb is to clean the training data first, then apply the same process to the testing data

df.isnull().sum() * 100 / len(df)  #calculate the percentage of missing data in each column 
''' DECISION:

drop Evaporation, Sunshine, Cloud9am, Cloud3pm out of the data'''

df = df.drop( ['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'], axis=1 )
''' OBSERVATION: 

RainToday and RainTomorrow are object-type and must be transfromed into interger-type '''

df['RainToday'] = df['RainToday'].map( {'No':0, 'Yes':1} ) #replacing Yes/No with 1/0

df['RainTomorrow'] = df['RainTomorrow'].map( {'No':0, 'Yes': 1} ) #replacing Yes/No in the target column with 1/0
''' OBSERVATION:

Date --> extract month from date'''

df['Date'] = pd.to_datetime( df['Date'] ).dt.month

df.rename(columns={'Date':'Month'}, inplace=True) #rename the title of column 'Date' into 'Month'
#see how df looks like

df.head()
''' OBSERVATION:

Location should be transfromed into interger or float64'''

df['Location'].value_counts() #see how many locations we have
#create a list of distinct locations

Location_list = df['Location'].value_counts().index.tolist() 

#create a dictionary which will be used later by map(...) method

Location_mapping = { i:j for i,j in zip( Location_list, range(len(Location_list)) ) } 

#transform distinct locations in the column 'Location' into distinct integers

df['Location'] = df['Location'].map( Location_mapping ) 
#see how df looks like

df.head()
''' OBSERVATION:

WindGustDir should be transfromed into interger or float64'''

WindGustDir_list = df['WindGustDir'].value_counts().index.tolist()

WindGustDir_mapping = { i:j for i,j in zip( WindGustDir_list, range(len(WindGustDir_list)) ) } 

df['WindGustDir'] = df['WindGustDir'].map( WindGustDir_mapping ) 
#see how df looks like

df.head()
''' OBSERVATION:

WindDir9am should be transfromed into interger or float64'''

WindDir9am_list = df['WindDir9am'].value_counts().index.tolist()

WindDir9am_mapping = { i:j for i,j in zip( WindDir9am_list, range(len(WindDir9am_list)) ) } 

df['WindDir9am'] = df['WindDir9am'].map( WindDir9am_mapping ) 
#see how df looks like

df.head()
''' OBSERVATION:

WindDir3pm should be transfromed into interger or float64'''

WindDir3pm_list = df['WindDir3pm'].value_counts().index.tolist()

WindDir3pm_mapping = { i:j for i,j in zip( WindDir3pm_list, range(len(WindDir3pm_list)) ) } 

df['WindDir3pm'] = df['WindDir3pm'].map( WindDir3pm_mapping ) 
#see how df looks like

df.head()
''' DEALING WITH MISSING DATA '''

#Before doing that, we MUST split df into training data and testing data

#This is because we do NOT deal with missing data on the whole dataset (i.e., df), but primarly on the training data

#For testing data, we RE-USE the new values raisen in processing the training data

from sklearn.model_selection import train_test_split #Using train_test_split(...) in the library sklearn

df_train, df_test = train_test_split( df, test_size = 0.25, random_state=42 )

df_train = df_train.copy()

df_test = df_test.copy()
#We want to create a new LIST by combining df_train and df_test 

df_combine = [ df_train, df_test ] #NOTICE: df_combine is a LIST but NOT a dataframe, we use df_combine in the loop for
'''WARNING: 

from now on, we primarily work on the TRAINING data df_train but not df'''

df_train.isnull().sum() * 100 / len(df_train) #calculate the percentage of missing data in each column 
''' OBSERVATION:

Except for Month, Location, RISK_MM and RainTomorrow, all the remaining columns are MISSING data'''

#SEE WHICH COLUMNS ARE MISSING DATA 

cols_NaN_Index = df.columns[ df.isnull().any() ]

#see how it looks like

cols_NaN_Index
for col in cols_NaN_Index:

    mean_val = df_train[col].mean() #NOTE THAT THIS VALUE IS ALSO USED FOR FILLING NAN IN THE TESTING DATA

    df_train[col].fillna( mean_val, inplace=True )

    df_test[col].fillna( mean_val, inplace=True )
df_train.isnull().sum() * 100 / len(df_train)  #calculate the percentage of missing data in each column 
#Now there is not any NaN value

#It is time to look at a heatmap (using the library seaborn) to see the correlation among features

import seaborn as sns

import numpy as np

import matplotlib as plt

fig = sns.heatmap(df_train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #df_train.corr()-->correlation matrix
#We can see that 'MaxTemp' and 'Temp3pm' seem to be highly correlated.

#We want to make sure, so we will look at the numerical values

df_train.corr()
#We can remove either 'MaxTemp' or 'Temp3pm' because the correlation value is 0.969 (close to 1)

#The same goes for the pairs ( 'MinTemp', 'Temp9am' ), ('Pressure9am', 'Pressure3pm')

#DECISION: Remove 'Temp3pm', 'Temp9am' and 'Pressure9am'

df_train = df_train.drop( ['Temp3pm', 'Temp9am', 'Pressure9am'] , axis=1 )

df_test = df_test.drop( ['Temp3pm', 'Temp9am', 'Pressure9am'] , axis=1 )
#see how df_train looks like

df_train.head()
#see the correlation among features in df_train again

fig = sns.heatmap(df_train.corr(),annot=True,cmap='RdYlGn',linewidths=0.2) #df_train.corr()-->correlation matrix
#Everything looks good.  

#Split data into 2 parts, one is training data, and the other is testing data

X_train = df_train.drop('RainTomorrow', axis = 1)

X_test = df_test.drop('RainTomorrow', axis = 1)

y_train = df_train['RainTomorrow']

y_test = df_test['RainTomorrow']

X_train.head()
''' NORMALIZATION '''

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler() 

scaled_values = scaler.fit_transform( X_train ) 

X_train.loc[:,:] = scaled_values
''' NORMALIZATION '''

scaled_values = scaler.transform( X_test ) #DO NOT USE fit METHOD BECAUSE IT'S BEEN MODIFIED ACCORDING TO X_train

X_test.loc[:,:] = scaled_values
''' MACHINE LEARNING '''

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
import time

t0=time.time()

# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

from sklearn.metrics import accuracy_score

acc_log = accuracy_score(y_test,y_pred) *100

print(acc_log)

t_log = time.time()-t0

print(t_log)
t0=time.time()

# Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

y_pred = gaussian.predict(X_test)

acc_gaussian = accuracy_score(y_test,y_pred) *100

print(acc_gaussian)

t_gaussian = time.time()-t0

print(t_gaussian)
t0=time.time()

# Decision Tree

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)

y_pred = decision_tree.predict(X_test)

acc_decision_tree = accuracy_score(y_test,y_pred) *100

print(acc_decision_tree)

t_decision_tree = time.time()-t0

print(t_decision_tree)
t0=time.time()

# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

acc_random_forest = accuracy_score(y_test,y_pred) *100

print(acc_random_forest)

t_random_forest = time.time()-t0

print(t_random_forest)
models = pd.DataFrame(

    {

    'ML Algorithm': ['Naive Bayes', 'Logistic Regression', 'Random Forest', 'Decision Tree'],

    'Score': [acc_gaussian, acc_log, acc_random_forest, acc_decision_tree],

    'Time': [t_gaussian, t_log, t_random_forest, t_decision_tree]

    }

)

models.sort_values(by='Score', ascending=False)