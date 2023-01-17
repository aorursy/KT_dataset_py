# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_validate

from sklearn import metrics

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('/kaggle/input/loanprediction/train_ctrUa4K.csv')

df.head()
df.describe()
df['Property_Area'].value_counts()
df['ApplicantIncome'].hist(bins=50)
df.boxplot(column='ApplicantIncome')
df.boxplot(column='ApplicantIncome', by = 'Education')
df['LoanAmount'].hist(bins=50)
df.boxplot(column='LoanAmount')
temp1 = df['Credit_History'].value_counts(ascending=True)

temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

print ('Frequency Table for Credit History:') 

print (temp1)



print ('\nProbility of getting loan for each Credit History class:')

print (temp2)
import matplotlib.pyplot as plt

plt.xlabel('Credit_History')

plt.ylabel('Count of Applicants')

plt.title("Applicants by Credit_History")

temp1.plot(kind='bar')
temp2.plot(kind = 'bar')

plt.xlabel('Credit_History')

plt.ylabel('Probability of getting loan')

plt.title("Probability of getting loan by credit history")
temp1 = df['Married'].value_counts(ascending=True)

temp2 = df.pivot_table(values='Loan_Status',index=['Married'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

print ('Frequency Table for Credit History:') 

print (temp1)



print ('\nProbility of getting loan on the basis of martial status:')

print (temp2)
temp1 = df['Self_Employed'].value_counts(ascending=True)

temp2 = df.pivot_table(values='Loan_Status',index=['Self_Employed'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

print ('Frequency Table for Credit History:') 

print (temp1)



print ('\nProbility of getting loan on the basis of employment:')

print (temp2)
temp1 = df['Property_Area'].value_counts(ascending=True)

temp2 = df.pivot_table(values='Loan_Status',index=['Property_Area'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

print ('Frequency Table for Credit History:') 

print (temp1)



print ('\nProbility of getting loan on the basis of employment:')

print (temp2)
temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])

temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
temp3 = pd.crosstab(df['Married'], df['Loan_Status'])

temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
temp3 = pd.crosstab(df['Self_Employed'], df['Loan_Status'])

temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
temp3 = pd.crosstab(df['Property_Area'], df['Loan_Status'])

temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
df.apply(lambda x : sum(x.isnull()), axis=0)
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
df['Self_Employed'].value_counts()
df['Self_Employed'].fillna('No', inplace=True)
df['LoanAmount_log'] = np.log(df['LoanAmount'])

df['LoanAmount_log'].hist(bins=20)
df['TotalIncome']=df['ApplicantIncome'] + df['CoapplicantIncome']

df['TotalIncome_log'] = np.log(df['TotalIncome'])

df['TotalIncome_log'].hist(bins=20)
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)

df['Married'].fillna(df['Married'].mode()[0], inplace=True)

df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)

df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)

df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']

le = LabelEncoder()

for i in var_mod:

    df[i] = le.fit_transform(df[i])

df.dtypes
#Generic function for making a classification model and accessing performance:

def classification_model(model, data, predictors, outcome):

  #Fit the model:

  model.fit(data[predictors],data[outcome])

  

  #Make predictions on training set:

  predictions = model.predict(data[predictors])

  

  #Print accuracy

  accuracy = metrics.accuracy_score(predictions,data[outcome])

  print ("Accuracy : %s" % "{0:.3%}".format(accuracy))



  #Perform k-fold cross-validation with 5 folds

#   kf = KFold(data.shape[0], n_folds=5)

  kf = KFold( n_splits = 5)

  kf.get_n_splits(data.values)



  error = []

  for train, test in kf.split(data.values):

    # Filter training data

    train_predictors = (data[predictors].iloc[train,:])

    

    # The target we're using to train the algorithm.

    train_target = data[outcome].iloc[train]

    

    # Training the algorithm using the predictors and target.

    model.fit(train_predictors, train_target)

    

    #Record error from each cross-validation run

    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))

 

  print ("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))



  #Fit the model again so that it can be refered outside the function:

  model.fit(data[predictors],data[outcome]) 
outcome_var = ['Loan_Status']

model = LogisticRegression()

predictor_var = ['Credit_History']

# print( type(df.values), df)

classification_model(model, df,predictor_var,outcome_var)
#We can try different combination of variables:

predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']

classification_model(model, df,predictor_var,outcome_var)
model = DecisionTreeClassifier()

predictor_var = ['Credit_History','Gender','Married','Education']

classification_model(model, df,predictor_var,outcome_var)
#We can try different combination of variables:

predictor_var = ['Credit_History','Loan_Amount_Term','LoanAmount_log']

classification_model(model, df,predictor_var,outcome_var)
model = RandomForestClassifier(n_estimators=100)

predictor_var = ['Gender', 'Married', 'Dependents', 'Education',

       'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',

        'LoanAmount_log','TotalIncome_log']

classification_model(model, df,predictor_var,outcome_var)
#Create a series with feature importances:

featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)

print (featimp)
model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)

predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History','Dependents','Property_Area']

classification_model(model, df,predictor_var,outcome_var)