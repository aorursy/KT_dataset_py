#Load the csv file as data frame.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv('../input/creditcardfraud/creditcard.csv')

print('Size of creditcardfraud dataframe is :',df.shape)

#Let us see how our data looks like!

df[0:5]
df['Class'].value_counts()
# Before we start pre-processing, let's see whether there are null values

df.isnull().sum()
#Change categorical numbers with meaningful values

df['Class'].replace({0:'Nonfradulent', 1:'Fradulent'},inplace = True)
 #How many record is fradulent? 

df['Class'].value_counts()
#What percentage record is fradulent?

percentageoffradulent=df['Class'].value_counts(normalize=True)*100

percentageoffradulent
# Get back to old 'Class' values for ml sake

df['Class'].replace({'Nonfradulent':0 , 'Fradulent':1},inplace = True)
#Feature Selection

#Using SelectKBest to get the top features!

from sklearn.feature_selection import SelectKBest, f_classif

X = df.loc[:,df.columns!='Class']

y = df[['Class']]

selector = SelectKBest(f_classif, k=10)

selector.fit(X, y)

X_new = selector.transform(X)

print(X.columns[selector.get_support(indices=True)]) #top 10 columns
#Define X and Y in data

X= df[['V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',

       'Amount']]

y=df[['Class']]
#Spliting data to train and test

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.10)
#Balance data with random oversampling

from imblearn.over_sampling import RandomOverSampler

oversample = RandomOverSampler(sampling_strategy='minority')

X_train, y_train = oversample.fit_resample(X_train, y_train)
#Fit the model

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=100, max_depth=4,random_state=0)

from sklearn.metrics import accuracy_score

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
#Compute accuracy of model

score = accuracy_score(y_test,y_pred)

print('Accuracy :',score)