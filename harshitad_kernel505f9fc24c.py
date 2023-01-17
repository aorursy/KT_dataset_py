# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/preprocessed_loans.csv')

df.head()
#Removing Unnamed: 0 column from the dataset

df.drop(df.columns[df.columns.str.contains('Unnamed: 0',case = False)],axis = 1, inplace = True)

df.head()


df.select_dtypes('object').apply(pd.Series.nunique, axis = 0)

# removing the following columns as it has thousands of categories 

#and it wont contain enough data within each category to show some significant effect.



df.drop('emp_title',axis = 1, inplace = True)
df.shape
#Cleaning some data as some of the columns meant they should have year but instead had month data in it.

df.rename(columns = {"last_pymnt_month": "last_pymnt_year", 

                                  "last_pymnt_year":"last_pymnt_month", 

                                  "last_credit_pull_month": "last_credit_pull_year",

                                  "last_credit_pull_year": "last_credit_pull_month",

                                  "earliest_cr_line_month": "earliest_cr_line_year",

                                  "earliest_cr_line_year": "earliest_cr_line_month"},inplace=True)
df.head()
null = df.isnull().sum()



# one column 'emp_length' has a lot of null values so it has to be imputed. 

#Rest all the columns have very few (negligible compared to the number of rows) null values so it would be better 

#if we delete those rows.



df['emp_length'].fillna(df['emp_length'].mode()[0], inplace=True)



# nullUpdated=df.isnull().sum()

# nullUpdated



null2=np.where(null.values!=0)



df.dropna(axis=0,inplace=True)

df.isnull().sum()

#Taking data where persons have fully paid or defaulted or have been charged off and saving it in a new dataframe



newdf = df[df['loan_status'].isin(["Fully Paid","Charged Off","Default"])].copy()

newdf.loan_status.value_counts() 

newdf['final_y']=[1 if i in ['Default','Charged Off'] else 0 for i in newdf.loan_status]



newdf.shape
#varY is the target variable , also dropping the loan status and the dummy column created before

varY=newdf['final_y']

newdf.drop('loan_status',axis=1,inplace=True)

newdf.drop('final_y',axis=1,inplace=True)

newdf.shape
from sklearn import preprocessing

count = 0



for col in newdf:

    if newdf[col].dtype == 'object':

        if len(list(newdf[col].unique())) <= 2:     

            le = preprocessing.LabelEncoder()

            newdf[col] = le.fit_transform(newdf[col])

            count += 1

            print (col)

            

print('%d columns were label encoded.' % count)
# one hot encoding for categorical variables

newdf = pd.get_dummies(newdf)

print(newdf.shape)
newdf.head()
#varX is the predictor variable

varX=newdf
# Splitting the data into train and test dataset



from sklearn.metrics import confusion_matrix 

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(varX, varY, test_size=0.2, random_state=500)
#using decision tree

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train)

prediction = dtree.predict(X_test)



#checking performance of the model

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, prediction))

#print("Misclassified samples %d " %(y_test!=prediction) )
# calculating accuracy manually ( DIVIDED MISCLASSIFIED SAMPLES BY TOTAL)and it is 0.9969 

print('Misclassified samples: %d' % (y_test != prediction).sum()) 



from sklearn.ensemble import RandomForestClassifier





forest = RandomForestClassifier()



forest.fit(X_train, y_train)



Y_est_forest = forest.predict(X_test)



print('Misclassified samples: %d' % (y_test != Y_est_forest).sum())











#Checking performance of RF model



from sklearn.metrics import confusion_matrix , classification_report



print(confusion_matrix(y_test,Y_est_forest))



print(classification_report(y_test,Y_est_forest))
# logistic Regression



from sklearn.linear_model import LogisticRegression



log_reg = LogisticRegression()



log_reg.fit(X_train, y_train)



Y_est_log = forest.predict(X_test)





print('Misclassified samples: %d' % (y_test != Y_est_log).sum())

print(confusion_matrix(y_test,Y_est_log))



print(classification_report(y_test,Y_est_log))