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
data = pd.read_excel('/kaggle/input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx','Data')
data.columns = ["ID","Age","Experience","Income","ZIPCode","Family","CCAvg","Education","Mortgage","PersonalLoan","SecuritiesAccount","CDAccount","Online","CreditCard"]
data.head()
#Missing Or Null data points
data.isnull().sum()
data.isna().sum()
data.shape
data.describe().transpose()
data.nunique()
# there are 52 records with negative experience. Before proceeding any further we need to clean the same
data[data['Experience'] < 0]['Experience'].count()
#clean the negative variable
dfExp = data.loc[data['Experience'] >0]
negExp = data.Experience < 0
column_name = 'Experience'
mylist = data.loc[negExp]['ID'].tolist() # getting the customer ID who has negative experience
# there are 52 records with negative experience
negExp.value_counts()
for id in mylist:
    age = data.loc[np.where(data['ID']==id)]["Age"].tolist()[0]
    education = data.loc[np.where(data['ID']==id)]["Education"].tolist()[0]
    df_filtered = dfExp[(dfExp.Age == age) & (dfExp.Education == education)]
    exp = df_filtered['Experience'].median()
    data.loc[data.loc[np.where(data['ID']==id)].index, 'Experience'] = exp
# checking if there are records with negative experience
data[data['Experience'] < 0]['Experience'].count()
data.describe().transpose()
x=data.iloc[:,[1,3,4,5,6,7,8,10,11,12,13]].values
y=data.iloc[:,9].values
# Divide the data as train and test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=4)
x_train.shape
#modelling SVM
from sklearn import svm
classifier=svm.SVC(kernel='rbf',gamma='auto',C=1)
classifier.fit(x_train,y_train)
y_predict=classifier.predict(x_test)

#Evaluation 
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict))

#Accuracy of our model.
from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_test,y_predict)
print(c)
Accuracy=sum(np.diag(c))/(np.sum(c))
Accuracy
from sklearn import svm
classifier=svm.SVC(kernel='linear',gamma='auto',C=1)
classifier.fit(x_train,y_train)
y_predict=classifier.predict(x_test)

#Evaluation 
from sklearn.metrics import classification_report
print(classification_report(y_test,y_predict))

#Accuracy of our model.
from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_test,y_predict)
print(c)
Accuracy=sum(np.diag(c))/(np.sum(c))
Accuracy