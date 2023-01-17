# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Configure necessary imports

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd



ins = pd.read_csv('/kaggle/input/ds-nigeria-2019-challenge-insurance-prediction/train_data.csv')
#Inspecting the data

ins.head()
# Encoding Categorical data with discrete numerical values

to_num = {'V':1,'N':0,'O':0,'U':1,'R':0}

ins.iloc[:,[4,5,6,7]]  =  ins.iloc[:,[4,5,6,7]].applymap(to_num.get)

ins.head()
ins.NumberOfWindows.value_counts()
#Conducting String replacement to convert the NumberOfWindows column to int64

ins.NumberOfWindows = ins.NumberOfWindows.str.replace(".","0")

ins.NumberOfWindows = ins.NumberOfWindows.str.replace(">=10","10")

ins.NumberOfWindows = ins.NumberOfWindows.astype('int64')

ins.NumberOfWindows.value_counts()
ins.info()
#Imputing missing values, 0 for binary categorical columns, column mode for discrete columns

ins.Garden = ins.Garden.fillna(0)

ins['Building Dimension'] = ins['Building Dimension'].fillna(0)

ins.Date_of_Occupancy = ins.Date_of_Occupancy.fillna(1960)

ins.Date_of_Occupancy = ins.Date_of_Occupancy.astype('int64')

ins.info()
plt.hist(ins['Date_of_Occupancy'],bins=25)

plt.xlabel('Year of Occupancy')

plt.ylabel('Count')
sns.catplot('YearOfObservation','Date_of_Occupancy',data = ins)

import numpy as np

t = ins[['YearOfObservation','Residential',

       'Building_Painted', 'Building_Fenced']].groupby('YearOfObservation').agg([np.sum])



t
t.plot()
#Splitting Training Data into further train and test sets

X = ins.iloc[:,1:12]

y = ins.Claim



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = .2,random_state = 2)
#Model training and evaluation

lr = LogisticRegression(C = 1,max_iter = 3000,class_weight = 'balanced')

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
roc_auc_score(y_test,y_pred)


coefs = dict(zip(X_train,lr.coef_.reshape(-1,1)))

coefs = pd.DataFrame(coefs)

coefs = coefs.T

coefs.columns = ['Odds']

coefs.Odds = np.exp(coefs.Odds)

coefs
inst = pd.read_csv("/kaggle/input/ds-nigeria-2019-challenge-insurance-prediction/test_data.csv")

inst.head()
#Cleaning the Data as was done in the test set

inst.iloc[:,[4,5,6,7]]  =  inst.iloc[:,[4,5,6,7]].applymap(to_num.get)

inst.NumberOfWindows = inst.NumberOfWindows.str.replace(".","0")

inst.NumberOfWindows = inst.NumberOfWindows.str.replace(">=10","10")

inst.NumberOfWindows = inst.NumberOfWindows.astype('int64')



inst.Garden = inst.Garden.fillna(0)

inst['Building Dimension'] = inst['Building Dimension'].fillna(0)

inst.Date_of_Occupancy = inst.Date_of_Occupancy.fillna(1960)



#Assigning new explanatory variables to fit the model

X_test1 = inst.iloc[:,1:12]



#Predicting the requisite values

y_pred1 = lr.predict(X_test1)



#Creating the result dataframe

result = pd.DataFrame(zip(inst['Customer Id'],y_pred1))

result.columns = ['Client','Claim']

result = result.set_index('Client')
result