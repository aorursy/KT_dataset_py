# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



Titan_train = pd.read_csv('../input/train.csv')



Titan_train.head()



Titan_train['Male'] = Titan_train['Sex'].apply(lambda x:1 if x == 'male' else 0)

Titan_train['Age'] = np.where((Titan_train['Age'].isnull().values == True) & (Titan_train['Pclass']==1), 38, Titan_train['Age'] )

Titan_train['Age'] = np.where((Titan_train['Age'].isnull().values == True) & (Titan_train['Pclass']==2), 30, Titan_train['Age'] )

Titan_train['Age'] = np.where((Titan_train['Age'].isnull().values == True) & (Titan_train['Pclass']==3), 25, Titan_train['Age'] )

        



X = Titan_train.iloc[:,[2,5,6,7,9,12]].values

Y = Titan_train.iloc[:,1].values



from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = .20, random_state = 0)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)



from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators = 20,criterion = 'entropy', random_state = 0)



rfc.fit(X_train, Y_train)

Y_pred = rfc.predict(X_test)



from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(Y_test, Y_pred))

print(classification_report(Y_test,Y_pred))





Titan_test = pd.read_csv('../input/test.csv')

Titan_test.head()

Titan_test.isnull().any()

Titan_test.groupby(by='Pclass')['Age'].mean()

Titan_test.groupby(by='Pclass')['Fare'].mean()



Titan_test['Male'] = Titan_test['Sex'].apply(lambda x:1 if x == 'male' else 0)

Titan_test['Age'] = np.where((Titan_test['Age'].isnull().values == True) & (Titan_test['Pclass']==1), 41, Titan_test['Age'] )

Titan_test['Age'] = np.where((Titan_test['Age'].isnull().values == True) & (Titan_test['Pclass']==2), 29, Titan_test['Age'] )

Titan_test['Age'] = np.where((Titan_test['Age'].isnull().values == True) & (Titan_test['Pclass']==3), 24, Titan_test['Age'] )

Titan_test['Fare'] = np.where((Titan_test['Fare'].isnull().values == True) & (Titan_test['Pclass']==1), 94.28, Titan_test['Age'] )

Titan_test['Fare'] = np.where((Titan_test['Fare'].isnull().values == True) & (Titan_test['Pclass']==2), 22.20, Titan_test['Age'] )

Titan_test['Fare'] = np.where((Titan_test['Fare'].isnull().values == True) & (Titan_test['Pclass']==3), 12.45, Titan_test['Age'] )



X_test_pred = Titan_test.iloc[:,[1,4,5,6,8,11]].values



X_test_pred = sc.transform(X_test_pred)



Y_test_pred = rfc.predict(X_test_pred)



# Any results you write to the current directory are saved as output.