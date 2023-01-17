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
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor



train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
test
train.describe()
train.dtypes
import seaborn as sns



sns.heatmap(train.isnull(), cbar=False)
train['Age'] = train['Age'].fillna(train['Age'].median())





# vamos a revisar si los valores nulos

sns.heatmap(train.isnull(), cbar=False)
test['Age'] = test['Age'].fillna(test['Age'].median())

test['Fare'] = test['Fare'].fillna(test['Fare'].median())



sns.heatmap(test.isnull(), cbar=False)
corr = train.corr()

corr.style.background_gradient(cmap='plasma').set_precision(2)
train = train.drop('Name',axis = 1)

train = train.drop('Sex',axis = 1)

train = train.drop('Ticket',axis = 1)

train = train.drop('Cabin',axis = 1)





test = test.drop('Name',axis = 1)

test = test.drop('Sex',axis = 1)

test = test.drop('Ticket',axis = 1)

test = test.drop('Cabin',axis = 1)
sns.heatmap(train.isnull(), cbar=False)
sns.heatmap(test.isnull(), cbar=False)
for i in [train, test]:

    i['Embarked'] = i['Embarked'].fillna('S')
from sklearn.preprocessing import OneHotEncoder



enc = OneHotEncoder(handle_unknown='ignore')





enc_df = pd.DataFrame(enc.fit_transform(train[['Embarked']]).toarray())

train = train.join(enc_df)



enc_df = pd.DataFrame(enc.fit_transform(test[['Embarked']]).toarray())

test = test.join(enc_df)





train = train.drop('Embarked',axis = 1)

test = test.drop('Embarked',axis = 1)
train.describe()
train.dtypes
X = np.array(train.iloc[:,train.columns != 'Survived'])

y = np.array(train.Survived).reshape(-1,1)
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix





rfc = RandomForestClassifier().fit(X, y.reshape(-1))

y_prima = rfc.predict(X)

print('Cross_val Score RandomForestClassifier = ', cross_val_score(rfc, X, y.reshape(-1), cv=5).mean())
class_names = ["Survived", "Dead"]



print(classification_report(y, y_prima, target_names=class_names))
import matplotlib.pyplot as plt



disp = plot_confusion_matrix(rfc, X, y,

                         display_labels=class_names,

                         cmap=plt.cm.Blues,

                         normalize=None)

plt.show()
test


predictions = rfc.predict(test.iloc[:,test.columns != 'Survived'])



print(test)
predictions = pd.DataFrame(predictions, columns=['Survived'])

test = pd.read_csv('/kaggle/input/titanic/test.csv')

predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)

predictions.to_csv('submission1.csv', sep=",", index = False)



print('end')