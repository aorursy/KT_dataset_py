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
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')

test_data = pd.read_csv('/kaggle/input/titanic/test.csv')
train_data.head()
import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score



fig = plt.figure(figsize=(14,14))

sns.heatmap(train_data.isnull())
train_data.drop(['PassengerId','Name','Ticket','Cabin','Embarked'],axis=1,inplace=True)

train_data
avg_age = train_data['Age'].mean()

avg_age
train_data['Age'].fillna(avg_age,inplace=True)
sex_train = pd.get_dummies(train_data['Sex'],drop_first=True)

sex_train.head()
train_data = pd.concat([train_data,sex_train],axis=1)

train_data.drop('Sex',axis=1,inplace=True)

train_data.info()
x = train_data.drop('Survived',axis=1)

y = train_data['Survived']
print(x.shape)

print(y.shape)
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

x_scale = scaler.fit_transform(x)

print(x_scale)
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2000)



classifiers = [

    DecisionTreeClassifier(),

    GaussianNB(),

    MLPClassifier(hidden_layer_sizes=(10, 10, 10),max_iter=1000)

]



models = []

for cls in classifiers:

    name = cls.__class__.__name__

    mod = cls.fit(X_train, y_train)

    predicted = mod.predict(X_test)

    

    print(name)

    print('------------------------------------')

    print('Accuracy  : '+ format(accuracy_score(y_test,predicted)))

    print('Recall    : '+ format(recall_score(y_test,predicted)))

    print('Precision : '+ format(precision_score(y_test,predicted)))

    print('F1 Score  : '+ format(f1_score(y_test,predicted)))

    print("")

    models.append(mod)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate

from sklearn.model_selection import cross_val_score



class ClassificationModel:

   

    def __init__(self,name, accuracy, recall,precision,f1):

        self.name = name

        self.accuracy = accuracy

        self.recall = recall

        self.precision = precision

        self.f1 = f1

    

   

    def display(self):

        print(self.name)

        print('------------------------------------')

        print('Accuracy     : '+ format(self.accuracy))

        print('Recall       : '+ format(self.recall))

        print('Precision    : '+ format(self.precision))

        print('F1 Score     : '+ format(self.f1))

        print('Avg F1 Score : '+ format(np.array(self.f1).mean()))

        print('')
print('5-fold cross validation\n')



model_object = []

for cls in classifiers:

    name = cls.__class__.__name__

    accuracy = []

    recall = []

    precision = []

    f1 = []

    cv = KFold(n_splits=5, random_state=42, shuffle=True)

    for train_index, test_index in cv.split(x_scale):

    

        X_train, X_test, y_train, y_test = x_scale[train_index], x_scale[test_index], y[train_index], y[test_index]

        cls.fit(X_train, y_train)

        predicted = cls.predict(X_test)



        accuracy.append(accuracy_score(y_test, predicted))

        recall.append(recall_score(y_test, predicted))

        precision.append(precision_score(y_test, predicted))

        f1.append(f1_score(y_test, predicted))

        

    model_object.append(ClassificationModel(name,accuracy,recall,precision,f1))



for obj in model_object:

    obj.display()