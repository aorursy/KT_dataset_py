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
import pandas as pd
import numpy as np
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

df = pd.read_csv('/kaggle/input/titanic/train.csv')
df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df.head()
gendersub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
gendersub 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
df.isnull().sum() 
df.describe()
df_test.describe()
data = df[['Pclass', 'Sex', 'SibSp', 'Age', 'Fare','Survived']]
data.head(10)
data['Age']=data['Age'].fillna(data['Age'].mean())
df.isnull().sum()
sns.countplot(x='Survived',data=df)
sns.countplot(x='Survived',hue='Sex',data=df)
sns.countplot(x='Survived',hue='Pclass',data=df)
sns.catplot(x="Embarked", y="Survived", kind="bar", data=df)
pd.crosstab(data.Pclass,df.Survived,margins=True).style.background_gradient()
df['Age'] = pd.cut(df['Age'],6,labels = [1,2,3,4,5,6])

df
data = df[['Sex','Fare','SibSp','Parch','Pclass','Age','Embarked']]
data
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df
#DATA Preparetion
train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.info()
age_mean = train['Age'].mean()
train['Age'].fillna(age_mean, inplace = True)
train['Embarked'].fillna('S', inplace = True)
train.info()
train['Age'] = pd.cut(train['Age'],6,labels = [1,2,3,4,5,6])

train
df = train[['Sex','Fare','SibSp','Parch','Pclass','Age','Embarked']]
df.head()
df = train[['Pclass', 'Sex', 'SibSp', 'Age', 'Fare','Survived']]
df.head(10)
def clean_data(df):
    data['AgeRange'] = df['Age'].apply(lambda x: int(x/20) if not math.isnan(x) else x)
    data['Male'] = df['Sex'].apply(lambda x: 1 if x == 'male' else 0)
    data['Female'] = df['Sex'].apply(lambda x: 0 if x == 'male' else 1)
    data['Survived'] = df['Survived'].apply(lambda x: 1 if x else 0)
    return df
df = clean_data(df)
df.head(11)
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import precision_score, recall_score

def evaluation(y_true, y_pred, pos_label=1):
    l = len(y_pred)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range (l):
        if y_pred[i] == pos_label: #tp,fp
            if y_pred[i] == y_true[i]:
                tp += 1
            else:
                fp +=1
        else:
            if y_pred[i] == y_true[i]:
                tn += 1
            else:
                fn +=1       
    p = tp / (tp+fp)
    r = tp/(tp+fn)
    f1 = 2*p*r/(p+r)
    return {"precision": p, "recall": r, "f1": f1}


X = data[['Pclass', 'SibSp', 'Fare', 'Male', 'Female']]
Y = data[['Survived']]
X = np.array(X)
Y = np.array(Y)
print(X.shape, Y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1) 
y_test = np.array(y_test)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
result = model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(evaluation(y_test, y_pred,1))
print(evaluation(y_test, y_pred,0))
from sklearn.naive_bayes import GaussianNB
bmodel = GaussianNB()
result = bmodel.fit(X_train,y_train)
y_pred = bmodel.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(evaluation(y_test, y_pred,1))
print(evaluation(y_test, y_pred,0))
from sklearn.neural_network import MLPClassifier
mlp_model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
result = mlp_model.fit(X_train,y_train)
y_pred = mlp_model.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print(evaluation(y_test, y_pred,1))
print(evaluation(y_test, y_pred,0))
def evaluate_model(model, X_train, Y_train, X_test, Y_test):
    result = model.fit(X_train,Y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
    print(evaluation(Y_test, y_pred,1))
    print(evaluation(Y_test, y_pred,0))
    
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
count = 1
for train_index, test_index in kf.split(X):
    print("Fold: ", count )
    count += 1
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    Y_train = Y_train.reshape(len(Y_train),)
    dmodel = DecisionTreeClassifier()
    bmodel = GaussianNB()
    mlp_model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
    print("Decision Tree: ")
    evaluate_model(dmodel, X_train, Y_train, X_test, Y_test)
    print("Naive Bays")
    evaluate_model(bmodel, X_train, Y_train, X_test, Y_test)
    print("Neuron Network")
    evaluate_model(mlp_model, X_train, Y_train, X_test, Y_test)
import tensorflow as tf
d_in = (X_train.shape[1],)
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(20, input_shape=d_in, 
                                activation=tf.keras.layers.PReLU()
))
model.add(tf.keras.layers.Dense(10,
                                activation=tf.keras.layers.PReLU()))
model.add(tf.keras.layers.Dense(5,
                                activation=tf.keras.layers.PReLU()))
model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
model.compile(loss=tf.keras.losses.MeanSquaredError(),
                        optimizer=tf.keras.optimizers.SGD(learning_rate=0.09))
model.summary()