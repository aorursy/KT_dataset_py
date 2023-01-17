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

import pylab as pl

import numpy as np

import scipy.optimize as opt

from sklearn import preprocessing

%matplotlib inline 

import matplotlib.pyplot as plt

import seaborn as sns

import itertools

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter

import pandas as pd

import numpy as np

import matplotlib.ticker as ticker

from sklearn import preprocessing

%matplotlib inline
path ="../input/titanic/test.csv"

path1="../input/titanic/train.csv"
df=pd.read_csv(path1)

df.head()
df1=pd.read_csv(path)

df1.head()
#creating a copy of datasets

df_copy=df.copy()

df1_copy=df1.copy()
df_copy['train']  = 1      #Survived column is not mentioned in the test set so creating a label so that we can sort them out while splitting

df1_copy['train']  = 0

df_con = pd.concat([df_copy, df1_copy], axis=0,sort=False)
corr_matrix =df_con.corr()     #correlation

corr_matrix["Survived"].sort_values(ascending=False)
object_col=df_con.select_dtypes(include=["object"])     #separating the object and numerical columns

num_col=df_con.select_dtypes(exclude=["object"])
object1=object_col.copy()

num=num_col.copy()
print(' column \t object_col  ')   #percentage of missing values in categorical features

print('='*30)

for col in object_col.columns:

    percentage_col = (object_col[col].isnull().sum()/len(object_col))*100

    

    print('{}\t{} % '.format(col.ljust(15,' '), round(percentage_col, 3)))


freq=object_col["Embarked"].mode()  #filling embarked column with most freq value

freq



object1["Embarked"] = object1["Embarked"].fillna("S")
object1.drop(["Cabin","Ticket","Name"],axis=1,inplace=True)
object1
print(' column \t object_col  ')   #cross checking percentage of missing values

print('='*30)

for col in object1.columns:

    percentage_col = (object1[col].isnull().sum()/len(object1))*100

    

    print('{}\t{} % '.format(col.ljust(15,' '), round(percentage_col, 3)))
bin_map={"S":1,"C":2,"Q":3}   #mapping 1 to N values

object1["Embarked"]=object1.Embarked.map(bin_map)
map1={"male":1,"female":2}

object1["Sex"]=object1.Sex.map(map1)
#object_columns = object1.select_dtypes(include=['object'])



#objectdf = pd.get_dummies(object1, columns=object_columns.columns)
object1.Embarked.value_counts()
object1.head()
print(' column \t num_col  ')

print('='*40)

for col in num_col.columns:

    percentage_num = (num_col[col].isnull().sum()/len(num_col))*100

    

    print('{}\t{} % '.format(col.ljust(15,' '), round(percentage_num, 3)))
age=num_col.Age.mode()

print(age)

fare=num_col.Fare.mode()

print(fare)
num["Age"]=num["Age"].fillna(value=24.0)

                        

num["Fare"]=num["Fare"].fillna(value=8.05)
num
num= num.fillna(0)
num.head()
X= num.Pclass.value_counts()

print(X)

sns.barplot(x=X,y="Survived",data=num)
print(' column \t num_col  ')

print('='*40)

for col in num.columns:

    percentage_num = (num[col].isnull().sum()/len(num))*100

    

    print('{}\t{} % '.format(col.ljust(15,' '), round(percentage_num, 3)))
num.head()
final = pd.concat([object1, num], axis=1,sort=False)   #concating the object and numerical features.

final.head()
final = final.drop(['PassengerId',],axis=1)    #separating train and test sets



df_train = final[final['train'] == 1]

df_train = df_train.drop(['train',],axis=1)





df_test = final[final['train'] == 0]

df_test = df_test.drop(['Survived'],axis=1)

df_test = df_test.drop(['train',],axis=1)
target= df_train['Survived']

df_train = df_train.drop(['Survived'],axis=1)
col=df_train.columns

col
X = df_train[col] .values  #.astype(float)

X[0:5]

y = target.values

y[0:5]
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

X[0:5]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size=0.2, random_state=4)

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)
from sklearn.linear_model import SGDClassifier
sgd_clf =SGDClassifier(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X ,y, test_size=0.2, random_state=4)
sgd_clf.fit(X_train,y_train)
sgd_clf.predict(X_test)
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf,X_train,y_train,cv=4,scoring="accuracy")
yha_prob = sgd_clf.predict(X_test)

yha_prob
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix

import itertools

def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

print(confusion_matrix(y_test, yha_prob, labels=[1,0]))
# Compute confusion matrix

cnf_matrix = confusion_matrix(y_test, yha_prob, labels=[1,0])

np.set_printoptions(precision=2)





# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
predict= sgd_clf.predict(df_test)

predict
result = pd.DataFrame({"PassengerId": df1["PassengerId"],

        

        "Survived": predict

    })

result.to_csv('result.csv', index=False)
result.head()