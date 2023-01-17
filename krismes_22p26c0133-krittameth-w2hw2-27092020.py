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
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from IPython.display import display
import seaborn as sns
import pandas_profiling
from sklearn.naive_bayes import CategoricalNB,BernoulliNB,ComplementNB,GaussianNB,MultinomialNB
from sklearn.neural_network import MLPClassifier
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")
df_sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
df_train
df_sub.head()
df_test.head()
report = pandas_profiling.ProfileReport(df_train)
display(report)
df1 = df_train.copy()
df2 = df_test.copy()
df1.head()
def extract_feature(df):
#     df.set_index("PassengerId", inplace = True)
    df.Embarked = df.Embarked.fillna("S")
    df['Cherbourg'] =  np.where(df['Embarked'] == 'C',1,0)
    df['Queenstown'] =  np.where(df['Embarked'] == 'Q',1,0)
    df['Southampton'] =  np.where(df['Embarked'] == 'S',1,0)
    df['UpperClass'] =  np.where(df['Pclass'] == 1,1,0)
    df['MiddleClass'] =  np.where(df['Pclass'] == 2,1,0)
    df['LowerClass'] =  np.where(df['Pclass'] == 3,1,0)
    df.Age = df.Age.fillna(df.groupby('Sex')['Age'].transform('mean'))
    sex_map = {'male':0 ,'female':1}
    df["Fare"].fillna(df.Fare.mean(),inplace=True)
    df['Sex'].replace(sex_map,inplace=True)
    df.drop(['Name','Cabin','Ticket','Pclass','Embarked'],inplace=True,axis=1)
    #########Edit 12 AM 6 Oct ##############
#     df.Age= (df.Age-df.Age.min())/(df.Age.max()-df.Age.min())
#     df.Fare= (df.Fare-df.Fare.min())/(df.Fare.max()-df.Fare.min())

extract_feature(df1)
extract_feature(df2)
df1
df1.isna().sum()
df2.isna().sum()
df_sub.isna().sum()
from sklearn.model_selection import cross_val_score,KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB,BernoulliNB,CategoricalNB,ComplementNB,MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,f1_score, precision_score, recall_score, roc_auc_score, accuracy_score, make_scorer
from sklearn.preprocessing import LabelBinarizer
from sklearn import preprocessing
df1.head()
df2.head()
df_sub.head()
x_train = df1.drop(["Survived",'PassengerId'],axis=1)
x_train = preprocessing.StandardScaler().fit(x_train).transform(x_train)
y_train = df1["Survived"]

x_test  = df2.drop(['PassengerId'],axis=1)
x_test = preprocessing.StandardScaler().fit(x_test).transform(x_test)
y_test  = df_sub["Survived"]
x_train[:5]
x_test[:5]
x_train.shape,y_train.shape,x_test.shape,y_test.shape
k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
def Pred(classifier,x_test,y_test):
    
    predictions = classifier.predict(X=x_test)
    print(predictions[:5])
    print(y_test[:5])
    f1 = cross_val_score(classifier,x_train,y_train,cv=k_fold,scoring="f1")
    precision = cross_val_score(classifier,x_train,y_train,cv=k_fold,scoring="precision")
    recall = cross_val_score(classifier,x_train,y_train,cv=k_fold,scoring="recall")
    accuracy = cross_val_score(classifier,x_train,y_train,cv=k_fold,scoring="accuracy")
    

    
    print("F1 score : {}\nPrecision : {}\nRecall : {}\nAccuracy : {}\nAverage F1 : {}".format(f1,precision,recall,accuracy,np.mean(f1)))

tree_de = DecisionTreeClassifier(max_depth = 3)
tree_de.fit(x_train, y_train)
Pred(tree_de,x_test,y_test)

nb = BernoulliNB()
nb.fit(x_train,y_train)
Pred(nb,x_test,y_test)
from sklearn.model_selection import GridSearchCV
def find_best_param(x_train,y_train):
    param_grid = [
            {
                'activation' : ['identity', 'logistic', 'tanh', 'relu'],
                'solver' : ['lbfgs', 'sgd', 'adam'],
                'hidden_layer_sizes': [
                 (1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,), (12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,),(21,)
                 ]

            }
           ]
    clf = GridSearchCV(MLPClassifier(), param_grid, cv=5,
                               scoring='accuracy')
    clf.fit(x_train,y_train)


    print("Best parameters set found on development set:")
    print(clf.best_params_)
    return clf.best_params_
# param = find_best_param(x_train,y_train)
# print(param)
######### From find_best_param ##########################

#{'activation': 'relu', 'hidden_layer_sizes': (5,), 'solver': 'lbfgs'}
mlp = MLPClassifier(hidden_layer_sizes= (20,),solver='adam',max_iter=2000)
mlp.fit(x_train,y_train)
Pred(mlp,x_test,y_test)
