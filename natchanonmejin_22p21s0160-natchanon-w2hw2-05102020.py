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
import pandas as pd #ดึงข้อมูลจากไฟล์ train และ test
titanic_train= pd.read_csv('../input/titanic/train.csv')
titanic_test= pd.read_csv('../input/titanic/test.csv')
titanic_train = titanic_train.drop(['PassengerId','Name', 'Cabin', 'Ticket'], axis=1)
titanic_test = titanic_test.drop(['PassengerId','Name', 'Cabin', 'Ticket'], axis=1)
#ตัดคอลัมน์ในส่วนที่คาดว่าไม่มีความเกี่ยวข้องในการทำนาย
titanic_train= titanic_train.dropna() #ตัดคอลัมน์ที่ข้อมูลสูญหาย
titanic_test= titanic_train.dropna()
from sklearn import preprocessing
def preprocess_titanic_train(df):
    processed_train = df.copy()
    le = preprocessing.LabelEncoder()
    processed_train.Sex = le.fit_transform(processed_train.Sex)
    processed_train.Embarked = le.fit_transform(processed_train.Embarked)
    return processed_train
processed_train= preprocess_titanic_train(titanic_train)#จัดเตรียมข้อมูลเพื่อเตรียมทำ dataset
X = processed_train.drop(['Survived'], axis=1).values #จัดเตรียมข้อมูลให้เป็น dataset
y = processed_train['Survived'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2) 
#แบ่งข้อมูลที่จัดเตรียมมาแล้วมาแยก train 80% test 20%
from sklearn.tree import DecisionTreeClassifier #Decision Tree model
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
from sklearn.naive_bayes import GaussianNB #Naive Bayes model
naive = GaussianNB()
naive.fit(X_train, y_train)
from sklearn.neural_network import MLPClassifier #Neural network model
neural = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
neural.fit(X_train,y_train)
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve

def evaluate_model(model, X_train, y_train, X_test, y_test):
    result = model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return result, y_pred
kf= KFold(n_splits=5)

count =0
for train_index, test_index in kf.split(X):

    count +=1
    X_train, X_test= X[train_index], X[test_index]
    y_train, y_test= y[train_index], y[test_index]
    
    Treemodel= DecisionTreeClassifier()
    Naivmodel= GaussianNB()
    Neumodel= MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=1000)
    print('Decision Tree Fold{0:0.0f}_________________________________________________'.format(count))
    evaluate_model(Treemodel, X_train, y_train, X_test, y_test)
    print('Naive Bayes Fold{0:0.0f}_________________________________________________'.format(count))
    evaluate_model(Naivmodel, X_train, y_train, X_test, y_test)
    print('Neural Network Fold{0:0.0f}_________________________________________________'.format(count))
    evaluate_model(Neumodel, X_train, y_train, X_test, y_test)
    
    disp1= plot_precision_recall_curve(decision_tree, X_test, y_test)
    disp1.ax_.set_title('2-class Precision-Recall curve Fold-{0:0.0f}'.format(count))
    
    disp2= plot_precision_recall_curve(naive, X_test, y_test)
    disp2.ax_.set_title('2-class Precision-Recall curve Fold-{0:0.0f}'.format(count))
    
    disp3= plot_precision_recall_curve(Neumodel, X_test, y_test)
    disp3.ax_.set_title('2-class Precision-Recall curve Fold-{0:0.0f}'.format(count))
