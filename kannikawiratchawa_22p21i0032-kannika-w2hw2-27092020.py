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
!pip install Boruta
!pip install dabl

import dabl
import pandas as pd
from boruta import BorutaPy

from sklearn.ensemble import RandomForestClassifier
import sklearn

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn import model_selection, preprocessing, metrics

from sklearn.naive_bayes import GaussianNB
import keras

from keras.models import Sequential

from keras.layers import Dense
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_fscore_support

from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer
train_titanic = pd.read_csv('../input/titanic/train.csv')
test_titanic = pd.read_csv('../input/titanic/train.csv')
train_titanic.shape
train_titanic.head()
train_titanic.isnull().any()
dabl.detect_types(train_titanic)
figure = plt.figure(figsize = [20,8])



sns.distplot(train_titanic['Age'])
train_titanic['Survived'].value_counts()
dabl.plot(train_titanic, target_col="Survived")
train_titanic.groupby('Sex')[['Survived']].mean()
sns.barplot(x='Pclass', y='Survived', data=train_titanic)
train_titanic.pivot_table('Survived', index='Sex', columns='Pclass')
train_titanic.pivot_table('Survived', index='Sex', columns='Pclass').plot()
sns.barplot(x='Sex', y='Survived', data=train_titanic)
sns.barplot(x='Embarked', y='Survived', data=train_titanic)
train_titanic.corr().style.background_gradient(cmap='Reds')
train_titanic = train_titanic.drop(['PassengerId','Name', 'Cabin', 'Ticket'], axis=1)

train_titanic = train_titanic.dropna()

train_titanic.head()
from sklearn import preprocessing

tr = preprocessing.LabelEncoder()

train_titanic.Sex = tr.fit_transform(train_titanic.Sex)

train_titanic.Embarked = tr.fit_transform(train_titanic.Embarked)

train_titanic.head()
X = train_titanic.drop('Survived', axis = 1)

Y = train_titanic['Survived']

X.head()
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=42)
decision_tree = DecisionTreeClassifier() 

decision_tree.fit(X_train, Y_train) 

Y_pred = decision_tree.predict(X_test)  

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree
gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)  

Y_pred_ba = gaussian.predict(X_test)  

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
acc_gaussian
mlp = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001, solver='sgd', verbose=10, random_state=21, tol=0.000000001)
mlp.fit(X_train, Y_train)

y_pred_mlp = mlp.predict(X_test)
from sklearn.metrics import precision_recall_fscore_support as score



predicted = Y_pred

y_test = Y_test



precision, recall, fscore, support = score(y_test, predicted)



print('precision: {}'.format(precision))

print('recall: {}'.format(recall))

print('fscore: {}'.format(fscore))

print('support: {}'.format(support))
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]

def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]

def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]

def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]



scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),

            'fp': make_scorer(fp), 'fn': make_scorer(fn)}
def evalua_metric(model_cv):

    

    precision1 = []

    recall1 = []

    f11 = []

    

    precision2 = []

    recall2 = []

    f12 = []

    

    avg_all = []

    

    for i in range(len(model_cv['fit_time'])):

        

          # class = 0

        tp_1 = model_cv['test_tn'][i]

        fn_1 = model_cv['test_fp'][i]

        fp_1 = model_cv['test_fn'][i]

        tn_1 = model_cv['test_tp'][i]

        precision_1 = tp_1/(tp_1 + fp_1)

        recall_1 = tp_1/(tp_1 + fn_1)

        f1_1 = 2*(precision_1 * recall_1)/(precision_1 + recall_1)



        precision1.append(np.round(precision_1, 4))

        recall1.append(np.round(recall_1, 4))

        f11.append(np.round(f1_1, 4))

        



        # Class = 1

        tn_2 = model_cv['test_tn'][i]

        fp_2 = model_cv['test_fp'][i]

        fn_2 = model_cv['test_fn'][i]

        tp_2 = model_cv['test_tp'][i]

        precision_2 = tp_2/(tp_2 + fp_2)

        recall_2 = tp_2/(tp_2 + fn_2)

        f1_2 = 2*(precision_2 * recall_2)/(precision_2 + recall_2)

        

        precision2.append(np.round(precision_2, 4))

        recall2.append(np.round(recall_2, 4))

        f12.append(np.round(f1_2, 4))

        

        #Average F-Measure

        avg_f2 = (f1_2 + f1_1)/2

        

        avg_all.append(np.round(avg_f2, 4))

    

    

    evaluation_df = pd.DataFrame({'Precision Class 0': precision1, 'Recall Class 0': recall1, 'F1-Score Class 0': f11, \

                        'Precision Class 1': precision2, 'Recall Class 1': recall2, 'F1-Score Class 1': f12, \

                        'Average F1-Score of All Dataset ': avg_all })

    

    return  evaluation_df
model_cv_tree = cross_validate(decision_tree, X, Y, cv=5, scoring=scoring)
evalua_metric(model_cv_tree)
model_cv_bay = cross_validate(gaussian, X, Y, cv=5, scoring=scoring)
evalua_metric(model_cv_bay)
model_cv_ann = cross_validate(mlp, X, Y, cv=5, scoring=scoring)
evalua_metric(model_cv_ann)