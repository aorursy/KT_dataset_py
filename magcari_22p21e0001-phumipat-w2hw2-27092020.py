import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math
test = pd.read_csv("/kaggle/input/titanic/test.csv")

train = pd.read_csv("/kaggle/input/titanic/train.csv")

train
train = train[['Pclass','Sex','SibSp','Parch','Survived']]

test = test[['Pclass','Sex','SibSp','Parch']]

train
train['Sex'] = train['Sex'].apply(lambda x: 1 if x == 'male' else 0)

test['Sex'] = test['Sex'].apply(lambda x: 1 if x == 'male' else 0)



train
from sklearn.model_selection import train_test_split



x = train[['Pclass','Sex','SibSp','Parch']]

x = np.array(x)

y = train[['Survived']]

y = np.array(y)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 42)
from sklearn import metrics

from sklearn.metrics import precision_score,recall_score



def evaluate(y_ans,y_pred,label = 1):

    tp = 0

    tn = 0

    fp = 0

    fn = 0

    for i in range(len(y_pred)):

        if y_pred[i] == label:

            if y_pred[i] == y_ans[i]:

                tp+=1

            else:

                fp+=1

        else:

            if y_pred[i] == y_ans[i]:

                tn+=1

            else:

                fn+=1

    

    prec = tp/(tp+fp)

    rec = tp/(tp+fn)

    f1 = 2*tp/(2*tp + fp + fn)

    return {"Precision": prec, "Recall": rec, "F1_Score": f1}
from sklearn.tree import DecisionTreeClassifier



dt_model = DecisionTreeClassifier()

dt_model.fit(x_train,y_train)

dt_y_pred = dt_model.predict(x_test)



print('Accuracy: ',metrics.accuracy_score(y_test,dt_y_pred))

print(evaluate(y_test,dt_y_pred,1))
from sklearn.naive_bayes import GaussianNB



nb_model = GaussianNB()

nb_model.fit(x_train,y_train)

nb_y_pred = nb_model.predict(x_test)



print('Accuracy: ',metrics.accuracy_score(y_test,nb_y_pred))

print(evaluate(y_test,nb_y_pred,1))
from sklearn.neural_network import MLPClassifier



mlp_model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)

mlp_model.fit(x_train,y_train)

mlp_y_pred = mlp_model.predict(x_test)



print('Accuracy: ',metrics.accuracy_score(y_test,mlp_y_pred))

print(evaluate(y_test,mlp_y_pred,1))
def train_evaluate(model,x_train,y_train,x_test,y_test):

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    

    print('Accuracy: ',metrics.accuracy_score(y_test,y_pred))

    print(evaluate(y_test,y_pred,1))

    print('')
from sklearn.model_selection import KFold

folds = KFold(n_splits = 5)

num = 1

for train_idx,test_idx in folds.split(x):

    print("Fold ",num)

    num+=1

    

    x_train,x_test = x[train_idx],x[test_idx]

    y_train,y_test = y[train_idx],y[test_idx]

    y_train = y_train.reshape(len(y_train))

    

    dt_model = DecisionTreeClassifier()

    nb_model = GaussianNB()

    mlp_model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)

    

    print("Decision Tree")

    train_evaluate(dt_model,x_train,y_train,x_test,y_test)

    

    print("Naive Bayes")

    train_evaluate(nb_model,x_train,y_train,x_test,y_test)

    

    print("Neuron Network")

    train_evaluate(mlp_model,x_train,y_train,x_test,y_test)

    print('\n')