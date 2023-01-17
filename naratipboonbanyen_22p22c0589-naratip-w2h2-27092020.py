# 22p22c0589_Naratip_W2H2_27092020
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
df = pd.read_csv('../input/titanic/train.csv')
df
df.info()
# Clean data
df = df[['Survived', 'Pclass', 'Sex','Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
df['Age'] = df['Age'].fillna(df['Age'].mode()[0])
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df.info()
#One hot encoding
# df = pd.get_dummies(df, columns=['Sex', 'Embarked'],drop_first = False)
df = pd.get_dummies(df, columns=['Sex', 'Embarked'],drop_first = True)
df
# split data 2 column for eazy to nor
df_norm = df[['Pclass','Age', 'SibSp', 'Parch', 'Fare']]
df_bi = df[['Survived','Sex_male','Embarked_Q','Embarked_S']]
# Standardization in some coulumn
from sklearn import preprocessing
cols = ['Pclass','Age', 'SibSp', 'Parch', 'Fare']
pt = preprocessing.PowerTransformer(method='yeo-johnson', standardize=True) # support only positive value
mat = pt.fit_transform(df_norm[cols])

X = pd.DataFrame(mat, columns=cols)

# comnine data that do Standardization with data that do one hot encode
X = pd.concat([X, df_bi], axis=1)
X
data = X[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_male','Embarked_Q', 'Embarked_S']]
target = X[['Survived']]

print(data.shape, target.shape)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.3, random_state=1) 
Y_test = np.array(Y_test)

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.metrics import precision_score, recall_score

def evaluation(y_true, y_pred):
    l = len(y_pred)
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range (l):
        if y_pred[i] == y_true[i] and y_pred[i] == 1 : #model 1 real 1
            tp += 1
        elif y_pred[i] != y_true[i] and y_pred[i] == 1 : #model 1 real 0
            fp +=1

        elif y_pred[i] == y_true[i] and y_pred[i] == 0: #model 0 real 1
            tn += 1
        elif y_pred[i] != y_true[i] and y_pred[i] == 0: #model 0 real 0
            fn +=1  

    p = tp / (tp+fp)
    r = tp/(tp+fn)
    f1 = 2*p*r/(p+r)
    
    print("precision : {0}, recall : {1}, f1 : {2}".format(p,r,f1))

    return f1
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
result = model.fit(X_train,Y_train)
y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
print(evaluation(Y_test, y_pred))

from sklearn.naive_bayes import GaussianNB

bmodel = GaussianNB()
result = bmodel.fit(X_train,Y_train)
y_pred = bmodel.predict(X_test)

print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
print(evaluation(Y_test, y_pred))

from sklearn.neural_network import MLPClassifier
mlp_model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)

result = mlp_model.fit(X_train,Y_train)
y_pred = mlp_model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
print(evaluation(Y_test, y_pred))


from sklearn.model_selection import KFold
import numpy as np

def evaluate_model(model, X_train, Y_train, X_test, Y_test):
    
    result = model.fit(X_train,Y_train)
    y_pred = model.predict(X_test)
    
    print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
    
    f_score = evaluation(Y_test, y_pred)
    
    return f_score



kf = KFold(n_splits=5,random_state=None, shuffle=False)
count = 1
f_measure = [['Decision Tree'],['Naive Bays'],['Neuron Network']]
          
for train_index, test_index in kf.split(data):
    print("\n\n\n\nFold: {0}\n".format(count))
    count = count + 1
    f_score = 0
    X_train, X_test = data.iloc[train_index], data.iloc[test_index]
    Y_train, Y_test = target.iloc[train_index], target.iloc[test_index]
    
    Y_test = np.array(Y_test)

    
    dmodel = DecisionTreeClassifier()
    bmodel = GaussianNB()
    mlp_model = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
    
    
    print("Decision Tree: ")
    f_score = evaluate_model(dmodel, X_train, Y_train, X_test, Y_test)
    f_measure[0].append(f_score)
    
    print("\nNaive Bays:")
    f_score = evaluate_model(bmodel, X_train, Y_train, X_test, Y_test)
    f_measure[1].append(f_score)
    
    print("\nNeuron Network:")
    f_score = evaluate_model(mlp_model, X_train, Y_train, X_test, Y_test)
    f_measure[2].append(f_score)
import numpy as np
meanf = []
for i in range(0,len(f_measure)):
    meanf.append([f_measure[i][0], np.mean(f_measure[i][1:])])

pd.DataFrame(meanf,columns=['Predictor','f_measure_mean'])
