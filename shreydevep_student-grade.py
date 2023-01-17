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
data = pd.read_csv('../input/student-alcohol-consumption/student-mat.csv')
data.info()
data.isnull().sum()
object_type_features = data.select_dtypes('object').columns

print(object_type_features)
for object_feature in object_type_features:

    print(object_feature,"=",data[object_feature].unique().size)
from sklearn import preprocessing

le = preprocessing.LabelEncoder()

for object_feature in object_type_features:

    data[object_feature] = le.fit_transform(data[object_feature])

data.info()
X = data.drop(['G3'], axis = 1)

y1 = data['G1']

y2 = data['G2']

y3 = data['G3']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y3, test_size=0.30 )
def score_model(y_pred_model,y_test):

    from sklearn.metrics import accuracy_score

    score = accuracy_score(y_test,y_pred_model)

    return score
def LogisticRegression(X_train, y_train,X_test_model):

    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter = 10000, random_state=0).fit(X_train, y_train)

    y_pred_LR = clf.predict(X_test_model)

    print("Logistic Regression")

    print("Training Accuracy" , clf.score(X_train,y_train)*100)

    print("Test Accuracy", score_model(y_pred_LR, y_test)*100)

    return y_pred_LR

LogisticRegression(X_train,y_train,X_test)
def RandomForest(X_train, y_train, X_test_pass):

    from sklearn.ensemble import RandomForestClassifier

    #X, y = make_classification(n_samples=1000, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=False)

    #model = RandomForestClassifier(n_estimators=1000,max_depth=2, random_state=0)

    model = RandomForestClassifier(n_estimators=1000)

    model.fit(X_train,y_train)

    y_pred_RP = model.predict(X_test_pass)

    #feature_importances = pd.DataFrame(model.feature_importances_,index = X_train.columns,columns=['importance']).sort_values('importance',ascending=False)

    #print(feature_importances)

    print("Random Forest ")

    print("Training Accuracy" , model.score(X_train,y_train)*100)

    print("Test Accuracy", score_model(y_pred_RP, y_test)*100)

    return y_pred_RP
RandomForest(X_train,y_train,X_test)
def DecisionTree(X_train, y_train,X_test_model):

    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier().fit(X_train, y_train)

    y_pred_LR = clf.predict(X_test_model)

    print("DecisionTree")

    print("Training Accuracy" , clf.score(X_train,y_train)*100)

    print("Test Accuracy", score_model(y_pred_LR, y_test)*100)

    return y_pred_LR

DecisionTree(X_train,y_train,X_test)
def NaiveBayesClf(X_train,y_train,X_test_model):

    from sklearn.naive_bayes import GaussianNB

    clf = GaussianNB().fit(X_train, y_train)

    y_pred_NB = clf.predict(X_test_model)

    print("Naive Bayes Classifier ")

    print("Training Accuracy" , clf.score(X_train,y_train)*100)

    print("Test Accuracy", score_model(y_pred_NB, y_test)*100)

    return y_pred_NB

    
NaiveBayesClf(X_train, y_train, X_test)