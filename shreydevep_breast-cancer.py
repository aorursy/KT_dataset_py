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
train_data = pd.read_csv('../input/breast-cancer/train.csv')

test_data = pd.read_csv('../input/breast-cancer/test.csv')
train_data.info()
X = train_data.drop(['class'],axis = 1)

X.drop(['Id'],axis = 1, inplace = True)

y = train_data['class']
X_sub = test_data.drop(['Id'],axis = 1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state= 1)
def LogisticRegression(X_train, y_train,X_test_model):

    from sklearn.linear_model import LogisticRegressionCV

    clf = LogisticRegressionCV(cv = 5,max_iter = 1000, random_state=0).fit(X_train, y_train)

    y_pred_LR = clf.predict(X_test_model)

    print(clf.score(X_train,y_train))

    return y_pred_LR

y_pred_LR = LogisticRegression(X_train, y_train,X_test)
def RandomForest(X_train, y_train, X_test_pass):

    from sklearn.ensemble import RandomForestClassifier

    #X, y = make_classification(n_samples=1000, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=False)

    model = RandomForestClassifier(max_depth=2, random_state=0)

    #model = RandomForestClassifier(n_estimators=1000,min_samples_split=25, max_depth=7, max_features=2)

    model.fit(X_train,y_train)

    y_pred_RP = model.predict(X_test_pass)

    feature_importances = pd.DataFrame(model.feature_importances_,index = X_train.columns,columns=['importance']).sort_values('importance',ascending=False)

    print(feature_importances)

    print(model.score(X_train,y_train))

    return y_pred_RP

y_pred_RP = RandomForest(X_train, y_train,X_test)

feature_importances = ['uniformity_of_cell_shape','uniformity_of_cell_size','bare_nuclei']

X_train2 = pd.DataFrame() 

X_test2 = pd.DataFrame() 

for feature in feature_importances:

    X_train2[feature] = X_train[feature]

    X_test2[feature] = X_test[feature]

y_pred_important_RP = RandomForest(X_train2,y_train,X_test2)

#y_pred_important_LR = LogisticRegression(X_train2, y_train, X_test2)
def score_model(y_pred_model,y_test):

    from sklearn.metrics import accuracy_score

    score = accuracy_score(y_pred_model,y_test)

    return score

score_model(y_pred_RP,y_test)
score_list = [score_model(y_pred_LR,y_test), score_model(y_pred_RP,y_test), score_model(y_pred_important_RP,y_test)]

for score in score_list:

    print(score)
y_pred_sub = RandomForest(X_train,y_train,X_sub)

y_pred_sub = pd.DataFrame({'Class': y_pred_sub[:]})
output = pd.DataFrame({'Id': test_data['Id'], 'Class': y_pred_sub['Class']})
output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")