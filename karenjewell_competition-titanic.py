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

        file = os.path.join(dirname, filename)

        print(file)

        filename.split('.')[0] = pd.read_csv(file)

        

        test = pd.read_csv(r"/kaggle/input/titanic/test.csv")

        train = pd.read_csv(r"/kaggle/input/titanic/train.csv")

    



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

import sklearn

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_validate
print(train.head(5))
train_t = train.drop(columns={"PassengerId","Name","Ticket","Cabin"})

list(train_t)
train_t = pd.concat([train_t.drop('Sex', axis=1), pd.get_dummies(train_t['Sex'])], axis=1)

train_t = pd.concat([train_t.drop('Embarked', axis=1), pd.get_dummies(train_t['Embarked'])], axis=1)

train_t = train_t.fillna('0')

print(train_t.head(5))
### Split train-test data

x = train_t.copy().drop(columns={'Survived'})

y = train_t['Survived']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

xy_train = list(zip(x_train, y_train))

xy_test = list(zip(x_test, y_test))



### Classifier models

dict_classifiers = {

    "Logreg": sklearn.linear_model.LogisticRegression(solver='lbfgs'),

    "GBC": sklearn.ensemble.GradientBoostingClassifier(),

    "DT": sklearn.tree.DecisionTreeClassifier(),

    "RF": sklearn.ensemble.RandomForestClassifier(max_depth=None, random_state=0),

    "NB": sklearn.naive_bayes.GaussianNB(),

}



results = ()

results=[['model','score']]



for model, model_inst in dict_classifiers.items():

    model_inst.fit(x_train, y_train)

    pred = np.array(model_inst.predict(x_test))

    print(model, "accuracy: ",metrics.accuracy_score(y_test, pred)*100)

    cv_results = cross_validate(model_inst, x, y, cv=3)

    print(cv_results['test_score'])

    #print(metrics.classification_report(y_test, pred, digits=3))

    

    results.append([model,metrics.accuracy_score(y_test, pred)*100])

    if model == 'RF':

        feat_importances = pd.Series(model_inst.feature_importances_, index=x.columns)

        feat_importances.nlargest(10).plot(kind='barh')

        plt.show()



print(results)
pd.DataFrame(results)


test_t = test.drop(columns={"PassengerId","Name","Ticket","Cabin"})

test_t = pd.concat([test_t.drop('Sex', axis=1), pd.get_dummies(test_t['Sex'])], axis=1)

test_t = pd.concat([test_t.drop('Embarked', axis=1), pd.get_dummies(test_t['Embarked'])], axis=1)

test_t = test_t.fillna('0')



final_clf = sklearn.ensemble.GradientBoostingClassifier()

final_clf.fit(x, y)

pred = np.array(final_clf.predict(test_t))

finalresults = pd.DataFrame()

finalresults['Survived'] = list(pred)

finalresults = finalresults.join(test[['PassengerId']])

print(finalresults)
finalresults.to_csv('results.csv',index=False)