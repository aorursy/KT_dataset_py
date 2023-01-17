import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline





mush_data = pd.read_csv("../input/mushrooms.csv")

mush_data.head()
for i in mush_data.columns:

    print ('ATTR --- '+ i)

    print (mush_data[i].value_counts())

    print ('\n')
mush_data.describe()
#Check for Nan or null values

mush_data.apply(lambda x: sum(x.isnull()))
sns.countplot('cap-shape',data=mush_data,hue='class')
sns.countplot('cap-surface',data=mush_data,hue='class')
sns.countplot('cap-surface',data=mush_data,hue='cap-shape')
sns.countplot('odor',data=mush_data,hue='class')

edible=mush_data[mush_data['class']=='e']['cap-surface'].value_counts()

poison = mush_data[mush_data['class']=='p']['cap-surface'].value_counts()



ff = pd.DataFrame([edible,poison])

ff.index=['Edible','Poison']

ff.plot(kind='bar',stacked=True,figsize=(12,8))
# let's convert the data into dummy format



def make_dummy(data,attr):

    dummy = pd.get_dummies(data[attr],prefix=attr)

    data = pd.concat([data,dummy],axis=1)

    data.drop(attr,axis=1,inplace=True)

    return data
data_temp = mush_data.drop('class',axis=1)

cols = data_temp.columns

for attr in cols:

    mush_data = make_dummy(mush_data,attr)   
mush_data.head()
X = mush_data.iloc[:,1:]

Y = mush_data.iloc[:,0]
Y.head()
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel



clf = RandomForestClassifier(n_estimators=50,max_features='sqrt')

clf.fit(X,Y)



features = pd.DataFrame(dtype='object')

features['feature'] = X.columns

features['importance'] = clf.feature_importances_

features.sort_values(by=['importance'], ascending=True, inplace=True)

features.set_index('feature', inplace=True)



features.plot(kind='barh', figsize=(30, 30))
# we can see that many features in the last part have zero importanxce 

# so we have to reduce the features to make data functionality faster

model = SelectFromModel(clf, prefit=True)

train_reduced = model.transform(X)

train_reduced.shape
# we can see the important features reduced mush

#import necessary library

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

num_folds = 10

seed = 7

scoring = 'accuracy'

validation_size = 0.20

seed = 7



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_size, random_state=seed)
from sklearn.cross_validation import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier



random_forest_classifier = RandomForestClassifier()



parameter_grid = {'n_estimators': [5, 10, 25, 50],

                  'criterion': ['gini', 'entropy'],

                  'max_features': ['sqrt', 'auto', 'log2'],

                  'warm_start': [True, False]}



cross_validation = StratifiedKFold(Y, n_folds=10)



grid_search = GridSearchCV(random_forest_classifier,

                           param_grid=parameter_grid,

                           cv=cross_validation)



grid_search.fit(X, Y)



print('Best score: {}'.format(grid_search.best_score_))

print('Best parameters: {}'.format(grid_search.best_params_))



grid_search.best_estimator_
model_ = grid_search.best_estimator_

model_.fit(X, Y)



xval = cross_val_score(model_, X,Y, cv = 10, scoring='accuracy')

np.mean(xval)
random_forest = grid_search.best_estimator_

Y_pred = random_forest.predict(X_test)
submission = pd.DataFrame({

        "class": Y_pred

    })



submission.to_csv("mushroom_final.csv",index=False, encoding='utf-8')
submission.head()