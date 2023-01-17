import numpy as np

import pandas as pd
from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns

train = pd.read_csv('../input/titanic-machine-learning-from-disaster/train.csv')

test = pd.read_csv('../input/titanic-machine-learning-from-disaster/test.csv')
PassengerId = test['PassengerId']
train.head()
train.info()
test.head()
train.info()
Xtrain = train.drop(["Cabin", "Ticket", "PassengerId" ,"Survived"], axis=1)

Xtest = test.drop(["Cabin", "Ticket",  "PassengerId"], axis=1)

#Xtrain['Survived'] = False

Xtrain['is_test'] = False

Xtest['is_test'] = True

X = pd.concat([Xtrain, Xtest], axis=0)

#X.index = range(len(X))

X.columns = X.columns.str.lower()
X.head()
train.Survived.head()
Xtrain.shape
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

y = le.fit_transform(train.Survived)
list(le.classes_)
X.info()
X.age.fillna(X.age.median(), inplace=True)
X.embarked.value_counts()
X.fare.fillna(X.fare.median(), inplace=True)
X.info()
X['sex'] = (X.sex=='male').astype(int)
X.head()
X = pd.get_dummies(X, columns=['embarked'])
X.head()
from collections import Counter
one_big_text = " ".join(X.name)

words = one_big_text.replace('/',' / ').split()

most_common = Counter(words).most_common()

most_common[:20]
Xname = pd.DataFrame()

for col, num in most_common[:10]:

    Xname[col] = X[~X.is_test].name.str.contains(col).astype(int)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=4)
clf.fit(Xname, train.Survived)
clf.feature_importances_
Xname.columns
X['is_Mrs'] = X.name.str.contains('Mrs. ').astype(int)

X['is_Mr'] = X.name.str.contains('Mr. ').astype(int)

X['is_Miss'] = X.name.str.contains('Miss.').astype(int)

X.info()
Xtrain_prep = X[X.is_test==False].drop(['is_test', 'name'], axis=1)

Xtest_prep = X[X.is_test==True].drop(['is_test', 'name'], axis=1)
Xtest_prep.head()
from sklearn.model_selection import GridSearchCV
depths = np.arange(1,10)

#features_num = np.arange(5,15)

grid = {'max_depth': depths}

#, 'max_features': features_num}

gridsearch = GridSearchCV(DecisionTreeClassifier(), grid, scoring='neg_log_loss', cv=5)
%%time

gridsearch.fit(Xtrain_prep, y)
results = pd.DataFrame(gridsearch.cv_results_)

results[['mean_test_score','std_test_score','params']].sort_values(by='mean_test_score', ascending=False)
clf_final = DecisionTreeClassifier(max_depth=3)
clf_final.fit(Xtrain_prep, y)
y_pred_proba = clf_final.predict_proba(Xtest_prep)

y_pred = clf_final.predict(Xtest_prep)
submission = pd.DataFrame({

        "PassengerId": PassengerId,

        "Survived": y_pred

    })

submission.to_csv('submission.csv', index=False)
from sklearn.tree import export_graphviz

with open("tree1.dot", 'w') as f:

     f = export_graphviz(clf_final,

                              out_file=f,

                              max_depth = 3,

                              impurity = True,

                              feature_names = list(Xtrain_prep),

                              class_names = ['Died', 'Survived'],

                              rounded = True,

                              filled= True )

        
def get_tree_dot_view(clf, feature_names=None, class_names=None):

    print(export_graphviz(clf, out_file=None, filled=True, feature_names=feature_names, class_names=class_names))
#get_tree_dot_view(clf_final, list(Xtrain_prep.columns), ['Died', 'Survived'])
import pydot



(graph,) = pydot.graph_from_dot_file('tree1.dot')

graph.write_png('tree1.png')
from PIL import Image, ImageDraw, ImageFont

from IPython.display import Image as PImage

img = Image.open("tree1.png")

img.save('sample-out.png')

PImage("sample-out.png")
acc_decision_tree = round(clf_final.score(Xtrain_prep, y) * 100, 2)

acc_decision_tree