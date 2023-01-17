# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt



mush = pd.read_csv('../input/mushrooms.csv')
mush.head()
mush.columns
#checking for missing values

mush.isnull().sum()
#Checking for unique parameters in the label column. i.e. 'class'

mush['class'].unique()
#clearly we have two data types. p = poisonous, and e =. edible.

#next we check for class imbalance



mush['class'].value_counts()
#an important step is to draw a graph to understand hw the various categorical variables  affect the label

# to simplify, I convert [p,e] to [0,1] so i can easily see which features are meaningful to the label



mush['class'] = mush['class'].replace(['p','e'], [0,1])
mush.head()
#so now that we've converted the label to numerical, lets plot a par plot to understand out categorical features



import numpy as np

cat_cols = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 

                  'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 

                  'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 

                  'stalk-surface-below-ring','stalk-color-above-ring','stalk-color-below-ring', 'veil-type','veil-color',

          'ring-number','ring-type','spore-print-color','population','habitat' ]



mush['dummy'] = np.ones(shape = mush.shape[0])

for col in cat_cols:

    print(col)

    counts = mush[['dummy', 'class', col]].groupby(['class', col], as_index = False).count()

    temp = counts[counts['class'] == 1][[col, 'dummy']]

    _ = plt.figure(figsize = (10,4))

    plt.subplot(1, 2, 1)

    temp = counts[counts['class'] == 1][[col, 'dummy']]

    plt.bar(temp[col], temp.dummy)

    plt.xticks(rotation=90)

    plt.title('Counts for ' + col + '\n edible')

    plt.ylabel('count')

    plt.subplot(1, 2, 2)

    temp = counts[counts['class'] == 0][[col, 'dummy']]

    plt.bar(temp[col], temp.dummy)

    plt.xticks(rotation=90)

    plt.title('Counts for ' + col + '\n posinous')

    plt.ylabel('count')

    plt.show()
# This process of drawing graph is important for feature selection. we do this to select the important variables especially in areas where we have a number of features and all might not be necessary

# There is a lot of information that we can deduce from the graphs. The key to interpreting these plots is comparing the proportion of the categories for each of the label values. If these proportions are distinctly different for each label category, the feature is likely to be useful in separating the label.



# Example

#1. Veil type is not likeky to be a significant feature in predicting the label as well as gill-attachement.

#2. The rest of the features have significant variations and is very significant in predicting the label.

features = mush.drop(['class', 'dummy'], axis = 1)
features.head()
label = mush['class']
label.shape, features.shape
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

import sklearn.model_selection as ms



le = LabelEncoder() 

for col in features.columns:

    features[col] = le.fit_transform(features[col])
features=pd.get_dummies(features,columns=features.columns,drop_first=True)

features.head()
X = features

y = label



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.metrics import confusion_matrix,classification_report
import sklearn.metrics as sklm

import numpy.random as nr

from sklearn import linear_model



logistic_mod = linear_model.LogisticRegression(class_weight = "balanced")

#performing logistic regression

nr.seed(123)

inside = ms.KFold(n_splits=10, shuffle = True)

nr.seed(321)

outside = ms.KFold(n_splits=10, shuffle = True)

nr.seed(3456)

param_grid = {"C": [0.1, 1, 10, 100, 1000]}

clf = ms.GridSearchCV(estimator = logistic_mod, param_grid = param_grid, 

                      cv = inside, # Use the inside folds

                      scoring = 'roc_auc',

                      return_train_score = True)

clf.fit(X, y)

clf.best_estimator_.C
logistic_mod = linear_model.LogisticRegression(C=clf.best_estimator_.C, class_weight="balanced") 

logistic_mod.fit( X_train, y_train)

print(logistic_mod.intercept_)

def print_metrics(labels, scores):

    metrics = sklm.precision_recall_fscore_support(labels, scores)

    conf = sklm.confusion_matrix(labels, scores)

    print('                 Confusion matrix')

    print('                 Score positive    Score negative')

    print('Actual positive    %6d' % conf[0,0] + '             %5d' % conf[0,1])

    print('Actual negative    %6d' % conf[1,0] + '             %5d' % conf[1,1])

    print('')

    print('Accuracy  %0.2f' % sklm.accuracy_score(labels, scores))

    print(' ')

    print('           Positive      Negative')

    print('Num case   %6d' % metrics[3][0] + '        %6d' % metrics[3][1])

    print('Precision  %6.2f' % metrics[0][0] + '        %6.2f' % metrics[0][1])

    print('Recall     %6.2f' % metrics[1][0] + '        %6.2f' % metrics[1][1])

    print('F1         %6.2f' % metrics[2][0] + '        %6.2f' % metrics[2][1])
probabilities = logistic_mod.predict_proba(X_test)

def score_model(probs, threshold):

    return np.array([1 if x > threshold else 0 for x in probs[:,1]])

threshold = 0.51

scores = score_model(probabilities, threshold)
print_metrics(y_test, scores)
from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



knn = KNN(n_neighbors = 6)

knn.fit(X_train, y_train)

print(knn.score(X_test, y_test))



y_pred = knn.predict(X_test)

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier



param_grid = {"max_features": [2, 3, 5, 10, 13], "min_samples_leaf":[3, 5, 10, 20]}

nr.seed(3456)

rf_clf = RandomForestClassifier(class_weight = "balanced")

nr.seed(4455)

rf_clf = ms.GridSearchCV(estimator = rf_clf, param_grid = param_grid, 

                      cv = inside, # Use the inside folds

                      scoring = 'roc_auc',

                      return_train_score = True)

rf_clf.fit(features, label)

print(rf_clf.best_estimator_.max_features)

print(rf_clf.best_estimator_.min_samples_leaf)
# There were to many warnings making the code result a bit longer but the best estimat
import sklearn.metrics as sklm

nr.seed(1115)

rf_mod = RandomForestClassifier(class_weight = "balanced", 

                                max_features = rf_clf.best_estimator_.max_features, 

                                min_samples_leaf = rf_clf.best_estimator_.min_samples_leaf) 

rf_mod.fit(X_train, y_train)

probabilities = rf_mod.predict_proba(X_test)

scores = score_model(probabilities, 0.54)

print_metrics(y_test, scores)     