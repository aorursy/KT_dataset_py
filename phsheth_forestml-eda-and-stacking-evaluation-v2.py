# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import required librarues



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split



from sklearn.preprocessing import RobustScaler

from sklearn.preprocessing import PowerTransformer

from sklearn.preprocessing import QuantileTransformer



from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.multiclass import OneVsRestClassifier



from sklearn.ensemble import ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier





from mlxtend.classifier import StackingCVClassifier

from sklearn.model_selection import cross_val_score



from sklearn.metrics import accuracy_score
train = pd.read_csv("/kaggle/input/learn-together/train.csv")

test = pd.read_csv("/kaggle/input/learn-together/test.csv")
# Remove the Labels and make them y

y = train['Cover_Type']



# Remove label from Train set

X = train.drop(['Cover_Type'],axis=1)



# Rename test to text_X

test_X = test







# split data into training and validation data

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=42)



X = X.drop(['Id'], axis = 1)

train_X = train_X.drop(['Id'], axis = 1)

val_X = val_X.drop(['Id'], axis = 1)

test_X = test_X.drop(['Id'], axis = 1)
train_X.describe()
val_X.describe()
sns.distplot(train_X['Elevation'], label = 'train_X')

sns.distplot(val_X['Elevation'], label = 'val_X')

sns.distplot(test_X['Elevation'], label = 'test_X')

plt.legend()

plt.title('Elevation')

plt.show()
sns.distplot(train_X['Aspect'], label = 'train_X')

sns.distplot(val_X['Aspect'], label = 'val_X')

sns.distplot(test_X['Aspect'], label = 'test_X')

plt.title('Aspect')

plt.legend()

plt.show()
sns.distplot(train_X['Horizontal_Distance_To_Hydrology'], label = 'train_X')

sns.distplot(val_X['Horizontal_Distance_To_Hydrology'], label = 'val_X')

sns.distplot(test_X['Horizontal_Distance_To_Hydrology'], label = 'test_X')

plt.title('Horizontal_Distance_To_Hydrology')

plt.legend()

plt.show()
sns.distplot(train_X['Vertical_Distance_To_Hydrology'], label = 'train_X')

sns.distplot(val_X['Vertical_Distance_To_Hydrology'], label = 'val_X')

sns.distplot(test_X['Vertical_Distance_To_Hydrology'], label = 'test_X')

plt.title('Vertical_Distance_To_Hydrology')

plt.legend()

plt.show()
sns.distplot(train_X['Horizontal_Distance_To_Roadways'], label = 'train_X')

sns.distplot(val_X['Horizontal_Distance_To_Roadways'], label = 'val_X')

sns.distplot(test_X['Horizontal_Distance_To_Roadways'], label = 'test_X')

plt.title('Horizontal_Distance_To_Roadways')

plt.legend()

plt.show()
sns.distplot(train_X['Hillshade_9am'], label = 'train_X')

sns.distplot(val_X['Hillshade_9am'], label = 'val_X')

sns.distplot(test_X['Hillshade_9am'], label = 'test_X')

plt.title('Hillshade_9am')

plt.legend()

plt.show()
### define the classifiers

### Parameters from :https://www.kaggle.com/joshofg/pure-random-forest-hyperparameter-tuning



classifier_rf = RandomForestClassifier(n_estimators = 719,

                                       max_features = 0.3,

                                       max_depth = 464,

                                       min_samples_split = 2,

                                       min_samples_leaf = 1,

                                       bootstrap = False,

                                       random_state=42)

classifier_xgb = OneVsRestClassifier(XGBClassifier(n_estimators = 719,

                                                   max_depth = 464,

                                                   random_state=42))

classifier_et = ExtraTreesClassifier(random_state=42)
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

sclf = StackingCVClassifier(classifiers=[classifier_rf,

                                         classifier_xgb,

                                         classifier_et],

                            use_probas=True,

                            meta_classifier=classifier_rf)







labels = ['Random Forest', 'XGBoost', 'ExtraTrees', 'MetaClassifier']









for clf, label in zip([classifier_rf, classifier_xgb, classifier_et, classifier_rf], labels):

    scores = cross_val_score(clf, train_X.values, train_y.values.ravel(),

                             cv=5,

                             scoring='accuracy')

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
sclf.fit(train_X.values, train_y.values.ravel())


val_pred = sclf.predict(val_X.values)
acc = accuracy_score(val_y, val_pred)

print(acc)
sclffin = StackingCVClassifier(classifiers=[classifier_rf,

                                            classifier_xgb,

                                            classifier_et],

                               use_probas=True,

                               meta_classifier=classifier_rf)
sclffin.fit(X.values, y.values.ravel())
test_ids = test["Id"]

test_pred = sclffin.predict(test_X.values)
# Save test predictions to file

output = pd.DataFrame({'Id': test_ids,

                       'Cover_Type': test_pred})

output.to_csv('submission.csv', index=False)