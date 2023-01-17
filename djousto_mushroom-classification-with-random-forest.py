# Pandas

import pandas as pd

pd.set_option('display.max_columns', 500)



# visualisation libs

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib as mpl



# and some options

import warnings

warnings.filterwarnings("ignore")

%matplotlib inline

sns.set(style="whitegrid", color_codes=True)

sns.set(rc={'figure.figsize':(15,10)})
# pandas lib to read data and manipulate DataFrames

data = pd.read_csv('../input/mushrooms.csv')

# what's this data looking like

data.head()
data.info()
from sklearn import preprocessing

le = preprocessing.LabelEncoder

dataEnc = data.apply(le().fit_transform)
y = dataEnc["class"]

X = dataEnc.drop(['class'],axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=10, max_features='auto', max_leaf_nodes=None,

            min_impurity_decrease=0.0, min_impurity_split=None,

            min_samples_leaf=1, min_samples_split=2,

            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=None,

            oob_score=False, random_state=0, verbose=0, warm_start=False)



_ = clf.fit(X_train, y_train)  

y_pred = clf.predict(X_test)
# Import necessary modules

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix



# Generate the confusion matrix and classification report

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
feats = dict(zip(X.columns,clf.feature_importances_))
sns.set(rc={'figure.figsize':(15,10)})

_ = plt.bar(range(len(feats)), list(feats.values()), align='center')

_ = plt.xticks(range(len(feats)), list(feats.keys()),rotation=90)