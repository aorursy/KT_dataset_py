import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt

% matplotlib inline



import warnings

warnings.filterwarnings('ignore')



import seaborn as sns

sns.set(color_codes=True)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/mushrooms.csv')
df.shape
df.head()
# This dataset is ready for exploration, no data cleaning required 

df.info()
# Class Distribuition

sns.countplot(x="class", data=df, palette="Greens_d")
class_dist = df['class'].value_counts()



print(class_dist)
prob_e = class_dist[0]/(class_dist[0]+class_dist[1])

prob_p = 1 - prob_e

print(prob_e)

print(prob_p)
from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()

for col in df.columns:

    df[col] = labelencoder.fit_transform(df[col])

 

df.head()
colormap = plt.cm.viridis

plt.figure(figsize=(15,15))

plt.title('Pearson Correlation of Features', size=15)



sns.heatmap(df.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
# sns.pairplot(df)
X = df.drop('class', axis=1)

y = df['class']

RS = 123



# Split dataframe into training and test/validation set 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RS)
from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from xgboost import XGBClassifier

import xgboost



classifiers = [

    KNeighborsClassifier(3),

    SVC(kernel="rbf", C=0.025, probability=True),

    NuSVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    XGBClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis()]
# Logging for Visual Comparison

log_cols=["Classifier", "Accuracy", "Log Loss"]

log = pd.DataFrame(columns=log_cols)



for clf in classifiers:

    clf.fit(X_train, y_train)

    name = clf.__class__.__name__

    

    print("="*30)

    print(name)

    

    print('****Results****')

    train_predictions = clf.predict(X_test)

    acc = accuracy_score(y_test, train_predictions)

    print("Accuracy: {:.4%}".format(acc))

    

    train_predictions = clf.predict_proba(X_test)

    ll = log_loss(y_test, train_predictions)

    print("Log Loss: {}".format(ll))

    

    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)

    log = log.append(log_entry)

    

print("="*30)
sns.set_color_codes("muted")

sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")



plt.xlabel('Accuracy %')

plt.title('Classifier Accuracy')

plt.show()



sns.set_color_codes("muted")

sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")



plt.xlabel('Log Loss')

plt.title('Classifier Log Loss')

plt.show()
# Inspect the learned Decision Trees

# One of the major advantage of Decision Trees is the fact that they can easily be interpreted.  

clf = DecisionTreeClassifier()



# Fit with all the training set

clf.fit(X, y)
importances = clf.feature_importances_

indices = np.argsort(importances)[::-1]

feature_names = X.columns



print("Feature ranking:")

for f in range(X.shape[1]):

    print("%s : (%f)" % (feature_names[f] , importances[indices[f]]))
f, ax = plt.subplots(figsize=(15, 15))

plt.title("Feature ranking", fontsize = 12)

plt.bar(range(X.shape[1]), importances[indices],

    color="b", 

    align="center")

plt.xticks(range(X.shape[1]), feature_names)

plt.xlim([-1, X.shape[1]])

plt.ylabel("importance", fontsize = 18)

plt.xlabel("index of the feature", fontsize = 18)