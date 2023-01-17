# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
women = train_data.loc[train_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
b = sns.countplot(x='Survived', data=train_data)

b.set_title("Survived Distribution");
b = sns.countplot(x='Pclass', data=train_data)

b.set_title("Survived Distribution");
pd.crosstab(train_data['Survived'], train_data['Pclass']).plot(kind="bar", figsize=(10,6))
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
test_data['Age'].fillna(round((test_data['Age'].mean())), inplace = True)

train_data['Age'].fillna(round((train_data['Age'].mean())), inplace = True)
train_data['Cabin'] = train_data['Cabin'].fillna("Missing")

train_data['Cabin'] = train_data['Cabin'].fillna("Missing")
#test_data['Fare'].fillna(round((test_data['Fare'].mean())), inplace = True)

#train_data['Fare'].fillna(round((train_data['Fare'].mean())), inplace = True)
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].mean())
train_data.isna().sum()
train_data.isna().sum()
train_data.shape
train_data = train_data.drop(columns=['Name'], axis=1)

test_data = test_data.drop(columns=['Name'], axis=1)
sex_mapping = {

    'male': 0,

    'female': 1

}



train_data.loc[:, "Sex"] = train_data['Sex'].map(sex_mapping)

test_data.loc[:, "Sex"] = test_data['Sex'].map(sex_mapping)
train_data = train_data.drop(columns=['Ticket'], axis=1)

test_data = test_data.drop(columns=['Ticket'], axis=1)
train_data = train_data.drop(columns=['Cabin'], axis=1)

test_data = test_data.drop(columns=['Cabin'], axis=1)
test_data['Embarked'].value_counts()
train_data = pd.get_dummies(train_data, prefix_sep="__",

                              columns=['Embarked'])

test_data = pd.get_dummies(test_data, prefix_sep="__",

                              columns=['Embarked'])
X = train_data.drop("Survived", axis=1)



# Target variable

y = train_data['Survived'].values
from sklearn.model_selection import train_test_split



from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression



from sklearn.metrics import roc_auc_score



from sklearn.model_selection import RandomizedSearchCV







np.random.seed(42)



# Split into train & test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
# Put models in a dictionary

models = {"KNN": KNeighborsClassifier(),

          "Logistic Regression": LogisticRegression(max_iter=10000), 

          "Random Forest": RandomForestClassifier(),

          "SVC" : SVC(probability=True),

          "DecisionTreeClassifier" : DecisionTreeClassifier(),

          "AdaBoostClassifier" : AdaBoostClassifier(),

          "GradientBoostingClassifier" : GradientBoostingClassifier(),

          "GaussianNB" : GaussianNB(),

          "LinearDiscriminantAnalysis" : LinearDiscriminantAnalysis(),

          "QuadraticDiscriminantAnalysis" : QuadraticDiscriminantAnalysis()}



# Create function to fit and score models

def fit_and_score(models, X_train, X_test, y_train, y_test):

    """

    Fits and evaluates given machine learning models.

    models : a dict of different Scikit-Learn machine learning models

    X_train : training data

    X_test : testing data

    y_train : labels assosciated with training data

    y_test : labels assosciated with test data

    """

    # Random seed for reproducible results

    np.random.seed(42)

    # Make a list to keep model scores

    model_scores = {}

    # Loop through models

    for name, model in models.items():

        # Fit the model to the data

        model.fit(X_train, y_train)

        # Predicting target values

        y_pred = model.predict(X_test)

        # Evaluate the model and append its score to model_scores

        #model_scores[name] = model.score(X_test, y_test)

        model_scores[name] = roc_auc_score(y_pred, y_test)

    return model_scores
model_scores = fit_and_score(models=models,

                             X_train=X_train,

                             X_test=X_test,

                             y_train=y_train,

                             y_test=y_test)

model_scores
gdb = GradientBoostingClassifier()

gdb.fit(X_train, y_train)
y_pred = gdb.predict(test_data)

y_pred
sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

sub.head()
sub['Survived'] = y_pred

sub.to_csv("titanic_results.csv", index=False)

sub.head()