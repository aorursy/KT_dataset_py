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
import numpy as np



from scipy.stats import uniform, randint



from sklearn.datasets import load_breast_cancer, load_diabetes, load_wine

from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split



import xgboost as xgb
def display_scores(scores):

    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(scores, np.mean(scores), np.std(scores)))
def report_best_scores(results, n_top=3):

    for i in range(1, n_top + 1):

        candidates = np.flatnonzero(results['rank_test_score'] == i)

        for candidate in candidates:

            print("Model with rank: {0}".format(i))

            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(

                  results['mean_test_score'][candidate],

                  results['std_test_score'][candidate]))

            print("Parameters: {0}".format(results['params'][candidate]))

            print("")
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
train_df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], 1, inplace=True)
cat_replace = {'Sex': {'male': 1, 'female': 0}}
train_df.replace(cat_replace, inplace=True)
# Now we one hot encode the Embarked category

train_df = pd.get_dummies(train_df, columns=["Embarked"])
train_df.head(50)
train_x = train_df.drop('Survived', 1)

train_y = train_df['Survived']
xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

xgb_model.fit(train_x, train_y)
y_pred = xgb_model.predict(train_x)
print(confusion_matrix(y_pred, train_y))
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
test_df.head()
submission = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
submission.head()
test_df.drop(['Name', 'PassengerId', 'Ticket', 'Cabin'], 1, inplace=True)



test_df.replace(cat_replace, inplace=True)



# Now we one hot encode the Embarked category

test_df = pd.get_dummies(test_df, columns=["Embarked"])
test_df.head()
y_test = xgb_model.predict(test_df)
submission['Survived'] = y_test
submission.head(50)
submission.to_csv('/kaggle/working/submission.csv', index=False)