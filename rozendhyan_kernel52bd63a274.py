# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# for display dataframe

from IPython.display import display

from IPython.display import display_html

def display_side_by_side(*args):

    html_str=''

    for df in args:

        html_str+=df.to_html()

    display_html(html_str.replace('table','table style="display:inline"'),raw=True)

# ignore warning

import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


# loading package

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve, train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.feature_selection import RFECV







df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

df_data = df_train.append(df_test)

#submit = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
df_train.info()
df_test.info()
# Male to female survival ratio

sns.countplot(df_data['Sex'], hue=df_data['Survived'])

display(df_data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().round(3))
sns.countplot(df_data['Pclass'], hue=df_data['Survived'])

df_data[["Pclass", "Survived"]].groupby(['Pclass'], as_index=False).mean().round(3)

# Convert Sex

df_data['Sex_Code'] = df_data['Sex'].map({'female' : 1, 'male' : 0}).astype('int')
# Split training set the testing set

df_train = df_data[:len(df_train)]

df_test = df_data[len(df_train):]
# Input set and labels

X = df_train.drop(labels=['Survived', 'PassengerId'], axis=1)

Y = df_train['Survived']
# Show Baseline

Base = ['Sex_Code', 'Pclass']

Base_Model = RandomForestClassifier(random_state=2, n_estimators=250, min_samples_split=20, oob_score=True)

Base_Model.fit(X[Base], Y)

print('Base oob score :%.5f' %(Base_Model.oob_score_))
# submission if you want

# submits

X_Submit = df_test.drop(labels=['PassengerId'],axis=1)



Base_pred = Base_Model.predict(X_Submit[Base])



submit = pd.DataFrame({"PassengerId": df_test['PassengerId'],

                      "Survived":Base_pred.astype(int)})

submit.to_csv("submit_Base.csv",index=False)
from sklearn.preprocessing import Imputer

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline

from sklearn.feature_selection import VarianceThreshold

import xgboost

from xgboost import XGBClassifier

df_data = pd.get_dummies(df_data) 
# Split training set the testing set

df_train = df_data[:len(df_train)]

df_test = df_data[len(df_train):]



# Input set and labels

X = df_train.drop(labels=['Survived', 'PassengerId'], axis=1)

Y = df_train['Survived']
X
X_train, X_eval, Y_train, Y_eval = train_test_split(X, Y, test_size=0.3, random_state=42)
# step1. Imputation transformer for completing missing values.

step1 = ('Imputer', Imputer())

# step2. MinMaxScaler

step2 = ('MinMaxScaler', MinMaxScaler())

# step3. feature selection

#step3 = ('FeatureSelection', SelectFromModel(RandomForestRegressor()))

step3 = ('FeatureSelection', VarianceThreshold())



finally_step = ('model', RandomForestClassifier(random_state=2,n_estimators=1000,min_samples_split=20,oob_score=True))



pipeline = Pipeline(steps=[step1, step2, step3, finally_step])
pipeline.fit(X, Y)
print(f"Train score: {pipeline.score(X_train, Y_train)}")
import sklearn.metrics as metrics

from sklearn.metrics import mean_squared_error

from sklearn.feature_selection import VarianceThreshold



Y_pred = pipeline.predict(X_train)

VARIANCE_SCORE = metrics.explained_variance_score(Y_train, Y_pred)

MSLE = np.sqrt(metrics.mean_squared_error(Y_train, Y_pred))

r2 = metrics.r2_score(Y_train, Y_pred)



print("MSLE = ", MSLE)

#print("RMSLE = ", RMSLE)

print("VARIANCE_SCORE = ", VARIANCE_SCORE)

print("R2", r2)

scores = cross_val_score(pipeline, X, Y, cv=3)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# submits



X_Submit = df_test.drop(labels=['Survived', 'PassengerId'],axis=1)



pred = pipeline.predict(X_Submit)



submit = pd.DataFrame({"PassengerId": df_test['PassengerId'],

                      "Survived":pred.astype(int)})

submit.to_csv("submit_final.csv",index=False)


