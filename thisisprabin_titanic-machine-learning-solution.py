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
import warnings

warnings.filterwarnings('ignore')
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



%matplotlib inline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.datasets import make_classification

from sklearn.ensemble import ExtraTreesClassifier



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score

import xgboost as xgb

from sklearn.model_selection import GridSearchCV
# Label encoder



lable_encoder = LabelEncoder()

plt.style.use('ggplot')
df = pd.read_csv("/kaggle/input/titanic/train.csv")
df.head()
# Checking the null values in the data sets.



plt.figure()

plt.title("Total null values")



plt.bar(df.isna().sum().index, df.isna().sum().values)

plt.xlabel('Columns')

plt.ylabel('Null value counts')



plt.xticks(rotation=90)

plt.show()
len(df)
# Name of the columns.



df.columns
# Creating a copy of the data set with required columns.



tmp_df = df[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']].copy()
tmp_df.head()
# Filling the null values of age with median and zero.



def impute_na(df_, variable, median):

    df_[variable+'_median'] = df_[variable].fillna(median)

    df_[variable+'_zero'] = df_[variable].fillna(0) 
# Calculating the median of age.



median = tmp_df.Age.median()

median
# Applying the function, to fill the null values.



impute_na(tmp_df, 'Age', median)

tmp_df.head()
# creating a dataframe with median filled null values.



median_df = tmp_df[['Survived', 'Pclass', 'Sex', 'Age_median', 'SibSp', 

                    'Parch', 'Embarked']]
median_df.dropna(inplace=True)

median_df.isnull().sum()
# Encoding the categorical column



median_df['Sex'] = lable_encoder.fit_transform(median_df['Sex'])

median_df['Embarked'] = lable_encoder.fit_transform(median_df['Embarked'])
forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
# Defining x and y values median filled dataframe



x = median_df[['Pclass', 'Sex', 'Age_median', 'SibSp', 'Parch', 'Embarked']]

y = median_df['Survived']
forest.fit(x, y)
# Using random forest to extract the important features from median filled dataframe



importances = forest.feature_importances_

print(importances)



std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

print(std)



indices = np.argsort(importances)[::-1]

print(indices)





plt.figure()

plt.title("Feature importances - (median_df)")

plt.bar(range(x.shape[1]), importances[indices], 

        yerr=std[indices], align="center")



plt.xlabel("Index")

plt.ylabel("Importance")



plt.xticks(range(x.shape[1]), indices)

plt.xlim([-1, x.shape[1]])

plt.show()
# Takeing the important features which are extracted from median dataframe.



x_m = median_df[['Pclass', 'Sex', 'Age_median']]

y_m = median_df['Survived']
# Train / test split.



x_train_m, x_test_m, y_train_m, y_test_m = train_test_split(x_m, y_m, 

                                                            test_size = 0.25, 

                                                            random_state = 0)
# Scaling the data.



sc = StandardScaler()

x_train_m = sc.fit_transform(x_train_m)

x_test_m = sc.transform(x_test_m)
# We are using three algorithms as metioned below. and will compare the model 

# outcome one by one.



models = {

    "logistic_regression": LogisticRegression(random_state=0),

    "svm": SVC(random_state=0),

    "random_forest": RandomForestClassifier(n_estimators=250, criterion='entropy', 

                                            random_state=0)

}
# We will be using for loop to one by one apply the model.



for name, model in models.items():

    

    model.fit(x_train_m, y_train_m)

    y_pred_m = model.predict(x_test_m)

    

    print(name)

    print("Precision - {}".format(precision_score(y_test_m, y_pred_m, average='macro')))

    print("Recall - {}".format(recall_score(y_test_m, y_pred_m, average='macro')))

    print("Accuracy - {}".format(accuracy_score(y_test_m, y_pred_m)))



    cm = confusion_matrix(y_test_m, y_pred_m)

    print()

    print("Confusion matrix")

    print(cm)

    print()

    print("----------------------------------------")

    print()
xgboost = xgb.XGBClassifier(n_estimators=50, colsample_bytree=0.7, gamma=0.3)
xgboost.fit(x_train_m, y_train_m)
y_pred_m = xgboost.predict(x_test_m)
print("Precision - {}".format(precision_score(y_test_m, y_pred_m, average='macro')))

print("Recall - {}".format(recall_score(y_test_m, y_pred_m, average='macro')))

print("Accuracy - {}".format(accuracy_score(y_test_m, y_pred_m)))



cm = confusion_matrix(y_test_m, y_pred_m)

print()

print("Confusion matrix")

print(cm)

print()
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
test_df.isna().sum()
len(test_df)
# Calculating the median of age.



median = test_df.Age.median()

median
# Applying the function, to fill the null values.



impute_na(test_df, 'Age', median)

tmp_df.head()
test_tmp = test_df[['Pclass', 'Sex', 'Age_median']]
# Encoding the categorical column



test_tmp['Sex'] = lable_encoder.fit_transform(test_tmp['Sex'])
# Scaling the data.



test_tmp = sc.transform(test_tmp)
test_predict = xgboost.predict(test_tmp)
test_predict
rdc = RandomForestClassifier(n_estimators=250, criterion='entropy', random_state=0)
rdc.fit(x_train_m, y_train_m)
y_pred_m = rdc.predict(x_test_m)
print("Precision - {}".format(precision_score(y_test_m, y_pred_m, average='macro')))

print("Recall - {}".format(recall_score(y_test_m, y_pred_m, average='macro')))

print("Accuracy - {}".format(accuracy_score(y_test_m, y_pred_m)))



cm = confusion_matrix(y_test_m, y_pred_m)

print()

print("Confusion matrix")

print(cm)

print()
pred_rdc = rdc.predict(test_tmp)
len(pred_rdc)
pred_rdc
len(test_df[['PassengerId']])
submission = test_df[['PassengerId']]
submission['Survived'] = pred_rdc
# submission.to_csv('submission/submission_rdc.csv')