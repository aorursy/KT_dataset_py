import numpy as np

import pandas as pd

from IPython.display import display

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



%matplotlib inline



df = pd.read_csv("../input/HR_comma_sep.csv")



display(df.head())
# Detecting patterns in the data of employees who left

df_leave = df[df.left != 0]

red_df_leave = df_leave.drop(['number_project','time_spend_company','Work_accident',

                              'left','promotion_last_5years'], axis=1)



sns.pairplot(red_df_leave.sample(n=500,random_state=0), hue='salary', hue_order=['low','medium','high'], size=2.5)

plt.show()
# Detecting patterns in the data of employees who stayed

df_stay = df[df.left != 1]

red_df_stay = df_stay.drop(['number_project','time_spend_company','Work_accident',

                              'left','promotion_last_5years'], axis=1)

sns.pairplot(red_df_stay.sample(n=500,random_state=0), hue="salary", hue_order=['low','medium','high'], size=2.5)

plt.show()
from sklearn import model_selection



num_df = pd.get_dummies(df)

X = num_df.drop('left', axis = 1)

y = num_df['left']



X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=0)
from sklearn import tree



dt_clf = tree.DecisionTreeClassifier(max_depth=1)

dt_clf.fit(X_train, y_train)

tree_score = dt_clf.score(X_test, y_test)

print(tree_score)
import xgboost as xgb

from sklearn.metrics import accuracy_score



accuracy = []

n_est = [1,10,50,100,200]



for i in n_est:

    gbm = xgb.XGBClassifier(max_depth=1, n_estimators=i, learning_rate=0.5).fit(X_train, y_train)

    accuracy.append(accuracy_score(y_test, gbm.predict(X_test)))
plt.plot(n_est, accuracy, '-o')

plt.plot((0, 200), (tree_score, tree_score), '-r')

plt.xlabel("Number of XGBoost estimators")

plt.ylabel("Accuracy")

plt.title("XGBoost vs. Simple Decision Tree")

plt.show()