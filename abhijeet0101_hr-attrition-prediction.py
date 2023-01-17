# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn import tree

from sklearn import metrics

from sklearn.decomposition import PCA

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

% matplotlib inline
train_df = pd.read_csv("../input/HR_comma_sep.csv")

train_df.head(5)
sns.boxplot(x="left", y= "satisfaction_level", data=train_df)

plt.show()
sns.boxplot(x="left", y= "last_evaluation", data=train_df)

plt.show()
sns.boxplot(x="left", y= "average_montly_hours", data=train_df)

plt.show()
sns.boxplot(x="left", y= "number_project", data=train_df)

plt.show()
sns.boxplot(x="left", y="time_spend_company", data=train_df)

plt.show()
train_df["salary"] = train_df["salary"].apply(lambda salary: 0 if salary == 'low' else 1)
#train_df.salary=train_data.salary.replace({'low':0,'high':1})

train_df.head(5)
train_df.Department.unique()
y = train_df["left"]

#drop department & left

columns = ['Department', 'left']

train_df = train_df.drop(columns, axis=1)

col = train_df.columns

X = train_df[col]

X

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42)
my_d_tree = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)

my_d_tree = my_d_tree.fit(X_train, y_train)
print(my_d_tree.feature_importances_) 

print(my_d_tree.score(X, y))
pred = my_d_tree.predict(X_test)

pred
pred = my_d_tree.predict(X_test)

df_confusion = metrics.confusion_matrix(y_test, pred)

df_confusion

print(my_d_tree.score(X,y))
from sklearn.ensemble import RandomForestClassifier



# Building and fitting my_forest

forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)

my_forest = forest.fit(X_train, y_train)



# Print the score of the fitted random forest

print(my_forest.score(X, y))

print(my_d_tree.feature_importances_) 
def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):

    plt.matshow(df_confusion, cmap=cmap) # imshow

    plt.title('Confusion Matrix')

    plt.colorbar()

    plt.ylabel('Actual')

    plt.xlabel('Predicted')



plot_confusion_matrix(df_confusion)