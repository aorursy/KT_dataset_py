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
from sklearn import preprocessing
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(style="white") #white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)
from sklearn.metrics import accuracy_score
animal=pd.read_csv('../input/zoo.csv')
ani_class=pd.read_csv('../input/class.csv')
animal.head()
# Check class table for later use.
ani_class
# Check data type for each variable
animal.info()
animal.isnull().sum()
animal.describe()
# Check if class_type has correct values
print(animal.class_type.unique())
print(animal.legs.unique())
# just curious which animal has 5 legs
animal.loc[animal['legs'] == 5]
# Join animal table and class table to show actual class names
df=pd.merge(animal,ani_class,how='left',left_on='class_type',right_on='Class_Number')
df.head()
plt.hist(df.class_type, bins=7)
# See which class the most zoo animals belong to
sns.factorplot('Class_Type', data=df,kind="count", aspect=2)
# heatmap to show correlations
plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("Correlation Heatmap")
corr = animal.corr()
sns.heatmap(corr, annot=True,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
# show vairable correlation which is more than 0.7 (positive or negative)
corr[corr != 1][abs(corr)> 0.7].dropna(how='all', axis=1).dropna(how='all', axis=0)
df.groupby('Class_Type').mean()
# checking leg number in each class
g = sns.FacetGrid(df, col="Class_Type")
g.map(plt.hist, "legs")
plt.show()
from sklearn.model_selection import train_test_split
# 80/20 split
#animal=animal.drop(['eggs', 'hair'], axis=1)
#X = animal.iloc[:,1:15]
#y = animal.iloc[:,15]
X = animal.iloc[:,1:17]
y = animal.iloc[:,17]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
from sklearn.linear_model import Perceptron
ppn = Perceptron(eta0=1, random_state=1)
ppn.fit(X_train, y_train)
# make prediction
y_pred = ppn.predict(X_test)
# check model accuracy
accuracy_score(y_pred,y_test)
# 70/30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
ppn = Perceptron(eta0=1, random_state=1)
ppn.fit(X_train, y_train)
y_pred = ppn.predict(X_test)
accuracy_score(y_pred,y_test)
from sklearn.model_selection import cross_val_score
score_ppn=cross_val_score(ppn, X,y, cv=5)
score_ppn
# The mean score and the 95% confidence interval of the score estimate are:
print("Accuracy: %0.2f (+/- %0.2f)" % (score_ppn.mean(), score_ppn.std() * 2))
from sklearn import tree
dt = tree.DecisionTreeClassifier()
score_dt=cross_val_score(dt, X,y, cv=5)
score_dt
# The mean score and the 95% confidence interval of the score estimate are:
print("Accuracy: %0.2f (+/- %0.2f)" % (score_dt.mean(), score_dt.std() * 2))
from sklearn.svm import SVC
svc = SVC(kernel='linear', C=1)
score_svc=cross_val_score(svc, X,y, cv=5)
score_svc
# The mean score and the 95% confidence interval of the score estimate are:
print("Accuracy: %0.2f (+/- %0.2f)" % (score_svc.mean(), score_svc.std() * 2))
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(multi_class='multinomial', solver='newton-cg')
score_lr=cross_val_score(lr, X,y, cv=5)
score_lr
# The mean score and the 95% confidence interval of the score estimate are:
print("Accuracy: %0.2f (+/- %0.2f)" % (score_lr.mean(), score_lr.std() * 2))
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'Logistic Regression', 'Perceptron', 'Decision Tree'],
    'Score': [score_svc.mean(), score_lr.mean(), score_ppn.mean(), score_dt.mean()]})
models.sort_values(by='Score', ascending=False)