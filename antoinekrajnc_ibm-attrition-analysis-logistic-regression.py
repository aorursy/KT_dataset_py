# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/WA_Fn-UseC_-HR-Employee-Attrition.csv")
dataset.head()
dataset.info()
sns.countplot(x=dataset.Attrition, data= dataset, palette='hls')
plt.show()
pd.crosstab(dataset.Department,dataset.Attrition).plot(kind='bar')
plt.title('Attrition par Departement')
plt.xlabel('Department')
plt.ylabel('Fréquence')
plt.show()
table1 = pd.crosstab(dataset.Department, dataset.Attrition)
table1.div(table1.sum(1).astype(float), axis=0).plot(kind='bar', stacked = True)
plt.title('Attrition par Departement')
plt.xlabel('Department')
plt.ylabel('Fréquence')
plt.show()
table2 = pd.crosstab(dataset.JobSatisfaction, dataset.Attrition)
table2.div(table2.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.title("Attrition en fonction de la satisfation au travail")
plt.xlabel("Satisfaction au travail")
plt.ylabel("Proportion d'employés")
plt.show()
table3 = pd.crosstab(dataset.YearsSinceLastPromotion, dataset.Attrition)
table3.div(table3.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.title("Attrition en fonction de la dernière promotion")
plt.xlabel("nombre d'années après la dernière promotion")
plt.ylabel("Proportion d'employés")
plt.show()
table4 = pd.crosstab(dataset.WorkLifeBalance, dataset.Attrition)
table4.div(table4.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.title("Attrition en fonction de l'equilibre Vie Perso / Travail")
plt.xlabel("Equilibre Vie Perso / Travail")
plt.ylabel("Proportion d'employés")
plt.show()
table=pd.crosstab(dataset.EducationField,dataset.Attrition)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title("Attrition en fonction de l'education")
plt.xlabel('Education')
plt.ylabel('Proportion Employé')
plt.show()
X, y = dataset.loc[:, dataset.columns !="Attrition"], dataset.loc[:, "Attrition"]
X = pd.get_dummies(X, drop_first= True)
X.head()
y = pd.get_dummies(y, drop_first= True)
y.head()
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X_new = SelectKBest(chi2, k=40).fit_transform(X,y)
SelectKBest(chi2, k=40).fit(X,y).get_support(indices=True)
y = np.ravel(y)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.25, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C=0.35000000000000003, solver="newton-cg", max_iter=200)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
from sklearn.model_selection import GridSearchCV
grid = {"C": np.arange(0.3,0.4,0.01),
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "max_iter": [200]
       }

classifier_opt = GridSearchCV(classifier, grid, scoring = 'accuracy', cv=10)
classifier_opt.fit(X_train,y_train)
print("Tuned_parameter k : {}".format(classifier_opt.best_params_))
print("Best Score: {}".format(classifier_opt.best_score_))
