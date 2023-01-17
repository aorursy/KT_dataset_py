import pandas as pd

cancer = pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")

cancer.columns
# drop some unnecessary columns

cancer = cancer.drop(['id', 'Unnamed: 32'], axis=1)

cancer.head()
cancer.info()
cancer.describe()
cancer['diagnosis'].value_counts()
import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np



#correlation of means

corr1 = cancer[cancer.columns[1:]].corr()

mask = np.triu(np.ones_like(corr1, dtype=bool))

plt.rcParams["figure.figsize"] = (50,50)

sns.set(font_scale=4)

cfig = sns.heatmap(corr1, annot=True,annot_kws={'size': 30}, mask = mask, 

                   fmt= '.2f', cmap='coolwarm', cbar=False)
# standard and split train and test sets



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



#drop = ['diagnosis','perimeter_mean', 'area_mean', 'radius_worst', 'perimeter_worst', 'area_worst']

data = cancer.drop('diagnosis', axis=1)

data = StandardScaler().fit_transform(data)

target = cancer['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=50)
from sklearn.decomposition import PCA



pca = PCA()

pca.fit(X_train)



# keep 95% of the variance



cumsum = np.cumsum(pca.explained_variance_ratio_)

print(cumsum)

d = np.argmax(cumsum >= 0.95) + 1

print('The dimension should be reduced to ' + str(d) + '.')

features = pca.n_components_

plt.bar(range(features), pca.explained_variance_)
from sklearn.linear_model import LogisticRegression

from sklearn.decomposition import KernelPCA

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV



pipe = Pipeline([

    ('kpca', KernelPCA(n_components = 10)),

    ('logreg', LogisticRegression(solver='lbfgs'))

])



# check if it is overfit

#pipe.fit(X_train, y_train)

#print(pipeline.score(X_train, y_train))

#print(pipeline.score(X_test, y_test))



param_grid = [{

    "kpca__gamma": np.linspace(0.01, 0.05, 10),

    "kpca__kernel": ["rbf", "sigmoid"],

    "logreg__C": np.linspace(0.01, 10, 20)

}]



grid_search = GridSearchCV(pipe, param_grid, cv = 3)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

print(grid_search.best_score_)
# find the best scores further

param_grid = [{

    "kpca__gamma": np.linspace(0.02, 0.05, 10),

    "kpca__kernel": ["sigmoid"],

    "logreg__C": np.linspace(8, 10, 10)

}]



grid_search = GridSearchCV(pipe, param_grid, cv = 3)

grid_search.fit(X_train, y_train)

print(grid_search.best_params_)

print(grid_search.best_score_)
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.model_selection import cross_val_score



pipe = Pipeline([

    ("kpca", KernelPCA(n_components=10, gamma=0.03, kernel="sigmoid")),

    ('logreg', LogisticRegression(solver='lbfgs', C=10))

])



pipe.fit(X_train, y_train)

print("The cross validation score is " + str(cross_val_score(pipe, X_train, y_train, cv=5).mean()) + '\n')



plt.rcParams["figure.figsize"] = (5,5)

print("For train set:")

y_pred = pipe.predict(X_train)

conf_train = confusion_matrix(y_train, y_pred)

sns.heatmap(conf_train, annot=True, cbar=False, annot_kws={'size': 20}, fmt= '.2f')

print("Precision score is " + str(precision_score(y_train, y_pred, pos_label = 'B')))

print("Recall score is " + str(recall_score(y_train, y_pred, pos_label = 'B')))

print("F1 score is " + str(f1_score(y_train, y_pred, pos_label = 'B')))

y_score = pipe.decision_function(X_train)

print("ROC AUC of is ", roc_auc_score(y_train, y_score))
print("For test set:")

y_pred = pipe.predict(X_test)

conf_test = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_test, annot=True, cbar=False, annot_kws={'size': 20}, fmt= '.2f')

print("Precision score is " + str(precision_score(y_test, y_pred, pos_label = 'B')))

print("Recall score is " + str(recall_score(y_test, y_pred, pos_label = 'B')))

print("F1 score is " + str(f1_score(y_test, y_pred, pos_label = 'B')))

y_score = pipe.decision_function(X_test)

print("ROC AUC of is ", roc_auc_score(y_test, y_score))