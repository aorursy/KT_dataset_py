import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
dataset = pd.read_csv('../input/creditcardfraud/creditcard.csv')
dataset.head().T
# Generating summary table for the data
dataset.describe().T
# Checking missing values
dataset.dtypes
corr = dataset.corr()
fig = plt.gcf()
fig.set_size_inches(12, 9)
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
Label = dataset['Class'].values
dataset.drop(['Class'], axis=1, inplace=True)
from sklearn.model_selection import train_test_split
  
X_train, X_test, y_train, y_test = train_test_split(dataset, Label, test_size = 0.30)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(penalty = 'l1')

# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'solver': ['saga','liblinear'],
              'max_iter': [100, 500]}

grid = GridSearchCV(model, param_grid, refit=True)
grid.fit(X_train, y_train)

print('Best Parameters: ', grid.best_params_)
print('Best Model Score', grid.best_score_)
y_pred = grid.predict(X_test)

# Let's Check Confusion Matrix
confusion_matrix(y_test, y_pred)
from sklearn.metrics import accuracy_score
print('Accuracy Score: ', accuracy_score(y_pred=y_pred, y_true=y_test)*100)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

