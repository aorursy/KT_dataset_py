# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

from sklearn import datasets

from sklearn.decomposition import PCA



from pandas.plotting import scatter_matrix



import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn import neighbors, datasets



from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split



# Import necessary modules

from scipy.stats import randint

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV



from sklearn.metrics import roc_curve

from sklearn.linear_model import LogisticRegression





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


df_diabetes = pd.read_csv(os.path.join(dirname, filename))
df_diabetes.head()
df_diabetes.info()


_ = scatter_matrix(df_diabetes, alpha=0.2, figsize  = [15, 15],

    marker   = ".")

plt.show()
histogram_intersection = lambda a, b: 1/(np.minimum(a, b).sum().round(decimals=1))



df_diabetes.corr(method=histogram_intersection)


df_diabetes =  df_diabetes.loc[df_diabetes['Glucose']!=0]

df_diabetes.plot.scatter(x='Outcome', y='Glucose')

plt.show()

MedianGlucose0 = np.median(df_diabetes.loc[df_diabetes.Outcome==0].Glucose.values)

print('MedianGLucose for non diabetic')

print(str(MedianGlucose0))

MedianGlucose1 = np.median(df_diabetes.loc[df_diabetes.Outcome==1].Glucose.values)

print('MedianGLucose for diabetic')

print(str(MedianGlucose1))
# # Create arrays for the features and the response variable

y = df_diabetes['Outcome'].values

X = df_diabetes.drop('Outcome',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



# Setup the hyperparameter grid

c_space = np.logspace(-5, 8, 15)

param_grid = {'C': c_space}





log_reg = LogisticRegression(random_state=42, solver='lbfgs', max_iter=10000)

# Instantiate the GridSearchCV object: logreg_cv

log_reg_cv = GridSearchCV(log_reg, param_grid, cv=5)



# Fit it to the data



log_reg_cv.fit(X_train, y_train)



# Print the tuned parameters and score

print("Tuned Logistic Regression Parameters: {}".format(log_reg_cv.best_params_)) 

print("Best score is {}".format(log_reg_cv.best_score_))

log_reg_cv.score(X_test, y_test)
y_pred = log_reg_cv.predict(X_test)
# Generate the confusion matrix and classification report

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
from sklearn.linear_model import LogisticRegression



from yellowbrick.classifier import DiscriminationThreshold

from yellowbrick.datasets import load_spam





# Instantiate the classification model and visualizer

model = LogisticRegression(multi_class="auto", solver="liblinear")

visualizer = DiscriminationThreshold(model)



visualizer.fit(X, y)        # Fit the data to the visualizer

#      visualizer.show()      # Finalize and render the figure
y_pred_prob = log_reg_cv.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0,1],[0,1], 'k--', label = 'Random choice..')

plt.plot(fpr, tpr, label = 'Logistic Regression')

plt.xlabel('False positive rate')

plt.legend()

plt.ylabel('True positive rate')

plt.title(' Logistic Regression ROC curve')

plt.show()
from sklearn.model_selection import cross_val_score



cv_score = cross_val_score(log_reg,X, y, cv=20, scoring='roc_auc')
np.mean(cv_score)
np.std(cv_score)
np.std(cv_score)
log_reg = LogisticRegression(C=30,random_state=42, solver='lbfgs', max_iter=10000)
log_reg.fit(X_train, y_train)
pred_log_reg = log_reg.predict(X_test)

y_log_reg_proba = log_reg.predict_proba(X_test)

log_reg.score(X_test,y_test)
log_reg.score(X,y)
cv_score = cross_val_score(log_reg,X, y, cv=10, scoring='roc_auc')
cv_score
np.mean(cv_score)