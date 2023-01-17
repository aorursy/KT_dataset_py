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

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

data = pd.read_csv("/kaggle/input/iris/Iris.csv")
data.head()
data.shape
data.describe()
data.isnull().sum()
sns.pairplot(data)

plt.show()
sns.heatmap(data.corr())

plt.show()
data.Species.value_counts().plot(kind='bar')

plt.show()
from sklearn.model_selection import train_test_split
X = data.drop(columns=['Id', 'Species'])

y = data.Species
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.svm import SVC

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import validation_curve

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

print("Freq dist of y_train")

u, c = np.unique(y_train, return_counts=True)

print(np.asarray((u, (c/len(y_train))*100  )).T)



print("\n")



print("Freq dist of y_test")

u, c = np.unique(y_test, return_counts=True)

print(np.asarray((u, (c/len(y_test))*100 )).T)

folds = KFold(n_splits = 5, shuffle = True, random_state = 100)
model = SVC()
cv_results = cross_val_score(model, X_train, y_train, cv=folds, scoring='accuracy')
cv_results
# tune the model



# specify the number of folds for k-fold CV

n_folds = KFold(n_splits = 5, shuffle = True, random_state = 102)



# specify range of parameters C & gamma as a list

params = [ {'gamma': [1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]



# create SVC object

linear_model = SVC()



# set up grid search scheme

model_cv = GridSearchCV(estimator = linear_model, 

                        param_grid = params, 

                        scoring= 'accuracy', 

                        cv = n_folds, 

                        verbose = 10,

                        return_train_score=True,

                        n_jobs=-1

                       )      







# fit the model on n_folds

model_cv.fit(X_train, y_train)



cv_results = pd.DataFrame(model_cv.cv_results_)

cv_results
# converting C to numeric type for plotting on x-axis

cv_results['param_C'] = cv_results['param_C'].astype('int')



plt.figure(figsize=(20, 5))



cnt = 1

for gamma in params[0]['gamma']:

    

    plt.subplot(1, len(params[0]['gamma']), cnt)

    gamma_data = cv_results[cv_results['param_gamma']==gamma]

    plt.plot(gamma_data["param_C"], gamma_data["mean_test_score"])

    plt.plot(gamma_data["param_C"], gamma_data["mean_train_score"])

    plt.xlabel('C')

    plt.ylabel('Accuracy')

    plt.title("Gamma="+str(gamma))

    plt.legend(['test accuracy', 'train accuracy'], loc='center right')

    plt.xscale('log')

    cnt=cnt+1



plt.show()

best_score = model_cv.best_score_

best_C = model_cv.best_params_['C']

best_gamma = model_cv.best_params_['gamma']





print(" The highest test accuracy is {0} at C = {1} AND gamma = {2}".format(best_score, best_C, best_gamma))
# model with the best value of C

model = SVC(C=best_C, gamma=best_gamma)



# fit

model.fit(X_train, y_train)



# predict

y_pred = model.predict(X_test)
print("accuracy", metrics.accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))
metrics.confusion_matrix(y_test, y_pred)