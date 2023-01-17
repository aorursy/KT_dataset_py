# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import warnings

warnings.filterwarnings("ignore")

warnings.warn("once")
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import validation_curve

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

import seaborn as sns



# Set Pandas options for better display

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

pd.set_option('max_colwidth', 200)
data = pd.read_csv("../input/train.csv")
data.shape
data.describe()
data.columns[data.isnull().any()]
data.info()
samples=data.sample(frac=0.25, random_state=100)
samples.shape
pd.DataFrame(round((samples.label.value_counts() / samples.shape[0]) * 100 , 2)).reset_index().sort_values(by='index')
data.label.value_counts().plot(kind='bar')

plt.show()
X = samples.drop("label", axis = 1)

y = samples.label.values.astype(int)
from sklearn.preprocessing import scale

X = scale(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)
print("Freq dist of y_train")

u, c = np.unique(y_train, return_counts=True)

print(np.asarray((u, (c/len(y_train))*100  )).T)



print("\n")



print("Freq dist of y_test")

u, c = np.unique(y_test, return_counts=True)

print(np.asarray((u, (c/len(y_test))*100 )).T)

# creating a KFold object with 5 splits 

folds = KFold(n_splits = 5, shuffle = True, random_state = 100)
# instantiating a model

model = SVC()

cv_results = cross_val_score(model, X_train, y_train, cv = folds, scoring = 'accuracy') 
# print 5 accuracies obtained from the 5 folds

print(cv_results)

print("mean accuracy = {}".format(cv_results.mean()))
# tune the model



# specify the number of folds for k-fold CV

n_folds = KFold(n_splits = 5, shuffle = True, random_state = 100)



# specify range of parameters C & gamma as a list

params = [ {'gamma': [1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]



# create SVC object

linear_model = SVC()



# set up grid search scheme

model_cv = GridSearchCV(estimator = linear_model, 

                        param_grid = params, 

                        scoring= 'accuracy', 

                        cv = n_folds, 

                        verbose = 100,

                        return_train_score=True,

                        n_jobs=5

                       )      







# fit the model on n_folds

model_cv.fit(X_train, y_train)

# results of grid search CV

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
# accuracy

print("accuracy", metrics.accuracy_score(y_test, y_pred))
random_test_sample=data.sample(frac=0.1, random_state=500)
X_random_test_sample = random_test_sample.drop("label", axis = 1)

y_random_test_sample = random_test_sample.label.values.astype(int)

X_random_test_sample = scale(X_random_test_sample)
y_pred_random_test_sample = model.predict(X_random_test_sample)
print("accuracy", metrics.accuracy_score(y_random_test_sample, y_pred_random_test_sample))
test=pd.read_csv("../input/test.csv")
test.head()
test = scale(test)
results=model.predict(test)
output_df=pd.DataFrame({"ImageId": np.arange(1,test.shape[0]+1), "Label": results})
output_df.to_csv('out_svm.csv', index=False, header=True)