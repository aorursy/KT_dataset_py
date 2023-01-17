import numpy as np

import pandas as pd

import datetime

import matplotlib.pyplot as plt

#import warnings; warnings.simplefilter('ignore')

%env JOBLIB_TEMP_FOLDER=/tmp
x_train = pd.read_csv('../input/train.csv')

y_train = x_train['label'].values

x_train = x_train.drop('label', axis = 1).values

test    = pd.read_csv('../input/test.csv').values

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(x_train)

x_train_scaler = scaler.transform(x_train)

test_scaler = scaler.transform(test)

#Using elbow-plot variance/dimensions

from sklearn.decomposition import PCA

pca = PCA()

pca.fit(x_train_scaler)

cumsum = np.cumsum(pca.explained_variance_ratio_)*100

d = [n for n in range(len(cumsum))]

plt.figure(figsize=(10, 10))

plt.plot(d,cumsum, color = 'red',label='cumulative explained variance')

plt.title('Cumulative Explained Variance as a Function of the Number of Components')

plt.ylabel('Cumulative Explained variance')

plt.xlabel('Principal components')

plt.axhline(y = 95, color='k', linestyle='--', label = '95% Explained Variance')

plt.legend(loc='best')
from sklearn.decomposition import PCA

pca = PCA(.95) 

pca.fit(x_train_scaler)



x_train_pca = pca.transform(x_train_scaler)

test_pca        = pca.transform(test_scaler)



sample = x_train[23]

sample.shape = (28,28)





a = plt.subplot(1,2,1)

a.set_title('Original Image')

plt.imshow(sample, cmap = plt.cm.gray_r)



sample = pca.inverse_transform(x_train_pca[23])

sample.shape = (28,28)



b = plt.subplot(1,2,2)

b.set_title("Reduced after PCA")

plt.imshow(sample, cmap = plt.cm.gray_r)



from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression



# making skeletal model

logistic_regression = LogisticRegression(n_jobs = -1)



# Set of parameters we want to try for out Model

parameters = { 'C' : [1.1,1.25,1.5]}



#Running the Model with above chosen parameter

grid_search = GridSearchCV(estimator = logistic_regression, param_grid = parameters , scoring = 'accuracy', cv = 3, n_jobs = -1 , verbose = 2)

grid_scores = grid_search.fit(x_train_pca , y_train)

print( grid_search.best_score_)

print(grid_search.best_params_)
# Making the Final Classification model.

logistic_regression = LogisticRegression( C = 1.25, n_jobs = -1)

tick =datetime.datetime.now()

logistic_regression.fit(x_train_pca, y_train)

tock=datetime.datetime.now()

lr_train_time = tock - tick

print("Time taken for training a Logistic Regression model = " + str(lr_train_time))

tick=datetime.datetime.now()

lr_train_predict=logistic_regression.predict(x_train_pca)

tock=datetime.datetime.now()

lr_pred_train_time = tock - tick

print('Time taken to predict the data points in the Test set is : ' + str(lr_pred_train_time))
#Making the confusion Matrix



from sklearn.metrics import confusion_matrix

cf= confusion_matrix(y_train, lr_train_predict)



# Visualizing the Confusion Matrix`



plt.matshow(cf , cmap = plt.cm.gray, )

plt.show()



# Analyzing the Errors

row_sums = cf.sum(axis=1 , keepdims = True)

normal_cf = cf/row_sums



np.fill_diagonal(normal_cf,0)

plt.matshow(normal_cf,cmap = plt.cm.gray)



plt.title("error Analysis")

plt.show()



k = logistic_regression.score(x_train_pca, y_train)

print('the Accuracy on the Training set come out to be : ' + str(k))
import sklearn.metrics as skm

print(skm.classification_report( y_train , lr_train_predict ))
predict = logistic_regression.predict(test_pca)

print(predict)


submission = pd.DataFrame({'ImageId': range(1,len(predict)+1),

                           'Label': predict})

print(submission)
submission.to_csv('out.csv', header=True, index = False)   # Generating output csv

print(submission)