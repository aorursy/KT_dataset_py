# libraries import 

import numpy as np  

import pandas as pd 

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# reading the data set

data_set = pd.read_csv('../input/social-network-ad/Social_Network_Ads.csv')
# Analysis of the dataset 

data_set.head()
# Plot

import plotly.express as px

fig = px.scatter_3d(data_set, x='Age', y= 'EstimatedSalary',z = 'Gender',

              color='Purchased', symbol='Purchased', opacity=0.7)

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

fig.show()
fig = px.scatter(data_set, x='Age', y= 'EstimatedSalary',

              color='Purchased', symbol='Purchased', opacity=0.7)

fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))

fig.show()
X = data_set.iloc[:,[2,3]].values

Y = data_set.iloc[:,4].values

# Splitting the dataset into train and test

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
# Preprocessing the dataset

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
# Applying the logistic regression classifier 

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=0)

lr.fit(X_train, Y_train)
# Applying the SVM 

from sklearn.svm import SVC

sv = SVC(kernel='rbf', random_state = 0)

sv.fit(X_train, Y_train)
from sklearn.model_selection import cross_val_score

score_lr = cross_val_score(estimator= lr, X= X_train,y=Y_train,cv= 10, n_jobs= -1)

score_sv = cross_val_score(estimator= sv, X= X_train,y=Y_train,cv= 10, n_jobs= -1)





print('SVM : Mean of the accuracies is %2f percent' % (score_sv.mean()*100))

print('SVM : the standard deviation of the accuracies is %2f percent' % (score_sv.std()*100))



print('log_Regression: Mean of the accuracies is %2f percent' % (score_lr.mean()*100))

print('log_Regression: the standard deviation of the accuracies is %2f percent' % (score_lr.std()*100))



plt.plot(range(len(score_sv)), score_sv, label='SVM')

plt.plot(range(len(score_lr)), score_lr, label = 'logistic regression')

plt.xlabel('Fold')

plt.ylabel('Accuracy')

plt.legend()
from sklearn.model_selection import GridSearchCV

# dictionary of parameters used as an input to gridsearch 

parameters = [

    {'C':[1,10,100,1000], 'kernel': ['linear']},

    {'C':[1,10,100,1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}

]

# applying grid search

gscv = GridSearchCV(estimator = sv, 

                   param_grid= parameters,

                   scoring = 'accuracy', cv= 10, n_jobs= -1)

gscv = gscv.fit(X_train, Y_train)



best_acc = gscv.best_score_

print('best accuracy is %2f percent ' %(best_acc*100))

best_parameters = gscv.best_params_

print('best parameters are : ', best_parameters)
predictions_gridsearch = gscv.predict(X_test)

from sklearn.metrics import confusion_matrix

cm_gridsearch = confusion_matrix(Y_test, predictions_gridsearch)

print(cm_gridsearch)
# Visualisation

# Training results 

from matplotlib.colors import ListedColormap

X_set, y_set = X_train, Y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))



plt.contourf(X1, X2, gscv.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('green','yellow')))



plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())



for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = ListedColormap(('green', 'yellow'))(i), label = j)

plt.title('Kernel SVM (Grid Search) -Training set')

plt.xlabel('Age')

plt.ylabel('Estimated Salary')

plt.legend()

plt.show()


X_set, y_set = X_test, Y_test

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, gscv.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('green','yellow')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = ListedColormap(('green','yellow'))(i), label = j)

plt.title('Kernel SVM (Grid Search)-Test set')

plt.xlabel('Age')

plt.ylabel('Estimated Salary')

plt.legend()

plt.show()