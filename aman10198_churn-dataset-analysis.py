# importing necessary libraries:

import matplotlib.pyplot as plt

%matplotlib inline

import pandas as pd

import numpy as np

import os

print(os.listdir("../input"))
from sklearn.metrics import accuracy_score,recall_score, precision_score

from sklearn.model_selection import GridSearchCV
# importing dataset

dataset = pd.read_csv('../input/WA_Fn-UseC_-Telco-Customer-Churn.csv', na_values = [' ','','#NA','NA','NULL','NaN', 'nan', 'n/a'], 

                      dtype = {'TotalCharges':np.float32, 'MonthlyCharges': np.float32} )
print(dataset.shape)

dataset.head(3)
dataset.describe()
# Dropping column not having any significance in predicting the customer decision so we will drop it

dataset.drop(columns = ['customerID', 'PaperlessBilling', 'PaymentMethod']

             , axis = 1, inplace = True)
#checking if any column in the data contain the na_values

dataset.isna().any()
total_rows_with_na_values = sum((dataset['TotalCharges'].isna())*1)

print(total_rows_with_na_values/dataset.shape[0])
dataset['TotalCharges'].fillna(dataset['TotalCharges'].mean(), inplace = True)

print("Is there any na value left in dataset:",dataset['TotalCharges'].isna().any())
X = dataset.iloc[:,:-1].values

y = (dataset.iloc[:,-1].values == 'Yes')*1
# encoding of labels of dataset

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()



for i in range(X.shape[1]):

    # for encoding of all columns having unique values lower than 5

    if len(np.unique(X[:,i])) < 5:

        X[:,i] = label_encoder.fit_transform(X[:,i])
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)
from statsmodels.stats.outliers_influence import variance_inflation_factor    



def calculate_vif_(X, thresh=5.0):

    variables = list(range(X.shape[1]))

    dropped = True

    while dropped:

        dropped = False

        vif = [variance_inflation_factor(X[:, variables], i) for i in range(X[:, variables].shape[1])]

        maxloc = vif.index(max(vif))

        print(max(vif))

        

        if max(vif) > thresh:

            del variables[maxloc]

            dropped = True



    print('Remaining variables:')

    print(variables)

    return X[:, variables]
X = calculate_vif_(X, 5)
import statsmodels.regression.linear_model as sm
def backwardElimination(x,y, SL):

    numVars = len(x[0])

    temp = np.zeros((x.shape[0],numVars)).astype(int)

    

    for i in range(0, numVars):

        regressor_OLS = sm.OLS(y, x).fit()

        

        maxVar = max(regressor_OLS.pvalues).astype(float)

        

        adjR_before = regressor_OLS.rsquared_adj.astype(float)

        

        if maxVar > SL:

            

            for j in range(0, numVars - i):

                

                if (regressor_OLS.pvalues[j].astype(float) == maxVar):

                    temp[:,j] = x[:, j]

                    x = np.delete(x, j, 1)

                    

                    tmp_regressor = sm.OLS(y, x).fit()

                    adjR_after = tmp_regressor.rsquared_adj.astype(float)

                    

                    if (adjR_before >= adjR_after):

                        x_rollback = np.hstack((x, temp[:,[0,j]]))

                        x_rollback = np.delete(x_rollback, j, 1)

                        print (regressor_OLS.summary())

                        return x_rollback

                    else:

                        continue

    regressor_OLS.summary()

    return x

 

X = np.append(arr = np.ones((X.shape[0], 1)).astype(int), values = X, axis = 1)
SL = 0.05

X_opt = X[:, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]

X_Modeled = backwardElimination(X_opt,y, SL)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_Modeled, y, test_size = 0.4, random_state = 32)
import seaborn as sns
# choosing only 1000 tuple from each class for better visualization

churn1 = dataset[dataset['Churn'] == 'Yes'][:1000]

churn0 = dataset[dataset['Churn'] == 'No'][:1000]
plt.figure(figsize = (10,5))

sns.scatterplot('TotalCharges', 'MonthlyCharges',data=churn1)

sns.scatterplot('TotalCharges', 'MonthlyCharges',data=churn0)

plt.show()
sns.scatterplot('TotalCharges', 'tenure',data=churn1)

sns.scatterplot('TotalCharges', 'tenure',data=churn0)

plt.show()
sns.scatterplot('MonthlyCharges', 'tenure',data=churn1)

sns.scatterplot('MonthlyCharges', 'tenure',data=churn0)

plt.show()
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA



print(X_test.shape)

sigma = (1/X_test.shape[0])*np.matmul(X_test.T, X_test)

u,s,v = np.linalg.svd(sigma)
# Reducing the X_test from 13 dimension to 2 dimension

z2 = np.matmul(X_test, u[:,:2])



churn1 = z2[y_test == 1]

churn0 = z2[y_test == 0]



plt.figure(figsize = (8,6))

sns.scatterplot(churn1[:,0], churn1[:,1], color = 'blue', s = 20)

sns.scatterplot(churn0[:,0],churn0[:,1], color = 'red', s = 20)

plt.show()
# Reducing the X_test from 13 dimension to 2 dimension

z = np.matmul(X_test, u[:,:3])



churn1 = z[y_test == 1]

churn0 = z[y_test == 0]





fig = plt.figure()

ax = fig.add_subplot(111, projection='3d', navigate =True)

ax.scatter(churn0[:,0], churn0[:,1], churn0[:,2], c='blue', s=10)

ax.scatter(churn1[:,0], churn1[:,1], churn1[:,2],  c='red', s=10)

ax.view_init(69, 175)

plt.show()
def model_evaluation(classifier, X, y):

    y_pred = classifier.predict(X)

    print("accuracy score:",accuracy_score(y, y_pred))

    print("precision score:",precision_score(y, y_pred))

    print("recall score",recall_score(y, y_pred))
from sklearn.naive_bayes import GaussianNB

naiveBayesClassifier = GaussianNB()

naiveBayesClassifier.fit(X_train,y_train)
print("***Training***")

model_evaluation(naiveBayesClassifier, X_train,y_train)

print("***Testing***")

model_evaluation(naiveBayesClassifier,X_test,y_test)
from sklearn.linear_model import LogisticRegression

logistic_classifier = LogisticRegression(random_state = 32, solver = 'lbfgs')

logistic_classifier.fit(X_train, y_train)
print("***Training***")

model_evaluation(logistic_classifier, X_train,y_train)

print("***Testing***")

model_evaluation(logistic_classifier,X_test,y_test)
from sklearn.neighbors import KNeighborsClassifier

knn_classifier = KNeighborsClassifier(n_neighbors = 9, p = 1, weights = 'uniform', leaf_size = 15, algorithm = 'ball_tree')

knn_classifier.fit(X_train, y_train)
# it will take some time but will return the best perimeter to run on the classifier

# then again run the classifier with the new parameters and fit on training set

parameters = [{'n_neighbors': [5,7,9], 'weights':['uniform','distance'],

              'algorithm':['ball_tree', 'kd_tree', 'brute'],

              'leaf_size':[15,30, 45], 'p':[1,2], }]





grid_search = GridSearchCV(estimator = knn_classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10,

                           n_jobs = -1)



grid_search = grid_search.fit(X_train, y_train)



best_parameters = grid_search.best_params_

print("best score on the parameter on 10 folds is :",grid_search.best_score_)

print(best_parameters)

# We are taking these best parameters and updating the parameters in the classifier above and fit on training set again
print("***Training***")

model_evaluation(knn_classifier, X_train,y_train)

print("***Testing***")

model_evaluation(knn_classifier,X_test,y_test)
from sklearn.svm import SVC

Svm_classifier = SVC(kernel = 'linear', random_state = 32, C = 1)

Svm_classifier.fit(X_train, y_train)
# This block will take some time to find best perimeter of the classifier

parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},

              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]

grid_search = GridSearchCV(estimator = Svm_classifier,

                           param_grid = parameters,

                           scoring = 'accuracy',

                           cv = 10,

                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_



print("best score on the parameter on 10 folds is :",grid_search.best_score_)

print(best_parameters)

# We are taking these best parameters and updating the parameters in the classifier above and fit on training set again
print("***Training***")

model_evaluation(Svm_classifier, X_train,y_train)

print("***Testing***")

model_evaluation(Svm_classifier,X_test,y_test)
from sklearn.tree import DecisionTreeClassifier

dt_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

dt_classifier.fit(X_train, y_train)
print("***Training***")

model_evaluation(dt_classifier, X_train,y_train)

print("***Testing***")

model_evaluation(dt_classifier,X_test,y_test)
# Fitting Random Forest Classification to the Training set

from sklearn.ensemble import RandomForestClassifier

rdt_classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)

rdt_classifier.fit(X_train, y_train)

print("***Training***")

model_evaluation(rdt_classifier, X_train,y_train)

print("***Testing***")

model_evaluation(rdt_classifier,X_test,y_test)
import keras

from keras.models import Sequential

from keras.layers import Dense
# Initialising the ANN

classifier = Sequential()



classifier.add(Dense(output_dim = 64, init = 'glorot_uniform', activation = 'relu', input_dim = 13))



classifier.add(Dense(output_dim = 128, init = 'glorot_uniform', activation = 'relu'))



classifier.add(Dense(output_dim = 64, init = 'glorot_uniform', activation = 'relu'))



classifier.add(Dense(output_dim = 32, init = 'glorot_uniform', activation = 'relu'))



classifier.add(Dense(output_dim = 1, init = 'glorot_uniform', activation = 'sigmoid'))



classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size = 32, epochs = 100)
print("***Training***")

y_train_pred = classifier.predict_classes(X_train)

print("accuracy: ", accuracy_score(y_train,y_train_pred))

print("precision: ", precision_score(y_train,y_train_pred))

print("recall: ", recall_score(y_train,y_train_pred))



print("***Testing***")

y_test_pred = classifier.predict_classes(X_test)

print("accuracy: ", accuracy_score(y_test,y_test_pred))

print("precision: ", precision_score(y_test,y_test_pred))

print("recall: ", recall_score(y_test,y_test_pred))