## Loading the packages 

import pandas as pd 

import matplotlib.pyplot as plt

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GridSearchCV
## Import and Explore the data 

df = pd.read_csv("../input/breast-cancer.csv")
df = df.drop(['id'], axis=1)
# A list comprehension to code the positive label as 1 and the negative label as 0

df['diagnosis'] = [1 if df['diagnosis'][i] =='M' else 0 for i in np.arange(0,len(df['diagnosis']),1)]
# Columns names list

df.columns
# DataFrame shape

df.shape
# Statistical Descriptive of df DataFrame

df.describe()
# df's variable type

df.dtypes
# Detect missing values

print(df.isnull().sum())
# Convert the 'diagnosis' target variable to categorical data 

df['diagnosis'] = df.diagnosis.astype('category')

assert df['diagnosis'].dtype=='category' ## if the 'diagnosis' variable wasn't converted to categorical, the assert function return an error
# Split the data into train and test

X = df[df.columns[1:]] ## Features

Y = df[df.columns[0]]  ## target

X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)
#-----------------------------------------

# Creat a KNN model and predictions Model1

#-----------------------------------------

model_knn = KNeighborsClassifier(n_neighbors=4)

model_knn.fit(X_train, Y_train)



# predict using the knn model

Y_pred = model_knn.predict(X_test)



# Print the classification metrics

print(confusion_matrix(Y_test, Y_pred))

print(classification_report(Y_test, Y_pred))
# Plot the ROC curve 

Y_pred_prob = model_knn.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_prob)

plt.plot([0,1], [0,1], 'r--')

plt.plot(fpr, tpr, label='KNN ROC curve model1')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('K Nearest Neighbors for knn')

plt.show()



# calculate the AUC and accuracy 

acc_model1 = model_knn.score(X_test, Y_test)

auc_model1 = roc_auc_score(Y_test, Y_pred_prob)

print('model 1: the AUC is equal to :{}'.format(auc_model1))

print('model 1: the ACCURACY is equal to : {}'.format(acc_model1))
#--------------------------------------------------------

# Overfitting and underfitting in function of k : Model 2

#--------------------------------------------------------

neighbor = np.arange(4,15) # range of value of K we test

train_accuracy = np.empty(len(neighbor)) # A list of multiple train's accuracies

test_accuracy = np.empty(len(neighbor)) # A list of multiple test's accuracies

for ind, k in enumerate(neighbor):

    mod_knn = KNeighborsClassifier(n_neighbors=k)

    mod_knn.fit(X_train, Y_train)

    train_accuracy[ind] = mod_knn.score(X_train, Y_train)

    test_accuracy[ind] = mod_knn.score(X_test, Y_test)

    

# Compare the accuracy in function of multiple value of k

plt.figure()

plt.plot(neighbor, train_accuracy, label='training accuracy')

plt.plot(neighbor, test_accuracy, label='testing accuracy')

plt.legend()

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.show()
# Get the best model and it evaluation

k0 = np.argmax(test_accuracy) + 4

b_knn = KNeighborsClassifier(n_neighbors=k0)

b_knn.fit(X_train, Y_train)

# predict using the b_knn model

bY_pred = b_knn.predict(X_test)

# Print the classification metrics

print(confusion_matrix(Y_test, bY_pred))

print(classification_report(Y_test, bY_pred))
# ROC curve for the b_knn

bY_pred_prob = b_knn.predict_proba(X_test)[:,1]

fpr, tpr, threshold = roc_curve(Y_test, bY_pred_prob, pos_label=1) 

plt.plot([0,1], [0,1], 'k--')

plt.plot(fpr, tpr, label = 'KNN-ROC Curve model 2')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('K Nearest Neighbors for b_knn')

plt.show()



# calculate the AUC and accuracy 

acc_model2 = b_knn.score(X_test, Y_test)

auc_model2 = roc_auc_score(Y_test, bY_pred_prob)

print('model 2 : the AUC is equal to: {}'.format(auc_model2))

print('model 2: the ACCURACY is equal to: {}'.format(acc_model2))
# ---------------------------------------

# Tune the parameter K of the KNN Model 3

# ---------------------------------------

param_grid = {'n_neighbors': np.arange(1,30)}

knn = KNeighborsClassifier()

Grid_knn = GridSearchCV(knn, param_grid, scoring='roc_auc', cv=5)

Grid_knn.fit(X_train, Y_train)



# predict using the tunde model

tY_pred = Grid_knn.predict(X_test)



# Print the classification metrics

print(confusion_matrix(Y_test, tY_pred))

print(classification_report(Y_test, tY_pred))
# ROC curve for the b_knn

tY_pred_prob = Grid_knn.predict_proba(X_test)[:,1]

fpr, tpr, threshold = roc_curve(Y_test, tY_pred_prob)

plt.plot([0,1], [0,1], 'k--')

plt.plot(fpr, tpr, label = 'KNN-ROC Curve model 3')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('K Nearest Neighbors for Grid_knn')

plt.show()



# calculate the AUC and accuracy 

acc_model3 = Grid_knn.score(X_test, Y_test)

auc_model3 = roc_auc_score(Y_test, tY_pred_prob)

print('model 3 : the AUC is equal to: {}'.format(auc_model3))

print('model 3: the ACCURACY is equal to: {}'.format(acc_model3))



# Get the best parameters for KNN model:

print(Grid_knn.best_params_)

print(Grid_knn.best_score_)
# -----------------------------------------------------------------------------------

# Let's tune our KNN model using a grid search to find best parameters to model with: model4

# -----------------------------------------------------------------------------------

from sklearn.model_selection import RandomizedSearchCV

params = {'n_neighbors': np.arange(4,30,1),

          'weights':['uniform', 'distance'],

          'algorithm':['ball_tree', 'kd_tree'],

          'metric':['euclidean']}

knn = KNeighborsClassifier()

Rando_knn = RandomizedSearchCV(knn, params, scoring='roc_auc', cv = 5)

Rando_knn.fit(X_train, Y_train)



# Predict using the tuned model

rY_pred = Rando_knn.predict(X_test)



# Print the classification metrics

print(confusion_matrix(Y_test, rY_pred))

print(classification_report(Y_test, rY_pred))
# Plot the ROC curve and accuracy

rY_pred_prob = Rando_knn.predict_proba(X_test)[:,1]

fpr, tpr, threshold = roc_curve(Y_test, rY_pred_prob)

plt.plot([0,1], [0,1], 'b--')

plt.plot(fpr, tpr, label='KNN-ROC Curve model 4')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('K Nearest Neighbors for Rando_knn')

plt.show()
# calculate the AUC and accuracy 

acc_model4 = Rando_knn.score(X_test, Y_test)

auc_model4 = roc_auc_score(Y_test, rY_pred_prob)

print('model 4 : the AUC is equal to: {}'.format(auc_model4))

print('model 4: the ACCURACY is equal to: {}'.format(acc_model4))



# Get the best parameters for KNN model:

print(Rando_knn.best_params_)

print(Rando_knn.best_score_)
# --------------------------------------------------------------------------------------------

# Cross validation for Knn : Relation between variation of accuracy and AUC in Function of value of K

# --------------------------------------------------------------------------------------------

from sklearn.model_selection import cross_val_score

k_range = range(1,30)

k_scores_auc=[]

k_scores_acc=[]

for k in k_range:

    #1.Rune the KNN

    knn = KNeighborsClassifier(n_neighbors=k)

    #2.obtain cross_val_score for KNeighborsClassifier with k neighbours

    scores_auc = cross_val_score(knn, X, Y, cv=5, scoring='roc_auc')

    scores_acc = cross_val_score(knn, X, Y, cv=5, scoring='accuracy')

    k_scores_auc.append(scores_auc.mean())

    k_scores_acc.append(scores_acc.mean())



plt.plot(k_range, k_scores_auc, label = 'AUC')

plt.plot(k_range, k_scores_acc, label = 'ACCURACY')

plt.legend()

plt.xlabel('k for Knn')

plt.ylabel('Evaluation in %')

plt.show()
#---------

## SUM UP

#---------

sumup = pd.DataFrame({'model':[1,2,3,4], 'accuracy':[acc_model1, acc_model2, acc_model3, acc_model4],

                      'AUC':[auc_model1, auc_model2, auc_model3,auc_model4]})

print(sumup)