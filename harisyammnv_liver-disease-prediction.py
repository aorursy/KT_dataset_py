import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data=pd.read_csv("../input/indian_liver_patient.csv")

data.head()
data.tail()
data.info() # should help us to locate if there are any missing or null values
# checking the stats

# given in the website 416 liver disease patients and 167 non liver disease patients

# need to remap the classes liver disease:=1 and no liver disease:=0 (normal convention to be followed)

count_classes = pd.value_counts(data['Dataset'], sort = True).sort_index()

count_classes.plot(kind = 'bar')

plt.title("Liver disease classes histogram")

plt.xlabel("Dataset")

plt.ylabel("Frequency")

data['Dataset'] = data['Dataset'].map({2:0,1:1}) 
data['Dataset'].value_counts()
data['Albumin_and_Globulin_Ratio'].fillna(value=0, inplace=True)
data_features=data.drop(['Dataset'],axis=1)

data_num_features=data.drop(['Gender','Dataset'],axis=1)

data_num_features.head()
data_num_features.describe() # check to whether feature scaling has to be performed or not 
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

cols=list(data_num_features.columns)

data_features_scaled=pd.DataFrame(data=data_features)

data_features_scaled[cols]=scaler.fit_transform(data_features[cols])

data_features_scaled.head()
data_exp=pd.get_dummies(data_features_scaled)

data_exp.head()
# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(12, 10))

plt.title('Pearson Correlation of liver disease Features')

# Draw the heatmap using seaborn

sns.heatmap(data_num_features.astype(float).corr(),linewidths=0.25,vmax=1.0, square=True, cmap="YlGnBu", linecolor='black',annot=True)
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report
import itertools



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        #print("Normalized confusion matrix")

    else:

        1

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
X=data_exp

y=data['Dataset'] 

X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.3,random_state=0)

len(Y_train[Y_train==0])/len(Y_train[Y_train==1])
len(Y_test[Y_test==0])/len(Y_test[Y_test==1])
clf=SVC(random_state=0,kernel='rbf')

clf.fit(X_train,Y_train)

predictions=clf.predict(X_test)


# Compute confusion matrix

cnf_matrix = confusion_matrix(Y_test,predictions)

np.set_printoptions(precision=2)



print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))



# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix

                      , classes=class_names

                      , title='Confusion matrix')

plt.show()
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, predictions)

roc_auc = auc(false_positive_rate, true_positive_rate)

print (roc_auc)
plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
# Import 'GridSearchCV', 'make_scorer', and any other necessary libraries

from sklearn import grid_search

from sklearn.metrics import make_scorer, fbeta_score,accuracy_score

#from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

# Initialize the classifier

clf = SVC(random_state=0,kernel='rbf')



#  Create the parameters list you wish to tune, using a dictionary if needed.

#  parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}

parameters = {'C': [10,50,100,200],'kernel':['poly','rbf','linear','sigmoid']}



# Make an fbeta_score scoring object using make_scorer()

scorer = make_scorer(fbeta_score,beta=0.5)



# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()

grid_obj = grid_search.GridSearchCV(clf,parameters,scoring=scorer,n_jobs=-1)



# Fit the grid search object to the training data and find the optimal parameters using fit()

grid_fit = grid_obj.fit(X_train,Y_train)



# Get the estimator

best_clf = grid_fit.best_estimator_



# Make predictions using the unoptimized and model

predictions = (clf.fit(X_train,Y_train)).predict(X_test)

best_predictions = best_clf.predict(X_test)
# Report the before-and-afterscores

print ("Unoptimized model\n------")

print ("Accuracy score on testing data: {:.4f}".format(accuracy_score(Y_test, predictions)))

print ("F-score on testing data: {:.4f}".format(fbeta_score(Y_test, predictions, beta = 2)))

print ("\nOptimized Model\n------")

print ("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(Y_test, best_predictions)))

print ("Final F-score on the testing data: {:.4f}".format(fbeta_score(Y_test, best_predictions, beta = 2)))

print (best_clf)
# Compute confusion matrix

cnf_matrix = confusion_matrix(Y_test,best_predictions)

np.set_printoptions(precision=2)



print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))



# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix

                      , classes=class_names

                      , title='Confusion matrix')

plt.show()
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, best_predictions)

roc_auc = auc(false_positive_rate, true_positive_rate)

print (roc_auc)
plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
from imblearn.over_sampling import SMOTE

oversampler=SMOTE(random_state=0)

os_features,os_labels=oversampler.fit_sample(X_train,Y_train)
len(os_labels[os_labels==1])/len(os_labels[os_labels==0])
clf=SVC(random_state=0,kernel='rbf') # unoptimized Model

clf.fit(os_features,os_labels)
# perform predictions on test set

predictions=clf.predict(X_test)
# Compute confusion matrix

cnf_matrix = confusion_matrix(Y_test,predictions)

np.set_printoptions(precision=2)



print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))



# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix

                      , classes=class_names

                      , title='Confusion matrix')

plt.show()
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, predictions)

roc_auc = auc(false_positive_rate, true_positive_rate)

print (roc_auc)
plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')


#Import 'GridSearchCV', 'make_scorer', and any other necessary libraries

from sklearn import grid_search

from sklearn.metrics import make_scorer, fbeta_score,accuracy_score

#from sklearn.ensemble import RandomForestClassifier

# TODO: Initialize the classifier

clf = SVC(random_state=0,kernel='rbf')



#  Create the parameters list you wish to tune, using a dictionary if needed.

#  parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}

parameters = {'C': [10,50,100,200],'kernel':['poly','rbf','linear','sigmoid']}



# Make an fbeta_score scoring object using make_scorer()

scorer = make_scorer(fbeta_score,beta=2)



# Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()

grid_obj = grid_search.GridSearchCV(clf,parameters,scoring=scorer,n_jobs=-1)



#  Fit the grid search object to the training data and find the optimal parameters using fit()

grid_fit = grid_obj.fit(os_features,os_labels)



# Get the estimator

best_clf = grid_fit.best_estimator_



# Make predictions using the unoptimized and model

predictions = (clf.fit(os_features,os_labels)).predict(X_test)

best_predictions = best_clf.predict(X_test)
# Report the before-and-afterscores

print ("Unoptimized model\n------")

print ("Accuracy score on testing data: {:.4f}".format(accuracy_score(Y_test, predictions)))

print ("F-score on testing data: {:.4f}".format(fbeta_score(Y_test, predictions, beta = 2)))

print ("\nOptimized Model\n------")

print ("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(Y_test, best_predictions)))

print ("Final F-score on the testing data: {:.4f}".format(fbeta_score(Y_test, best_predictions, beta = 2)))

print (best_clf)
# Compute confusion matrix

cnf_matrix = confusion_matrix(Y_test,best_predictions)

np.set_printoptions(precision=2)



print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))



# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix

                      , classes=class_names

                      , title='Confusion matrix')

plt.show()
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, best_predictions)

roc_auc = auc(false_positive_rate, true_positive_rate)

print (roc_auc)
plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(random_state=0) # unoptimized Model

clf.fit(os_features,os_labels)
# perform predictions on test set

predictions=clf.predict(X_test)


# Compute confusion matrix

cnf_matrix = confusion_matrix(Y_test,predictions)

np.set_printoptions(precision=2)



print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))



# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix

                      , classes=class_names

                      , title='Confusion matrix')

plt.show()
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, predictions)

roc_auc = auc(false_positive_rate, true_positive_rate)

print (roc_auc)
plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')
# TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries

from sklearn import grid_search

from sklearn.metrics import make_scorer, fbeta_score,accuracy_score

from sklearn.ensemble import RandomForestClassifier

# TODO: Initialize the classifier

clf = RandomForestClassifier(random_state=0)



# TODO: Create the parameters list you wish to tune, using a dictionary if needed.

# HINT: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}

parameters = {'n_estimators': [100,250,500], 'max_depth': [3,6,9]}



# TODO: Make an fbeta_score scoring object using make_scorer()

scorer = make_scorer(fbeta_score,beta=2)



# TODO: Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()

grid_obj = grid_search.GridSearchCV(clf,parameters,scoring=scorer,n_jobs=-1)



# TODO: Fit the grid search object to the training data and find the optimal parameters using fit()

grid_fit = grid_obj.fit(os_features,os_labels)



# Get the estimator

best_clf = grid_fit.best_estimator_



# Make predictions using the unoptimized and model

predictions = (clf.fit(os_features,os_labels)).predict(X_test)

best_predictions = best_clf.predict(X_test)
# Report the before-and-afterscores

print ("Unoptimized model\n------")

print ("Accuracy score on testing data: {:.4f}".format(accuracy_score(Y_test, predictions)))

print ("F-score on testing data: {:.4f}".format(fbeta_score(Y_test, predictions, beta = 2)))

print ("\nOptimized Model\n------")

print ("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(Y_test, best_predictions)))

print ("Final F-score on the testing data: {:.4f}".format(fbeta_score(Y_test, best_predictions, beta = 2)))

print (best_clf)


# Compute confusion matrix

cnf_matrix = confusion_matrix(Y_test,best_predictions)

np.set_printoptions(precision=2)



print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))



# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix

                      , classes=class_names

                      , title='Confusion matrix')

plt.show()
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, best_predictions)

roc_auc = auc(false_positive_rate, true_positive_rate)

print (roc_auc)
plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')