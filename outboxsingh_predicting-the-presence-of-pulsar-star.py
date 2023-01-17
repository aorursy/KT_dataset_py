#Importing the python modules



import numpy as np

import pandas as pd

import sklearn.metrics as metrics

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import cross_val_score

from sklearn.metrics import precision_score,recall_score,f1_score,roc_auc_score,roc_curve

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split,GridSearchCV

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
#Importing the data files



file = pd.read_csv('../input/pulsar_stars.csv')
#Extracting the target variable



y=file.target_class

X=file[file.columns[:8]]

X.shape
#Plotting the classes



pd.value_counts(y).plot.bar()

plt.title('Data on star detection')

plt.xlabel('Class')

plt.ylabel('Frequency')

y.value_counts()
#Scaling the dataset



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X.astype(np.float64))
#Splitting the dataset into train and test datasets



X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)



print("Number transactions X_train dataset: ", X_train.shape)

print("Number transactions y_train dataset: ", y_train.shape)

print("Number transactions X_test dataset: ", X_test.shape)

print("Number transactions y_test dataset: ", y_test.shape)
#As there is class imbalance in the dataset we use SMOTE to synthetically oversample from the minority class
print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))

print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))



sm = SMOTE(random_state=2)

X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())



print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))

print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))



print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))

print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))
#Using Grid Search to tune the parameters of the Random Forest



rnd_clf=RandomForestClassifier(random_state=100)

param_grid = { 

    'n_estimators': [100,150],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [5,6,7,8],

    'criterion' :['gini', 'entropy']

}

CV_rfc = GridSearchCV(estimator=rnd_clf, param_grid=param_grid, cv= 5)

rnd_cv_fit=CV_rfc.fit(X_train_res,y_train_res)
#Searching the best parameters of the Random Forest Classifier



CV_rfc.best_params_
#Fitting the tuned Random Forest Classifier on the train data



rnd=RandomForestClassifier(random_state=100,n_estimators=150,criterion="gini",max_depth=8,max_features="log2")

rnd_fit=rnd_clf.fit(X_train_res,y_train_res)

y_test_fit=rnd_fit.predict(X_test)
#Printing the accuracy,precision,recall and f1 score on the test dataset



print("Cross-Validated Accuracy on 3 cv sets:",cross_val_score(rnd,X_test,y_test,cv=3,scoring="accuracy"))

print("Precision Score:",precision_score(y_test,y_test_fit))

print("Recall Score:",recall_score(y_test,y_test_fit))

print("F1-score:",f1_score(y_test,y_test_fit))
#Plotting the ROC AUC



roc_curve(y_test,y_test_fit)

fpr, tpr, threshold = roc_curve(y_test, y_test_fit)

roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()