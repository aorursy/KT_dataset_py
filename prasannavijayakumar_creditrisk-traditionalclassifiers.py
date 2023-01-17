# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#import necessary modules

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



sns.set()
credit= pd.read_csv('/kaggle/input/credit-risk/original.csv')

credit.head()        
print(credit.shape)
credit.info()
credit.describe(include='all')
#lets replace this negative values with nan values

credit.loc[~(credit['age'] > 0), 'age']=np.nan
unique_vals= {

    k: credit[k].unique()

    for k in credit.columns

    

}



unique_vals
#drop clientid from dataset

credit= credit.drop('clientid', axis=1)
credit.isnull().sum()
# 6 missing values in 2000 records is roughly 1.2% of total records. we will drop null values

credit= credit.dropna()

credit.shape
credit['default']= credit['default'].astype('category')

credit['age']=credit['age'].astype('int')
credit.describe()
credit.corr()
credit.var()
credit['default'].value_counts()
credit.groupby('default').mean()
credit.groupby('age').mean()
plt.figure(figsize=(20,10))

credit.hist()

plt.show()
fig, (ax1, ax2, ax3)= plt.subplots(1,3)

credit['age'].plot(kind='box', ax=ax1, figsize=(12,6))

credit['income'].plot(kind='box', ax=ax2, figsize=(12,6))

credit['loan'].plot(kind='box', ax=ax3, figsize=(12,6))

plt.show()
sns.barplot(y='age', x='default', data=credit)

plt.xlabel('Defaults')

plt.ylabel('age of defaulters')

plt.title('Average age of defaulters on Loan', fontsize=12)

plt.show()
sns.pairplot(data=credit, hue='default',diag_kind='kde')
# Find the mean and standard dev

std = credit['loan'].std()

mean = credit['loan'].mean()

# Calculate the cutoff

cut_off = std * 3

lower, upper = mean - cut_off, mean + cut_off

# Trim the outliers

trimmed_df = credit[(credit['loan'] < upper) \

                           & (credit['loan'] > lower)]

trimmed_df.shape
# The trimmed box plot

trimmed_df[['loan']].boxplot()

plt.show()
#Split the independent and outcome variable

X= credit.iloc[:,0:3]

y=credit.iloc[:,3]

y.value_counts()
#lets split to training and test set for training the model and validating the model

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test= train_test_split(X, y, random_state=9999, stratify=y)

#stratify is used since the target class distribution is imbalanced



X_train.shape, X_test.shape, y_train.shape, y_test.shape
#lets perform scaling. all our features are numerical columns

#it is important that we need to have our features to be in same scale.



from sklearn.preprocessing import StandardScaler



sc= StandardScaler()

X_train= sc.fit_transform(X_train)

X_test=sc.fit_transform(X_test)
from sklearn.linear_model import LogisticRegression



#instantiate LogisticRegression model

logreg= LogisticRegression(solver='lbfgs')
#perform cross validation to ensure the model is good model

from sklearn.model_selection import cross_val_score



cv_scores= cross_val_score(logreg, X, y, cv=5)



# Print the 5-fold cross-validation scores

print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
#Fit the linear regression model to training data

logreg.fit(X_train, y_train)



# Predict the test set

y_pred = logreg.predict(X_test)

y_pred
# Making the confusion matrix

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



cm = confusion_matrix(y_test,y_pred)

acc_score = accuracy_score(y_test, y_pred)



print(f"Accuracy = {acc_score*100:.2f}%")

print(f"Confusion matrix = \n{cm}")
#Check Training and Test Set Accuracy



training_accuracy= logreg.score(X_train, y_train)

test_accuracy= logreg.score(X_test, y_test)



print(f"Training Set accuracy = {training_accuracy*100:.2f}%")

print(f"Test Set accuracy = {test_accuracy*100:.2f}%")
# Complete classification report

print(classification_report(y_test,y_pred))
# Coefficients of the model and its intercept

print(dict(zip(X.columns, abs(logreg.coef_[0]).round(2))))

print(logreg.intercept_)
from sklearn.feature_selection import RFE

from sklearn.metrics import accuracy_score



# Create the RFE with a LogisticRegression estimator and 2 features to select

rfe = RFE(estimator=logreg, n_features_to_select=2, verbose=1)

# Fits the eliminator to the data

rfe.fit(X_train, y_train)

# Print the features and their ranking (high = dropped early on)

print(dict(zip(X.columns, rfe.ranking_)))

# Print the features that are not eliminated

print(X.columns[rfe.support_])

# Calculates the test set accuracy

acc = accuracy_score(y_test, rfe.predict(X_test))

print("{0:.1%} accuracy on test set.".format(acc))
from sklearn.metrics import roc_curve, auc



#compute predicted probabilities: y_pred_prob

y_pred_prob= logreg.predict_proba(X_test)[:,1]



#Generate ROC curve values: fpr, tpr, thresholds

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)



# Calculate the AUC



roc_auc = auc(fpr, tpr)

print ('ROC AUC: %0.3f' % roc_auc )



#Plot ROC curve

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.legend(loc="lower right")

plt.show()
#instantiate RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier



rfc= RandomForestClassifier(n_estimators=10, max_depth=3)



#Fit the RandomForest model to training data

rfc.fit(X_train, y_train)



# Predict the test set

y_pred_rfc = rfc.predict(X_test)

y_pred_rfc
# Making the confusion matrix

cm_rfc = confusion_matrix(y_test,y_pred_rfc)

acc_score_rfc = accuracy_score(y_test, y_pred_rfc)



print(f"Accuracy = {acc_score_rfc*100:.2f}%")

print(f"Confusion matrix = \n{cm_rfc}")
#Check Training and Test Set Accuracy



training_accuracy_rfc= rfc.score(X_train, y_train)

test_accuracy_rfc= rfc.score(X_test, y_test)



print(f"Training Set accuracy = {training_accuracy_rfc*100:.2f}%")

print(f"Test Set accuracy = {test_accuracy_rfc*100:.2f}%")
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.metrics import make_scorer





#lets get hyperparameters defined in our model

rfc.get_params()
param_grid= {"max_depth": [2, 4, 6, 8, 10],

            "max_leaf_nodes": [2, 4, 6],

            "min_samples_split":[2, 4, 6, 8],

            "n_estimators": [10, 50, 100, 150]}



#create scoring parameter as accuracy_score. There are some default scoring methods defined. however if we want to create we can create using make_Scorer

#Here i am using Accuracy score as scorring method. we can also use recall_score etc

scorer= make_scorer(accuracy_score)
rcv =RandomizedSearchCV(estimator=rfc,param_distributions=param_grid,n_iter=10,cv=5,scoring=scorer)

rcv.fit(X, y)



# print the mean test scores:

print('The accuracy for each run was: {}.'.format(rcv.cv_results_['mean_test_score']))

# print the best model score:

print('The best accuracy for a single model was: {}'.format(rcv.best_params_))
#Use the best params and reinstantiate RandomForestClassifier model

model=RandomForestClassifier(n_estimators= 50, min_samples_split= 2, max_leaf_nodes= 6, max_depth= 10)



#fit the training set to model

model.fit(X_train, y_train)



# Making the confusion matrix

cm_rfc2 = confusion_matrix(y_test,model.predict(X_test))

acc_score_rfc2 = accuracy_score(y_test, model.predict(X_test))



print(f"Accuracy = {acc_score_rfc2*100:.2f}%")

print(f"Confusion matrix = \n{cm_rfc2}")
#Check Training and Test Set Accuracy



training_accuracy_rfc2= model.score(X_train, y_train)

test_accuracy_rfc2= model.score(X_test, y_test)



print(f"Training Set accuracy = {training_accuracy_rfc2*100:.2f}%")

print(f"Test Set accuracy = {test_accuracy_rfc2*100:.2f}%")
from sklearn.metrics import roc_curve, auc



#compute predicted probabilities: y_pred_prob

y_pred_prob_rfc= model.predict_proba(X_test)[:,1]



#Generate ROC curve values: fpr, tpr, thresholds

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_rfc)



# Calculate the AUC



roc_auc = auc(fpr, tpr)

print ('ROC AUC: %0.3f' % roc_auc )



#Plot ROC curve

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve of RandomForest Model')

plt.legend(loc="lower right")

plt.show()
#Instantiate Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier



dt= DecisionTreeClassifier(max_depth=4, random_state=9999)



dt.fit(X_train, y_train)



#fit the training set to model

dt.fit(X_train, y_train)
# Making the confusion matrix

cm_dt = confusion_matrix(y_test,dt.predict(X_test))

acc_score_dt = accuracy_score(y_test, dt.predict(X_test))



print(f"Accuracy = {acc_score_dt*100:.2f}%")

print(f"Confusion matrix = \n{cm_dt}")
#Check Training and Test Set Accuracy



training_accuracy_dt= dt.score(X_train, y_train)

test_accuracy_dt= dt.score(X_test, y_test)



print(f"Training Set accuracy = {training_accuracy_dt*100:.2f}%")

print(f"Test Set accuracy = {test_accuracy_dt*100:.2f}%")
# Complete classification report

print(classification_report(y_test,dt.predict(X_test)))
#compute predicted probabilities: y_pred_prob

y_pred_prob_dt= dt.predict_proba(X_test)[:,1]



#Generate ROC curve values: fpr, tpr, thresholds

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_dt)



# Calculate the AUC



roc_auc = auc(fpr, tpr)

print ('ROC AUC: %0.3f' % roc_auc )



#Plot ROC curve

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve of Decision Tree Model')

plt.legend(loc="lower right")

plt.show()
#lets draw decision tree



from sklearn import tree



decision_tree= tree.export_graphviz(dt, out_file='tree.dot', feature_names=credit.iloc[:, :3].columns, 

                                    max_depth=4, filled=True, rounded=True)
!dot -Tpng tree.dot -o tree.png
image= plt.imread('tree.png')

plt.figure(figsize=(20, 20))

plt.imshow(image)