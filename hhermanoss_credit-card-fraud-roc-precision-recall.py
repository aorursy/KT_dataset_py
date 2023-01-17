#Load the csv file as data frame.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import average_precision_score

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

%matplotlib inline



df = pd.read_csv('../input/creditcardfraud/creditcard.csv')

print('Size of creditcardfraud dataframe is :',df.shape)

#Let us see how our data looks like!

df[0:5]
df['Class'].value_counts()
# Before we start pre-processing, let's see whether there are null values

df.isnull().sum()
#Change categorical numbers with meaningful values

df['Class'].replace({0:'Nonfradulent', 1:'Fradulent'},inplace = True)
 #How many record is fradulent? 

df['Class'].value_counts()
#What percentage record is fradulent?

percentageoffradulent=df['Class'].value_counts(normalize=True)*100

percentageoffradulent
# Get back to old 'Class' values for ml sake

df['Class'].replace({'Nonfradulent':0 , 'Fradulent':1},inplace = True)
#Define X and Y in data

X= df[df.columns[:-1]]

y=df[['Class']]
#Spliting data to train and test

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30, random_state=42)
#Hyperparameter tuning using cross-validation and grid search

steps = [('scaler', StandardScaler()), 

        ('logreg', LogisticRegression(penalty = 'l1', solver = 'saga', tol = 1e-6,

                                      max_iter = int(1e6), warm_start = True, n_jobs = -1))]

        

pipeline = Pipeline(steps)

param_grid = {'logreg__C': np.arange(0., 1, 0.1)}

logreg_cv = GridSearchCV(pipeline, param_grid, cv = 5,  n_jobs = -1)

logreg_cv.fit(X_train, y_train) 
#What are the best parameter and best score?

print ('best score:', logreg_cv.best_score_)

print ('best parameter:',logreg_cv.best_params_)
#Fit lasso logistic regression using the best parameter above

scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

logreg = LogisticRegression(penalty = 'l1', solver = 'saga', tol = 1e-6,  max_iter = int(1e6),

                            warm_start = True, C = logreg_cv.best_params_['logreg__C'])

logreg.fit(X_train_scaled, y_train)
#Let's plot the coefficients to see which features have been selected

lasso_coef = logreg.coef_.reshape(-1,1)

plt.figure(figsize = (20,10))

plt.plot([0,29],[0,0])

_ = plt.plot(range(30), lasso_coef, linestyle='--', marker='o', color='r')

_ = plt.xticks(range(30), range(30), rotation=60)

_ = plt.ylabel('Coefficients')

plt.xlabel('Features', fontsize = 16)

plt.ylabel('Coefficients', fontsize = 16)

plt.xticks(size = 18)

plt.yticks(size = 18)

plt.title('Feature Coefficients from Lasso Logistic Regression', fontsize = 28)

plt.show();
#Make predictions using the test dataset

X_test_scaled = scaler.transform(X_test)

y_pred_prob = logreg.predict_proba(X_test_scaled)[:,1] 
#Plot ROC Curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.figure(figsize = (20,10))

plt.plot([0, 1], [0, 1], linestyle = '--')

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate', fontsize = 16)

plt.ylabel('True Positive Rate', fontsize = 16)

plt.xticks(size = 18)

plt.yticks(size = 18)

plt.title('Lasso Logistic Regression ROC Curve', fontsize = 28)

plt.show();
round(roc_auc_score(y_test, y_pred_prob), 2)
#Now, plot the Precision-Recall Curve and calculate the area under the curve

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_prob)

plt.figure(figsize = (20,10))

plt.plot([0, 1], [0.01/0.98, 0.01/0.98], linestyle = '--')

plt.plot(recall, precision)

plt.xlabel('Recall', fontsize = 16)

plt.ylabel('Precision', fontsize = 16)

plt.xticks(size = 18)

plt.yticks(size = 18)

plt.title('Lasso Logistic Regression Precision-Recall Curve', fontsize = 28)

plt.show();
round(average_precision_score(y_test, y_pred_prob), 2)