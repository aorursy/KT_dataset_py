import pandas as pd

import numpy  as np

import time

import matplotlib.pyplot as plt

fraud = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv', header=0)
fraud.shape
fraud[fraud.isnull().sum(axis=1)>0]
fraud.Class.value_counts()
plt.hist(fraud.Amount, bins=50)

plt.show()
fraud.Amount.describe()
fraud[fraud.Class==0].Amount.describe()
fraud[fraud.Class==1].Amount.describe()
plt.hist(fraud[fraud.Class==1].Amount, bins=30)

plt.show()
import seaborn as sns

corr = fraud.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(1, 100, n=100),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=90,

    horizontalalignment='right'

);
#Lets build a simple logistic regression model first

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve
#Create train and test data from the dataset. Using 30% of the data for validation. Stratifying the data so we have same proportion of the classes in train and test set.

X = fraud.iloc[:,0:30].values

y = fraud.iloc[:,30].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50, stratify=y)
#train a logistic regression model and use it to predict class for the test set

lgr = LogisticRegression(max_iter=1000)

lgr.fit(X_train, y_train)

y_pred = lgr.predict(X_test)

y_prob = lgr.predict_proba(X_test)[:,1]
#Print classification report and 50% threshold

print(classification_report(y_test, y_pred))
# Function to draw ROC plot

def draw_roc_plot(y_test, y_prob):

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    area = roc_auc_score(y_test, y_prob)

    plt.plot([0,1],[0,1],'k--')

    plt.plot(fpr, tpr, label = 'AUC = %0.2f' % area)

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.legend(loc='lower right')

    plt.show()
draw_roc_plot(y_test, y_prob)
from sklearn.model_selection import GridSearchCV
print(time.time())

#Set up the hyperparameter grid

c_space=np.logspace(-5, 8, 10)

#solver = ['lbfgs','saga']

param_grid = {'C': c_space}





#Instantiate LogisticRegression classifier

lgr = LogisticRegression(max_iter=1200)



#Instantiate GridSearchCV

lgr_cv = GridSearchCV(lgr, param_grid = param_grid, cv=5)



#Train Model

lgr_cv.fit(X_train, y_train)



#find best parameters

print(lgr_cv.best_params_)

best_C = lgr_cv.best_params_

print(time.time())
#refitting the model with optimum c value

lgr = LogisticRegression(max_iter=1200, C = best_C.get('C'))

lgr.fit(X_train, y_train)

y_pred = lgr.predict(X_test)

y_prob = lgr.predict_proba(X_test)[:,1]
print(classification_report(y_test, y_pred))
draw_roc_plot(y_test, y_prob)
from sklearn.ensemble import RandomForestClassifier
#Instantiate a RandomForest Classifier

rf = RandomForestClassifier()



print(time.time())



#Fit the model

rf.fit(X_train, y_train)



#Calculate probability of the outcome of both classes

y_prob = rf.predict_proba(X_test)[:,1]

print(time.time())
draw_roc_plot(y_test, y_prob)
print(time.time())

param_grid={'n_estimators':[100,200,500], 'max_depth':[4, 6]}



rf = RandomForestClassifier()

rf_cv = GridSearchCV(rf, param_grid=param_grid, cv=5)

rf_cv.fit(X_train, y_train)

print(time.time())
best = rf_cv.best_params_
rf = RandomForestClassifier(max_depth = best.get('max_depth'), n_estimators=best.get('n_estimators'))

rf.fit(X_train, y_train)

y_prob = rf.predict_proba(X_test)[:,1]

area = roc_auc_score(y_test, y_prob)
draw_roc_plot(y_test, y_prob)
from sklearn.ensemble import GradientBoostingClassifier
#Instantiate GradientBoostingClassifier

gbc = GradientBoostingClassifier()



#Fit the model

gbc.fit(X_train, y_train)



#Calculate outcome probabilities

y_prob = gbc.predict_proba(X_test)[:,1]

roc_auc_score(y_test, y_prob)
#Lets do some parameter tuning

#param_grid = {'loss':['deviance', 'exponential'], 'n_estimators':[100,200,300]}

#gbc = GradientBoostingClassifier()

#gbc_cv = GridSearchCV(gbc, param_grid = param_grid, cv=5)

#gbc_cv.fit(X_train, y_train)
#best = gbc_cv.best_params_
#Kernel kept timing out on me on Kaggle while doing parameter tuning. Putting tuned parameter based on the execution on my machine.

best = {'loss': 'exponential', 'n_estimators': 200}



#lets refit the model using these tuned parameters

gbc = GradientBoostingClassifier(loss=best.get('loss'), n_estimators = best.get('n_estimators'))

gbc.fit(X_train, y_train)

y_prob = gbc.predict_proba(X_test)[:,1]

draw_roc_plot(y_test, y_prob)