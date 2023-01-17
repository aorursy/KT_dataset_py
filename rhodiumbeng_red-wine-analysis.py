import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# For this kernel, I amm only using the red wine dataset
data = pd.read_csv('../input/winequality-red.csv')
data.head()
data.describe()
extra = data[data.duplicated()]
extra.shape
# Let's proceed to separate 'quality' as the target variable and the rest as features.
y = data.quality                  # set 'quality' as target
X = data.drop('quality', axis=1)  # rest are features
print(y.shape, X.shape)           # check correctness
# data.hist(figsize=(10,10))
sns.set()
data.hist(figsize=(10,10), color='red')
plt.show()
colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Correlation of Features', y=1.05, size=15)
sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, 
            linecolor='white', annot=True)
# Create a new y1
y1 = (y > 5).astype(int)
y1.head()
# plot histogram
ax = y1.plot.hist(color='green')
ax.set_title('Wine quality distribution', fontsize=14)
ax.set_xlabel('aggregated target value')
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import confusion_matrix
seed = 8 # set seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.2,
                                                    random_state=seed)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Instantiate the Random Forest Classifier
RF_clf = RandomForestClassifier(random_state=seed)
RF_clf
# Compute k-fold cross validation on training dataset and see mean accuracy score
cv_scores = cross_val_score(RF_clf,X_train, y_train, cv=10, scoring='accuracy')
print('The accuracy scores for the iterations are {}'.format(cv_scores))
print('The mean accuracy score is {}'.format(cv_scores.mean()))
RF_clf.fit(X_train, y_train)
pred_RF = RF_clf.predict(X_test)
# Print 5 results to see
for i in range(0,5):
    print('Actual wine quality is ', y_test.iloc[i], ' and predicted is ', pred_RF[i])
print(accuracy_score(y_test, pred_RF))
print(log_loss(y_test, pred_RF))
print(confusion_matrix(y_test, pred_RF))
# Import and istantiate the Logistic Regression model
from sklearn.linear_model import LogisticRegression
LR_clf = LogisticRegression(random_state=seed)
LR_clf
# Compute cross validation scores on training dataset and see mean score
cv_scores = cross_val_score(LR_clf, X_train, y_train, cv=10, scoring='accuracy')
print('The cv scores from the iterations are {}'.format(cv_scores))
print('The mean cv score is {}'.format(cv_scores.mean()))
LR_clf.fit(X_train, y_train)
pred_LR = LR_clf.predict(X_test)
# Print 5 results to see
for i in range(0,5):
    print('Actual wine quality is ', y_test.iloc[i], ' and predicted is ', pred_LR[i])
print(accuracy_score(y_test, pred_LR))
print(log_loss(y_test, pred_LR))
print(confusion_matrix(y_test, pred_LR))
from sklearn.model_selection import GridSearchCV
grid_values = {'n_estimators':[50,100,200],'max_depth':[None,30,15,5],
               'max_features':['auto','sqrt','log2'],'min_samples_leaf':[1,20,50,100]}
grid_RF = GridSearchCV(RF_clf,param_grid=grid_values,scoring='accuracy')
grid_RF.fit(X_train, y_train)
grid_RF.best_params_
RF_clf = RandomForestClassifier(n_estimators=100,random_state=seed)
RF_clf.fit(X_train,y_train)
pred_RF = RF_clf.predict(X_test)
print(accuracy_score(y_test,pred_RF))
print(log_loss(y_test,pred_RF))
print(confusion_matrix(y_test,pred_RF))
