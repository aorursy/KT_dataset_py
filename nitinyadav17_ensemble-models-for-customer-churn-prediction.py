import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import pandas_profiling as pp

import seaborn as sns

from sklearn.metrics import classification_report

from sklearn.metrics import average_precision_score

from sklearn.metrics import roc_auc_score

from xgboost import plot_importance

from sklearn.model_selection import RandomizedSearchCV 

from sklearn.utils import resample,shuffle
cust_churn = pd.read_csv('../input/mobile-company-customers-retention/Cellphone (1).csv')
cust_churn.shape
cust_churn.head(10)
cust_churn['Churn'].value_counts(normalize = True)*100
cust_churn.info()
cust_final = cust_churn
report = pp.ProfileReport(cust_final)
report
cust_final['CustServCalls'] = np.where(cust_final['CustServCalls']>3.5,3.5,cust_final['CustServCalls'])

cust_final['MonthlyCharge'] = np.where(cust_final['MonthlyCharge']>98,98,cust_final['MonthlyCharge'])
fig, ax = plt.subplots(figsize=(20,10))         

corr = cust_final.corr()

sns.heatmap(corr, cmap='YlGnBu', annot_kws={'size':30}, ax=ax)

ax.set_title("Correlation Matrix", fontsize=14)

plt.show()
cust_final = cust_final.drop(columns=['DataPlan','DataUsage'])
X = cust_final.drop(columns = 'Churn')

y = cust_final['Churn']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 123)
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)

classifier.fit(X_train,y_train)
from sklearn.metrics import confusion_matrix

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

print(cm)
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
print(classification_report(y_test,y_pred))
print('AUPRC = {}'.format(roc_auc_score(y_test, y_pred)))
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(criterion='entropy', random_state=0,n_estimators=100)

classifier.fit(X_train,y_train)
from sklearn.metrics import confusion_matrix

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

print(cm)
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
from sklearn.model_selection import GridSearchCV

parameters = [{'criterion':['entropy','gini'],'min_samples_split':range(2,10),'n_estimators':range(50,250,50),'max_depth':range(5,20,1)}]

grid_search = RandomizedSearchCV(estimator = classifier,

                           param_distributions= parameters,

                           scoring = 'accuracy',

                           cv = 10,

                           n_jobs = -1)

grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_

best_parameters = grid_search.best_params_

print("Best Accuracy: {:.2f} %".format(best_accuracy*100))

print("Best Parameters:", best_parameters)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(criterion='entropy', random_state=0,n_estimators=200,min_samples_split = 8,max_depth = 15)

classifier.fit(X_train,y_train)
from sklearn.metrics import confusion_matrix

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

print(cm)
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
print(classification_report(y_test,y_pred))
print('AUPRC = {}'.format(roc_auc_score(y_test, y_pred)))


from xgboost.sklearn import XGBClassifier

classifier = XGBClassifier()

classifier.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

print(cm)
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
param_grid = {'n_estimators': range(50,500,50),

                    'learning_rate': [ 0.1, 0.15,0.2,0.25],

                    'gamma':  [0.20,0.10, 0.15],

                    'max_delta_step': [24, 26, 22],

                    'max_depth':range(2,20),

             'min_child_weight': [1, 2, 3, 4]}       



ransearch = RandomizedSearchCV(classifier, n_iter = 50, param_distributions=param_grid, cv=10, n_jobs=-1, verbose=2)

ransearch = ransearch.fit(X_train,y_train)

best_accuracy = ransearch.best_score_

best_parameter = ransearch.best_params_

print('Accuracy: {:.2f}%'.format(best_accuracy*100))

print('parameter:',best_parameter)
classifier = XGBClassifier(n_estimators=100,

 max_depth= 17,

 max_delta_step = 24,

 learning_rate = 0.25,

 gamma = 0.15,

min_child_weight = 1)

classifier.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(cm)

print(classification_report(y_test,y_pred))
from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)

print("Accuracy: {:.2f} %".format(accuracies.mean()*100))

print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
print('AUPRC = {}'.format(roc_auc_score(y_test, y_pred)))
fig = plt.figure(figsize = (14, 9))

ax = fig.add_subplot(111)



colours = plt.cm.Set1(np.linspace(0, 1, 9))



ax = plot_importance(classifier, height = 1, color = colours, grid = False, \

                     show_values = False, importance_type = 'cover', ax = ax);

for axis in ['top','bottom','left','right']:

            ax.spines[axis].set_linewidth(2)

        

ax.set_xlabel('importance score', size = 16);

ax.set_ylabel('features', size = 16);

ax.set_yticklabels(ax.get_yticklabels(), size = 12);

ax.set_title('Ordering of features by importance to the model learnt', size = 20);