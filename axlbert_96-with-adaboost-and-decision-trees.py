import pandas as pd

import numpy as np

from IPython.display import display

import matplotlib.pyplot as plt

import seaborn as sns

from time import time

##without this matplotlib will render somwehere outside jupyter###

%matplotlib inline
full_data = pd.read_csv("../input/HR_comma_sep.csv")

display(full_data.describe())
print(full_data.keys())
indices = [5,500,5000]



samples = pd.DataFrame(full_data.loc[indices], columns = full_data.keys())

display(samples)
count_left = np.count_nonzero(full_data.left == 1)

count_promo = np.count_nonzero(full_data.promotion_last_5years == 1)

count_accidents = np.count_nonzero(full_data.Work_accident == 1)

left_percent = float(count_left)/float(full_data.shape[0])*100





print("we observed %s people that left the company" % count_left)

print("This is an equivalent of {:.2f}% of the analyzed workforce".format(left_percent))

print("we observed %s people with promotions" % count_promo)

print("we observed %s people that had accidents" % count_accidents)
target = full_data['left']

features = full_data.drop('left', axis = 1)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

numerical = ["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company"]

features[numerical] = scaler.fit_transform(features[numerical])



# Show an example of a record with scaling applied

display(features.head(n = 5))
featureshot = pd.get_dummies(features)

encoded = list(featureshot.columns)



print(encoded)
display(featureshot.head(n = 3))
corr = featureshot.corr()

# plot the heatmap

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask, 1)] = True



with sns.axes_style("white"):

    sns.heatmap(corr, mask=mask, annot=True, annot_kws={"size": 7}, cmap='RdBu', fmt='+.2f', cbar=False)
empleft = full_data[full_data.left == 1]

lefthot = pd.get_dummies(empleft)

empstay = full_data[full_data.left == 0]

stayhot = pd.get_dummies(empstay)
group1 = ["satisfaction_level","number_project","average_montly_hours","time_spend_company"]

scatter_left = lefthot[group1]

pd.scatter_matrix(scatter_left, alpha = 0.3, figsize = (14,8), diagonal = 'kde');
from collections import Counter as ct

data1 = empleft["sales"]

departments_count = ct(data1)

df = pd.DataFrame.from_dict(departments_count, orient='index')

df.plot(kind='bar')

plt.title("Number of people that left by department")

data1 = stayhot["satisfaction_level"]

data2= lefthot["satisfaction_level"]

plt.hist(data1, label='Stayed')

plt.hist(data2, label='Left')

plt.title("Employee Satisfaction")

plt.ylabel('Samples')

plt.ylabel('Satisfaction')

legend = plt.legend(loc='upper left', shadow=True)
data1 = stayhot["average_montly_hours"]

data2= lefthot["average_montly_hours"]

plt.hist(data1, label='Stayed')

plt.hist(data2, label='Left')

plt.title("Monthly hours worked")

plt.ylabel('Samples')

plt.ylabel('Hours worked')

legend = plt.legend(loc='upper left', shadow=True)
data1 = stayhot["number_project"]

data2= lefthot["number_project"]

plt.hist(data1, label='Stayed')

plt.hist(data2, label='Left')

plt.title("Projects conducted")

plt.ylabel('Samples')

plt.ylabel('number projects')

legend = plt.legend(loc='upper left', shadow=True)
from sklearn.cross_validation import train_test_split



X_train, X_test, y_train, y_test = train_test_split(featureshot, target, test_size = 0.3, random_state = 42)





print("Training set has {} samples.".format(X_train.shape[0]))

print("Testing set has {} samples.".format(X_test.shape[0]))
from sklearn.metrics import make_scorer, fbeta_score, accuracy_score

from sklearn.grid_search import GridSearchCV

from sklearn.tree import DecisionTreeClassifier



clf = DecisionTreeClassifier()



parameters = parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)},{'min_samples_split': (0.1,0.2,0.3,0.4,0.5)},{'min_samples_leaf': (1,2,3,4,5,6,7)},{'min_weight_fraction_leaf': (0.0,0.1,0.2)}

scorer = make_scorer(fbeta_score,beta=0.5)



grid_obj = GridSearchCV(clf, param_grid = parameters, scoring=scorer)

grid_fit = grid_obj.fit(X_train,y_train)



best_clf = grid_fit.best_estimator_

predictions = (clf.fit(X_train, y_train)).predict(X_test)

best_predictions = best_clf.predict(X_test)
print(clf)

from IPython.display import display

#display(pd.DataFrame(grid_obj.grid_scores_))



print ("Unoptimized model\n------")

print ("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))

print ("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))

print ("\nOptimized Model\n------")

print ("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))

print ("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
from sklearn.ensemble import AdaBoostClassifier





regr2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=None),n_estimators=100, random_state=42)



regr2.fit(X_train,y_train)



pred_new =(regr2.predict(X_test))

print ("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, pred_new)))

print ("F-score on testing data: {:.4f}".format(fbeta_score(y_test, pred_new, beta = 0.5)))