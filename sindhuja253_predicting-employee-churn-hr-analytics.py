import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz

from subprocess import check_call

from sklearn.model_selection import GridSearchCV



from PIL import Image, ImageDraw, ImageFont

from sklearn.metrics import confusion_matrix,precision_score,roc_auc_score,recall_score,accuracy_score

# Read the file

data = pd.read_csv("../input/turnover.csv")

data.head()
#  check for any null values

data.isnull().any()
# Encoding salary variable to categories

# Encoding categories of the salary variable, which you know is ordinal based on the values you observed

print(data.sales.unique())

print(data.salary.unique())

data['salary'] = data['salary'].astype('category').cat.reorder_categories(['low','medium','high']).cat.codes

data.head()

#Proportion of employees left by department

pd.crosstab(data.sales, data.left)
# Bar chart of turnover frequency for department column

pd.crosstab(data.sales,data.left).plot(kind='bar')

plt.title('Turnover Frequency for Department')

plt.xlabel('Department')

plt.ylabel('Frequency of Turnover')

plt.savefig('department_bar_chart')
#Bar chart for employee salary level and the frequency of turnover

table=pd.crosstab(data.salary, data.left)

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

plt.title('Stacked Bar Chart of Salary Level vs Turnover')

plt.xlabel('Salary Level')

plt.ylabel('Proportion of Employees')

plt.savefig('salary_bar_chart')
#Histogram of numeric variables

num_bins = 10



data.hist(bins=num_bins, figsize=(20,15))

plt.savefig("hr_histogram_plots")

plt.show()
#  Getting dummies

# You will now transform the department variable,

# which you know is nominal based on the values you observed. 

# To do that, you will use so-called dummy variables.

 

department = pd.get_dummies(data['sales'])

# To avoid dummy trap

#A dummy trap is a situation where different dummy variables convey the same information. In this case, if an employee is, say, from the accounting department (i.e. value in the accounting column is 1), then you're certain that s/he is not from any other department (values everywhere else are 0). Thus, you could actually learn about his/her department by looking at all the other departments.

data = data.drop(["sales"],axis = 1)

data = data.join(department)

data.head()
### train test split

target = data.left

features = data.drop(['left'],axis = 1)

X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.25,random_state=42)
## Predicting employee churn using decision trees and calculation of accuracy scores for training and testing set respectively

# Decision Tree

model = DecisionTreeClassifier(criterion='entropy',max_depth=8,min_samples_leaf=100)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

# print("Training Accuracy", model.score(X_train,y_train)*100)

# print("Test Accuracy",model.score(X_test,y_test)*100)

# Confusion matrix

print(confusion_matrix(y_test,y_pred))

print("precision:",precision_score(y_test,y_pred))

print("Recall:",recall_score(y_test,y_pred) )

# AUC - ROC Curve

print("ROC_AUC_score:",roc_auc_score(y_test,y_pred))
# Setting up GridSearch parameters

# Generate values for maximum depth

depth = [i for i in range(5,21,1)]

samples = [i for i in range(50,500,50)]

parameters = dict(max_depth=depth, min_samples_leaf=samples)

parameters = dict(max_depth=depth, min_samples_leaf=samples)

param_search = GridSearchCV(model, parameters)

param_search.fit(X_train, y_train)

print(param_search.best_params_)
# Sorting important features,Selecting important features

# Calculate feature importances

feature_importances = model.feature_importances_

feature_list = list(features)

relative_importances = pd.DataFrame(index=feature_list, data=feature_importances, columns=["importance"])

relative_importances.sort_values(by="importance", ascending=False)

feature_importances = model.feature_importances_

feature_list = list(features)

relative_importances = pd.DataFrame(index=feature_list, data=feature_importances, columns=["importance"])

relative_importances.sort_values(by="importance", ascending=False)

selected_features = relative_importances[relative_importances.importance>0.01]

selected_list = selected_features.index

features_train_selected = X_train[selected_list]

features_test_selected = X_test[selected_list]
from sklearn.metrics import roc_auc_score,roc_curve

# Initialize the best model using parameters provided in description

model_best = DecisionTreeClassifier(criterion='entropy',max_depth=5, min_samples_leaf=50,class_weight="balanced", random_state=42)

model_best.fit(features_train_selected, y_train)

prediction_best = model_best.predict(features_test_selected)

rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, prediction_best)

print(confusion_matrix(y_test,prediction_best))

print("precision:",precision_score(y_test,prediction_best))

print("Recall:",recall_score(y_test,prediction_best) )

# AUC - ROC Curve

roc_auc = roc_auc_score(y_test,prediction_best)

print("ROC_AUC_score:",roc_auc)

plt.figure(figsize=(5,5))

plt.plot(rf_fpr, rf_tpr, label='Decision Tree (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('ROC')

plt.show()
# Evaluation Metrics results for Grid Search Decision tree



print("precision:",precision_score(y_test,prediction_best))

print("Recall:",recall_score(y_test,prediction_best))

print("ROC_AUC_score:",roc_auc)

# Evaluation Metrics results old Decsion tree model

print("precision:",precision_score(y_test,y_pred))

print("Recall:",recall_score(y_test,y_pred) )

print("ROC_AUC_score:",roc_auc_score(y_test,y_pred))