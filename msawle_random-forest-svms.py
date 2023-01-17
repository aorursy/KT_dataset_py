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
# Importing the required libabries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import roc_curve, auc

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
data = pd.read_csv(r'/kaggle/input/hr-attrition/HR_Employee_Attrition_Data.csv')
# Description of DataFrame



print("The data has {} rows and {} columns".format(data.shape[0], data.shape[1]))

print('#'*75)

print(data.dtypes)

data_cols = list(data.columns)

print(data_cols)
# Description of the dataset



print(data.describe(include='all').T)
data.head()
# Resetting the index

data.set_index('EmployeeNumber', inplace = True)
data.head()
# Mapping the catagorical variable 'Attrition' to 'Numerical' values using map



data['Attrition'] = data['Attrition'].map({'Yes':1, 'No':0})



# 1 Indicates employee resigning and 0 indicates employee staying with the Org
data.head(10)
cols_object = [var for var in data.columns if data[var].dtype == 'O']

print(cols_object)
data.drop('Over18', axis = 1, inplace = True)
from sklearn import preprocessing



def preprocessor(df):

    res_df = df.copy()

    le = preprocessing.LabelEncoder()

    

    res_df['BusinessTravel'] = le.fit_transform(res_df['BusinessTravel'])

    res_df['Department'] = le.fit_transform(res_df['Department'])

    res_df['EducationField'] = le.fit_transform(res_df['EducationField'])

    res_df['Gender'] = le.fit_transform(res_df['Gender'])

    res_df['JobRole'] = le.fit_transform(res_df['JobRole'])

    res_df['MaritalStatus'] = le.fit_transform(res_df['MaritalStatus'])

    res_df['OverTime'] = le.fit_transform(res_df['OverTime'])

    

    return res_df
encoded_df = preprocessor(data)
encoded_df.head()

print(encoded_df.dtypes)
feature_space = encoded_df.iloc[:, encoded_df.columns != 'Attrition']

feature_class = encoded_df.iloc[:, encoded_df.columns == 'Attrition']
X_train, X_test, y_train, y_test = train_test_split(feature_space, feature_class, test_size = 0.2, random_state =42 )
X_train.values
rf = RandomForestClassifier(random_state = 42)
y_test = y_test.values.ravel() 

y_train = y_train.values.ravel() 
import time

np.random.seed(42)



start = time.time()



param_dist = {'max_depth':[2,3,4,5,6,7,8],

             'bootstrap':[True, False],

             'max_features':['auto', 'sqrt', 'log2', None],

             'criterion':['gini', 'entropy']}



cv_rf = GridSearchCV(rf, cv = 10, param_grid = param_dist, n_jobs = 3)

cv_rf.fit(X_train, y_train)

print('Best Parameters using grid search: \n', cv_rf.best_params_)

end = time.time()

print('Time taken in grid search: {0: .2f}'.format(end - start))
rf.set_params(criterion = 'entropy',

                  max_features = None, 

                  max_depth = 8, bootstrap = True)
rf.set_params(warm_start = True, oob_score = True)
# Estimation of the error rate for each n_estimators



# For estimating n_estimators, warm_start has to be set as True and oob_score as True

# rf.set_params(***) - sets the parameters for the model defined earlier. 

# In thi scase, rf is the model name for RandomForestClassifier



min_estimators = 100

max_estimators = 1000











error_rate = {}



for i in range(min_estimators, max_estimators+1):

    rf.set_params(n_estimators = i)

    rf.fit(X_train, y_train)

    oob_error = 1 - rf.oob_score_

    error_rate[i] = oob_error
oob_series = pd.Series(error_rate)
plt.style.use('ggplot')

fig, ax = plt.subplots(figsize=(16, 12))



ax.set_facecolor('#e6e6ff')



oob_series.plot(kind='line',color = 'red')

plt.axhline(0.074, color='#33FFE5',linestyle='--')

plt.axhline(0.071, color='#33FFE5',linestyle='--')

plt.xlabel('n_estimators')

plt.ylabel('OOB Error Rate')

plt.title('OOB Error Rate Across various Forest sizes \n(From 100 to 1000 trees)')

plt.show()
for i in range(100, 1000, 100):

    print('OOB Error rate for {} trees is {}'.format(i, oob_series[i]))
# Refine the tree via OOB Output

rf.set_params(n_estimators=800,

                  bootstrap = True,

                  warm_start=False, 

                  oob_score=False)
rf.fit(X_train, y_train)
cols = [var for var in X_train.columns]

cols_df = pd.DataFrame(cols, columns = ['Feature_Name'])



importance = list(rf.feature_importances_)

print(importance, len(importance))
imp = pd.DataFrame(importance, columns = ['Importance'])

feature_imp = pd.concat([cols_df, imp], axis = 1)
feature_imp
# Plotting a barplot to identify & visualize feature importance

plt.figure(figsize=(16,12))

x = sns.barplot(feature_imp['Feature_Name'], feature_imp['Importance'])

x.set_xticklabels(labels=feature_imp.Feature_Name.values, rotation=90)

plt.show()
predictions = rf.predict(X_test)



probability = rf.predict_proba(X_test)



fpr, tpr, threshold = roc_curve(y_test, probability[:,1])

print(fpr)

print(tpr)

print(threshold)

# Printing of the Confusion Matrix



cm = confusion_matrix(y_test, predictions)



print(cm)

print(type(cm))
# Code to plot Confusion matrix in graphical way



sns.set(font_scale=1.8) # scaling the font sizes

plt.figure(figsize=(10,10))



sns.heatmap(cm, annot=True, cbar=False, fmt = '', xticklabels = ['TRUE', 'FALSE'], yticklabels = ['TRUE', 'FALSE'])

plt.xlabel('Predicted', color = 'blue', fontsize = 'xx-large' )

plt.ylabel('Actual', color = 'blue', fontsize = 'xx-large')

plt.show()
TP = cm[0][0]

FP = cm[0][1]

FN = cm[1][0]

TN = cm[1][1]



print(TP, FP, FN, TN)
Recall = (TP / (TP+FN))

Specificity = (TN / (FP+TN))

Accuracy = ((TP+TN) / (TP+TN+FP+FN))





print("The Recall score is: {} ".format(np.round(Recall,2)))

print("The Specificity score is : {}".format(np.round(Specificity,2)))

print("The Accuracy score is: {}".format(np.round(Accuracy,2)))
# The Accuracy score obtained using rf.score and manual calculation yield the same result

print(rf.score(X_test, y_test))
roc_auc = auc(fpr, tpr)



plt.figure(figsize=(12,10))



plt.plot(fpr, tpr, color = 'red', lw =2, label = 'Decision Tree (AUC = {})'.format(np.round(roc_auc, 2)))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')



plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Area Under Curve')

plt.legend(loc="lower right")

plt.show()
data_svm = encoded_df.copy()



from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()
cont_cols = [var for var in data_svm.columns if data_svm[var].dtype != 'O']

print(cont_cols)
corr_mat = data_svm.corr()

sns.set(font_scale=1.5)

f, ax = plt.subplots(figsize=(20, 16))

sns.heatmap(corr_mat, vmax =1, annot = True , square = False, annot_kws={"size":15},  cbar = False, fmt=".2f", cmap='coolwarm')





# Arguments used for heatmap

# cmap = colormap (coolwarm )
cols_drop = ['JobLevel', 'YearsWithCurrManager', 'StandardHours', 'EmployeeCount', 'YearsInCurrentRole']
data_wo_corr = data_svm.drop(cols_drop, axis = 1)

data_wo_corr.head()
# feature separation



X = data_wo_corr.drop(['Attrition'], axis = 1)

y = data_wo_corr['Attrition']





scaler.fit(X)

X = scaler.transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.svm import SVC

from sklearn import metrics

svc=SVC() #Default hyperparameters

svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)

print('Accuracy Score:')

print(metrics.accuracy_score(y_test,y_pred))

# Using Linear Kernel



svc=SVC(kernel='linear')

svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)

print('Accuracy Score:')

print(metrics.accuracy_score(y_test,y_pred))
# Using Polynomial kernel



svc=SVC(kernel='poly')

svc.fit(X_train,y_train)

y_pred=svc.predict(X_test)

print('Accuracy Score:')

print(metrics.accuracy_score(y_test,y_pred))
kernel_choice = ['linear', 'poly', 'rbf']



for val in kernel_choice:

    svc = SVC(kernel = val)

    svc.fit(X_train, y_train)

    y_pred = svc.predict(X_test)

    print("Accuracy score using {} kernel is: {}".format(val, metrics.accuracy_score(y_test,y_pred)))
#from sklearn.cross_validation import cross_val_score

from sklearn.model_selection import cross_val_score



C_range=list(range(1,26))

acc_score=[]

for c in C_range:

    svc = SVC(kernel='linear', C=c)

    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

    acc_score.append(scores.mean())

print(acc_score)
import matplotlib.pyplot as plt

%matplotlib inline





C_values=list(range(1,26))

# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)

plt.plot(C_values,acc_score)

plt.xticks(np.arange(0,27,2))

plt.xlabel('Value of C for SVC')

plt.ylabel('Cross-Validated Accuracy')
C_range=list(np.arange(0.1,6,0.1))

acc_score=[]

for c in C_range:

    svc = SVC(kernel='linear', C=c)

    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

    

    acc_score.append(scores.mean())

print(acc_score)    
import matplotlib.pyplot as plt

%matplotlib inline



C_values=list(np.arange(0.1,6,0.1))

# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)



plt.figure(figsize=(16,8))

plt.plot(C_values,acc_score)

plt.xticks(np.arange(0.0,6,0.3))

plt.xlabel('Value of C for SVC ')

plt.ylabel('Cross-Validated Accuracy')
gamma_range=[0.0001,0.001,0.01,0.1,1,10,100]

acc_score=[]

for g in gamma_range:

    svc = SVC(kernel='rbf', gamma=g)

    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

    acc_score.append(scores.mean())

print(acc_score)  
import matplotlib.pyplot as plt

%matplotlib inline



gamma_range=[0.0001,0.001,0.01,0.1,1,10,100]



# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)

plt.figure(figsize = (16,12))

plt.plot(gamma_range,acc_score)

plt.xlabel('Value of gamma for SVC ')

plt.xticks(np.arange(0.0001,100,5))

plt.ylabel('Cross-Validated Accuracy')
from sklearn.svm import SVC

svm_model= SVC()



tuned_parameters = { 'kernel': ['linear', 'rbf', 'poly'],

 'C': (np.arange(0.1,1,0.1)) , 'gamma': [0.01,0.02,0.03,0.04,0.05], 'degree': [2,3,4] }
#from sklearn.grid_search import GridSearchCV

from sklearn.model_selection import GridSearchCV



model_svm = GridSearchCV(svm_model, tuned_parameters,cv=10,scoring='accuracy')



model_svm.fit(X_train, y_train)

print(model_svm.best_score_)
print(model_svm.best_params_)
svm_model.set_params(C = 0.9, degree = 3, gamma = 0.05, kernel = 'poly', probability = True)
svm_model.fit(X_train, y_train)



y_pred = svm_model.predict(X_test)
proba = svm_model.predict_proba(X_test)



fpr, tpr, threshold = roc_curve(y_test, proba[:,1])

print(fpr)

print(tpr)

print(threshold)
cm = confusion_matrix(y_test, y_pred)



print(cm)

print(type(cm))
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(svm_model, X_test, y_test)
from sklearn.metrics import ConfusionMatrixDisplay



ConfusionMatrixDisplay(cm,display_labels = [1,0]).plot()
# Code to plot Confusion matrix in graphical way



sns.set(font_scale=1.8) # scaling the font sizes

plt.figure(figsize=(10,10))



sns.heatmap(cm, annot=True, cbar=False, fmt = '', xticklabels = ['TRUE', 'FALSE'], yticklabels = ['TRUE', 'FALSE'])

plt.xlabel('Predicted', color = 'blue', fontsize = 'xx-large' )

plt.ylabel('Actual', color = 'blue', fontsize = 'xx-large')

plt.show()
TP = cm[0][0]

FP = cm[0][1]

FN = cm[1][0]

TN = cm[1][1]



print(TP, FP, FN, TN)



Recall = (TP / (TP+FN))

Specificity = (TN / (FP+TN))

Accuracy = ((TP+TN) / (TP+TN+FP+FN))





print("The Recall score is: {} ".format(np.round(Recall,2)))

print("The Specificity score is : {}".format(np.round(Specificity,2)))

print("The Accuracy score is: {}".format(np.round(Accuracy,2)))
roc_auc = auc(fpr, tpr)



plt.figure(figsize=(12,10))



plt.plot(fpr, tpr, color = 'red', lw =2, label = 'Decision Tree (AUC = {})'.format(np.round(roc_auc, 2)))

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')



plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Area Under Curve')

plt.legend(loc="lower right")

plt.show()