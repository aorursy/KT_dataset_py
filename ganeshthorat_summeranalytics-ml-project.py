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
# Import the train and test data



pd.set_option('max_columns', None) #to display all the columns

train_df = pd.read_csv("../input/datasets-attrition-rate/train.csv")

X_test = pd.read_csv("../input/datasets-attrition-rate/test.csv")

train_df.head()
X = train_df.drop(['Attrition','Id','EmployeeNumber','Behaviour'],axis = 1)

Y = train_df['Attrition']

X_test_N = X_test.drop(['Id','EmployeeNumber','Behaviour'],axis = 1)
s = (X.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, Y,

                                                      train_size=0.8, test_size=0.2,

                                                      random_state=0)
from sklearn.preprocessing import LabelEncoder



# Make copy to avoid changing original data 

label_X_train = X_train.copy()

label_X_valid = X_valid.copy()

label_X_test = X_test_N.copy()



# Apply label encoder to each column with categorical data

label_encoder = LabelEncoder()

for col in object_cols:

    label_X_train[col] = label_encoder.fit_transform(X_train[col])

    label_X_valid[col] = label_encoder.transform(X_valid[col])

    label_X_test[col] = label_encoder.transform(X_test_N[col])  
from sklearn import preprocessing



min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

X_train_minmax = min_max_scaler.fit_transform(label_X_train)

X_valid_minmax = min_max_scaler.transform(label_X_valid)

X_test_minmax = min_max_scaler.transform(label_X_test)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



n_estimators = [100, 200, 300, 500]

max_depth = [1, 5, 8, 15, 20, 25]

min_samples_split = [1, 1.5, 2, 5, 8]



hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  

              min_samples_split = min_samples_split)



forest = RandomForestClassifier(random_state = 0)

gridF = GridSearchCV(forest, hyperF, cv = 5, verbose = 3, 

                      n_jobs = -1)



bestF = gridF.fit(X_train_minmax, y_train)

print(bestF.best_params_)
from xgboost import XGBClassifier



gamma = [0.01, 0.1, 0.5, 1]

max_depth = [1, 2, 5, 8, 10,12]

learning_rate = [0.1, 1, 2, 5, 10]



hyperF_XGB = dict(gamma = gamma, max_depth = max_depth,  

              learning_rate = learning_rate)



XGB = XGBClassifier(random_state = 0)

gridF_XGB = GridSearchCV(XGB, hyperF_XGB, cv = 5, verbose = 1, 

                      n_jobs = -1)



bestF_XGB = gridF_XGB.fit(X_train_minmax, y_train)

print(bestF_XGB.best_params_)
from sklearn.ensemble import GradientBoostingClassifier



n_estimators = [100, 200, 300, 500, 700, 1000]

max_depth = [1, 2, 5, 6, 8]

learning_rate = [0.01, 0.05, 0.1, 1, 2,]



hyperF_gb = dict(n_estimators = n_estimators, max_depth = max_depth,

             learning_rate = learning_rate)



gb = GradientBoostingClassifier(random_state = 0)

gridF_gb = GridSearchCV(gb, hyperF_gb, cv = 5, verbose = 1, 

                      n_jobs = -1)



bestF_gb = gridF_gb.fit(X_train_minmax, y_train)

print(bestF_gb.best_params_)
from sklearn import svm



C = [1, 2, 3, 4, 5, 6, 6.5, 7]

kernel = [ 'poly', 'rbf', 'sigmoid']

gamma = ['scale', 'auto']



hyperF_svm = dict(C = C, kernel = kernel,  

              gamma=gamma)



model_svm = svm.SVC(random_state=0)

gridF_svm = GridSearchCV(model_svm, hyperF_svm, cv = 5, verbose = 1, 

                      n_jobs = -1)



bestF_svm = gridF_svm.fit(X_train_minmax, y_train)

print(bestF_svm.best_params_)
RF_reg = RandomForestClassifier(max_depth= 20, min_samples_split= 2, n_estimators= 300, random_state=0)

RF_reg.fit(X_train_minmax,y_train)



from sklearn import metrics

y_pred_class_train = RF_reg.predict(X_train_minmax)



y_pred_class = RF_reg.predict(X_valid_minmax)

print('Validation Accuracy: ')

print(metrics.accuracy_score(y_valid, y_pred_class))



print('\n Confusion Matrix:')

confusion = metrics.confusion_matrix(y_valid, y_pred_class)

print(confusion)



# calculate the fpr and tpr for all thresholds of the classification

fpr, tpr, threshold = metrics.roc_curve(y_valid, y_pred_class)

roc_auc = metrics.auc(fpr, tpr)



# method I: plt

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
Gradient_reg = GradientBoostingClassifier(learning_rate= 0.05, max_depth= 5, n_estimators= 500, random_state=0)

Gradient_reg.fit(X_train_minmax,y_train)



y_pred_class = Gradient_reg.predict(X_valid_minmax)



from sklearn import metrics



print('Confusion Matrix: \n')

confusion = metrics.confusion_matrix(y_valid, y_pred_class)

print(confusion)



# calculate the fpr and tpr for all thresholds of the classification

fpr, tpr, threshold = metrics.roc_curve(y_valid, y_pred_class)

roc_auc = metrics.auc(fpr, tpr)



# method I: plt

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
XGB_reg = XGBClassifier(gamma= 0.1,learning_rate= 0.1, max_depth=12)

XGB_reg.fit(X_train_minmax,y_train)



y_pred_class = XGB_reg.predict(X_valid_minmax)



from sklearn import metrics



print('Confusion Matrix: \n')

confusion = metrics.confusion_matrix(y_valid, y_pred_class)

print(confusion)



# calculate the fpr and tpr for all thresholds of the classification

fpr, tpr, threshold = metrics.roc_curve(y_valid, y_pred_class)

roc_auc = metrics.auc(fpr, tpr)



# method I: plt

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
model_svm = svm.SVC(C= 7, gamma= 'scale', kernel= 'rbf',random_state=0,probability=True)

model_svm.fit(X_train_minmax,y_train)



y_pred_class = model_svm.predict(X_valid_minmax)



from sklearn import metrics



print('Confusion Matrix: \n')

confusion = metrics.confusion_matrix(y_valid, y_pred_class)

print(confusion)



# method I: plt

import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
f_impt= pd.DataFrame(RF_reg.feature_importances_,index=X_train.columns)

f_impt = f_impt.sort_values(by=0,ascending=False)

f_impt.columns = ['feature importance']

f_impt
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_score



def get_score(n_estimators):

    """Return the average MAE over 3 CV folds of random forest model.

    

    Keyword argument:

    n_estimators -- the number of trees in the forest

    """

    # Replace this body with your own code

    my_pipeline = Pipeline(steps=[('model', RandomForestClassifier(n_estimators, random_state=0))])

    scores = -1 * cross_val_score(my_pipeline, X_train_minmax, y_train,

                              cv=10,

                              scoring='accuracy')



    return scores.mean()



results = {}

for i in range(1,8):

    results[50*i] = get_score(50*i)



import matplotlib.pyplot as plt

%matplotlib inline



plt.plot(list(results.keys()), list(results.values()))

plt.show()
attrition_percent = Gradient_reg.predict_proba(X_test_minmax)

dataset = pd.DataFrame({'Attrition': attrition_percent[:, 1]})



Id_col = train_df['Id']

Id_col.head()

dataset.insert(0, "Id",Id_col , True)



dataset.set_index('Id', inplace=True)

print (dataset)