# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn import metrics

from sklearn.feature_selection import RFE

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

import statsmodels.api as sm

import matplotlib.pyplot as plt 

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score 
# Reading data

df = pd.read_csv("../input/equipfails/equip_failures_training_set.csv")
# Splitting the data in training and testing dataset

np.random.seed(1910) 

df = df.sample(frac=1).reset_index(drop=True)

train, test = train_test_split(df, test_size=0.2)

df = train

df_test = test
# Replacing string 'na' by numpy NaN values

df = df.replace('na',np.NaN)

dfnew = df.astype(float)

dfnew.describe()

dfnew=dfnew.fillna(dfnew.median())
# Separating independent and dependent variables

datacols = list(df)

y = dfnew[datacols[1]]

x = dfnew[datacols[2:]]
y = pd.DataFrame(y)

scaled_data = pd.DataFrame()

lcol = len(x.columns)

for i in range(lcol):

    data = x[x.columns[i]]

    data = data.values.astype(float)

    data = data.reshape(-1,1)

    min_max_scaler = preprocessing.MinMaxScaler()

    scaled_array = min_max_scaler.fit_transform(data)

    testdf = pd.DataFrame(scaled_array,columns=[x.columns[i]])

    scaled_data=pd.concat([scaled_data, testdf.reindex(testdf.index)], axis=1)

x_scaled = scaled_data
#Recursive Feature Selection

x_imp_scaled = pd.DataFrame()

logreg = LogisticRegression()

rfe = RFE(logreg, 15)

rfe = rfe.fit(x_scaled, y.values.ravel())

for i in range(len(rfe.support_)):

    if rfe.support_[i] == True :

        print(x_scaled.columns[i])

        testdf = pd.DataFrame(x_scaled[x_scaled.columns[i]],columns=[x_scaled.columns[i]])

        x_imp_scaled = pd.concat([x_imp_scaled, testdf.reindex(testdf.index)], axis=1)

# Replacing 'na' strings with NaN

df_test = df_test.replace('na',np.NaN)

dfnew_test = df_test.astype(float)

dfnew_test.describe()

# Replacing the NaN values with median values

dfnew_test=dfnew_test.fillna(dfnew_test.median())

datacols_test = list(df_test)

y_test = dfnew_test[datacols_test[1]]

x_test = dfnew_test[datacols_test[2:]]

y_test = pd.DataFrame(y_test)

scaled_data = pd.DataFrame()

lcol = len(x_test.columns)

# Normalizing the test data

for i in range(lcol):

    data = x_test[x_test.columns[i]]

    data = data.values.astype(float)

    data = data.reshape(-1,1)

    min_max_scaler = preprocessing.MinMaxScaler()

    scaled_array = min_max_scaler.fit_transform(data)

    testdf = pd.DataFrame(scaled_array,columns=[x_test.columns[i]])

    scaled_data=pd.concat([scaled_data, testdf.reindex(testdf.index)], axis=1)

x_scaled_test = scaled_data

# Slicing the test data to important features only

x_imp_scaled_test = pd.DataFrame()

for i in range(len(rfe.support_)):

    if rfe.support_[i] == True :

        print(x_scaled.columns[i])

        testdf = pd.DataFrame(x_scaled_test[x_scaled.columns[i]],columns=[x_scaled.columns[i]])

        x_imp_scaled_test = pd.concat([x_imp_scaled_test, testdf.reindex(testdf.index)], axis=1)

logreg.fit(x_imp_scaled, y)

# Predicting on logistic regression model

y_test_pred = logreg.predict(x_imp_scaled_test)

# Printing the score of the model

logreg.score(x_imp_scaled_test, y_test)

# Creating the confusion matrix and printing it

confusion_matrix_logit = confusion_matrix(y_test, y_test_pred)

print(confusion_matrix_logit)

# Printing classification report

print(classification_report(y_test, y_test_pred))
# ROC Curve

logit_roc_auc = roc_auc_score(y_test, y_test_pred)

fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(x_imp_scaled_test)[:,1])

plt.figure()

plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)

#plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")

plt.savefig('Log_ROC')

plt.show()
print ("Accuracy : ", accuracy_score(y_test,y_test_pred)*100) 
model_gini = DecisionTreeClassifier(criterion = "gini", 

                random_state = 100,max_depth=3, min_samples_leaf=5) 

# Performing training 

model_gini.fit(x_imp_scaled, y) 

y_gini_pred = model_gini.predict(x_imp_scaled_test)

confusion_matrix_gini = confusion_matrix(y_test,y_gini_pred)

print(confusion_matrix_gini)

print(classification_report(y_test, y_gini_pred))

print ("Accuracy : ", accuracy_score(y_test,y_gini_pred)*100) 
data_test = pd.read_csv("../input/equipfails/equip_failures_test_set.csv")

data_test = data_test.replace('na',np.NaN)

datanew_test = data_test.astype(float)

datanew_test.describe()

datanew_test=datanew_test.fillna(datanew_test.median())

datacols_test = list(data_test)

x_test = datanew_test[datacols_test[:]]

scaled_data = pd.DataFrame()

lcol = len(x_test.columns)

for i in range(lcol):

    data = x_test[x_test.columns[i]]

    data = data.values.astype(float)

    data = data.reshape(-1,1)

    min_max_scaler = preprocessing.MinMaxScaler()

    scaled_array = min_max_scaler.fit_transform(data)

    testdf = pd.DataFrame(scaled_array,columns=[x_test.columns[i]])

    scaled_data=pd.concat([scaled_data, testdf.reindex(testdf.index)], axis=1)

x_scaled_data_test = scaled_data

# Slicing the test data to important features only

x_imp_scaled_data_test = pd.DataFrame()

for i in range(len(rfe.support_)):

    if rfe.support_[i] == True :

        print(x_scaled.columns[i])

        testdf = pd.DataFrame(x_scaled_data_test[x_scaled.columns[i]],columns=[x_scaled.columns[i]])

        x_imp_scaled_data_test = pd.concat([x_imp_scaled_data_test, testdf.reindex(testdf.index)], axis=1)
# Predicting on new test data

y_gini_pred_data = model_gini.predict(x_imp_scaled_data_test)

pred_df = pd.DataFrame(y_gini_pred_data,columns=['target'])

pred_df.reset_index(level=0, inplace=True)

pred_df.columns =['id','target']

pred_df.to_csv('myPrediction_onTestData.csv',index=False)