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

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import os

import csv
US_data = pd.read_csv("/kaggle/input/us-college-data/College_Data.csv")

US_data = US_data.rename(columns = {"Unnamed: 0":"US_College_name"})

US_data.head()
print("Count of missing values to each column is")

US_data.isnull().sum()
US_data.info()
US_data1 = US_data[US_data.columns.difference(['US_College Name'])]

US_data1["Number_of_University"] = 1
x = '''

     Table gives information about Private and Public university in US

     It contains following information

     Number of university in private and public sector

     number of application sent

     number of application accepted

     number of students enroll in university

     number of phd faculty

     Note:Yes = Private college and No = Goverment College\n\n'''

      

print(x ,US_data1.groupby("Private")["Number_of_University","Apps","Accept","Enroll","PhD"].sum())
print("Distribution of Number of fulltime undergraduate w.r.t. university")

sns.swarmplot(x = 'Private',y = 'F.Undergrad',data = US_data)
print("Distribution of Number of Parttime undergraduate w.r.t. university")

sns.swarmplot(x = 'Private',y = 'P.Undergrad',data = US_data)
print("Distribution of Number of out of state w.r.t. university")

sns.swarmplot(x = 'Private',y = 'Outstate',data = US_data)
print("Distribution of perctage of alumni who donated w.r.t. university")

sns.swarmplot(x = 'Private',y = 'perc.alumni',data = US_data)
print("Distribution of Graduation rate  w.r.t. university")

sns.swarmplot(x = 'Private',y = 'Grad.Rate',data = US_data)
print("Building Random Foreset model has Started:")
print("seperating data into the two part \n1)Input variables which X and 2)Output variable which y ")

X = US_data[US_data.columns.difference(['Private'])]

y = US_data['Private']

y = y.replace('Yes',1)

y = y.replace('No',0)
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

from sklearn.metrics import mean_squared_error

print("Seperating X and y data into two part \n80% belongs to trian while 20% belongs to test \nAnd droping string variables")

X_train,val_X,y_train,val_y = train_test_split(X,y,test_size = 0.2,random_state = 0)

X_train_1 = X_train[X_train.columns.difference(['US_College_name'])]

val_X_1 = val_X[val_X.columns.difference(['US_College_name'])]

US_data_model = RandomForestClassifier(n_estimators = 500,bootstrap = True,max_features = 'sqrt')

US_data_model.fit(X_train_1,y_train)
print("Average absolute error value is " ,mean_absolute_error(val_y,US_data_model.predict(val_X_1)))

print("Average error square value is" ,mean_squared_error(val_y,US_data_model.predict(val_X_1)))

print("Root mean square error value is",np.sqrt(mean_squared_error(val_y,US_data_model.predict(val_X_1))))
y_pred_test = US_data_model.predict_proba(val_X_1)[:,1]

y_pred_train = US_data_model.predict_proba(X_train_1)[:,1]
from sklearn.metrics import roc_auc_score,average_precision_score,auc,roc_curve,precision_recall_curve
print("ROC Curve")

fpr , tpr ,thresold = roc_curve(val_y,y_pred_test)

roc_auc = auc(fpr,tpr)

plt.plot(fpr,tpr,label = 'ROC curve (area = %0.2f)'% roc_auc)

plt.xlabel("False Positve rate")

plt.ylabel("True Positive rate")

plt.legend(loc = 'lower right')
print("Precision Vs Recall Plot")

precision , recall , threshold = precision_recall_curve(val_y,y_pred_test)

average_precision =  average_precision_score(val_y,y_pred_test)

plt.plot(recall,precision,label = 'Precision recall curve (area = %0.2f)'% average_precision)

plt.xlabel("recall")

plt.ylabel("Precision")

plt.legend(loc = 'lower right')
y_pred_test = np.where(y_pred_test > 0.332,1,0)

y_pred_train = np.where(y_pred_train > 0.332,1,0)
print("Confusion Matrix using test values")

matrix = confusion_matrix(val_y,y_pred_test)

sns.heatmap(matrix ,annot = True,cbar = True)
print("Confusion Matrix using train values")

matrix = confusion_matrix(y_train,y_pred_train)

sns.heatmap(matrix ,annot = True,cbar = True)
print("Following is Actual and predicted value table")

prediction_data = pd.DataFrame(val_X['US_College_name'])

prediction_data['Predicted_value'] = y_pred_test

prediction_data['Actual_value'] = val_y

prediction_data = prediction_data.sort_index(axis = 0)

prediction_data.head()
print("Number of wrong prediction is ",prediction_data[prediction_data['Predicted_value'] != prediction_data['Actual_value']].Predicted_value.count()," out of total ",prediction_data['Predicted_value'].count(),"\nAnd Percentage of wrong prediction is ",round(prediction_data[prediction_data['Predicted_value'] != prediction_data['Actual_value']].Predicted_value.count()/prediction_data['Predicted_value'].count(),4),"\nNote Yes = 1 and No = 0 ")