# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# I import the libraries as and when they are required in the code. 
# Load the data
raw_data = pd.read_csv('../input/indian_liver_patient.csv')
ip_data = raw_data.copy()
#ip_data
# See the types of data and missing values in the dataset.
# Categorical data (like gender) need to be converted using dummy variables. 
ip_data.info()
#Since there are only 4 records with null values in Albumin_and_Globulin_Ratio column, we can drop those records.
ip_data = ip_data.dropna(how='any', axis = 0)

# Convert Gender to 0s (for Male) and 1s (for Female)
ip_data['Gender'] = ip_data['Gender'].map({'Male':0, 'Female':1})
ip_data['Dataset'] = ip_data['Dataset'].map({2:0, 1:1}) # To solve ValueError: endog must be in the unit interval.

# Check if mapping happened properly
ip_data
# Correlations between variables help us identify the features that can be excluded. 
# We can exclude one in two features which has a strong correlatoin (|correlation| > 0.5) with another feature
# e.g: In the below given case, Total_Bilirubin and Direct_Bilirubin are strongly correlated. So we can
# discard one of those features. The exclusion of features happens 2 cells down.  
ip_corr = ip_data.drop(['Gender', 'Dataset'], axis = 1)
ip_corr.corr()
samples_count = ip_data.shape[0]
# You will see drastic change in models' accuracy when you change the train and test sample proportion (80:20, 70:30 etc)
ip_train_count = int(0.8*samples_count) 
ip_test_count = samples_count - ip_train_count

ip_train_data = ip_data[:ip_train_count]
ip_test_data = ip_data[ip_train_count:]
print(ip_train_count)
print(ip_test_count)
import statsmodels.api as sm
## You can start with all features and then optimize the model by removing the features from the model.

X1_train = ip_train_data[['Age','Gender','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase',
              'Aspartate_Aminotransferase','Total_Protiens','Albumin', 'Albumin_and_Globulin_Ratio' ]]

#X1_train = ip_train_data[['Age', 'Direct_Bilirubin','Alamine_Aminotransferase','Total_Protiens','Albumin']]

X_train= sm.add_constant(X1_train)
#X_train= X1_train.copy() # Independent variables without constant. I have doubts in the use of vaiables without adding a constant
y_train = ip_train_data['Dataset']


X1_test = ip_test_data[['Age','Gender','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase',
              'Aspartate_Aminotransferase','Total_Protiens','Albumin', 'Albumin_and_Globulin_Ratio' ]]

#X1_test = ip_test_data[['Age','Direct_Bilirubin','Alamine_Aminotransferase','Total_Protiens','Albumin' ]]

X_test = sm.add_constant(X1_test)
#X_test = X1_test.copy() # Independent variables without constant. I have doubts in the use of vaiables without adding a constant
y_test = ip_test_data['Dataset']
reg_log = sm.Logit(y_train,X_train)
result_log = reg_log.fit()

result_log.summary2()

# See the values against each feature in 'P>|z|' column. The value less than 0.05 means the feature is insignificant in the model.
# Surprisingly, gender is insignificant in this model. 
# We saw a strong correlation between Total_Bilirubin and Direct_Bilirubin, and we could drop one of those features.
# But in this model both Total_Bilirubin and Direct_Bilirubin are insignificant(P > 0.05). So we must discard both features.

# Confusion matrix using the train dataset
result_log.predict()
result_log.pred_table()
# Print X_test and X_train and see if the order of features are same in both datasets
X_test.head()
X_train.head()
logit_predicted = result_log.predict(X_test)
# Uncomment below line, if you wanted to see the predicted values
# logit_predicted
# Confusion Matrix of predicted values. if you wanted to use sklearn's 'confusion_matrix', 
# you should convert float values in 'logit_predicted' to 0s and 1s.

# Here I use a custom confusion matrix code 
bins = np.array([0,0.5,1])
cm_log = np.histogram2d(y_test, logit_predicted, bins = bins)[0]
logit_accuracy = (cm_log[0,0] + cm_log[1,1])/cm_log.sum()

print('Confusion Matrix (Logit): \n', cm_log)
print('---------------------------------------------')
print('Accuracy (Logit): \n', round(logit_accuracy*100,2), '%')
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train,y_train)

log_predicted = logreg.predict(X_test)
print('Confusion Matrix (Logistic Reg): \n', confusion_matrix(y_test,log_predicted))
print('---------------------------------------------')
print('Accuracy (Logistict Reg): \n', round(accuracy_score(y_test, log_predicted)*100,2), '%')
from sklearn.naive_bayes import GaussianNB

gaussian = GaussianNB()
result_gauss = gaussian.fit(X_train,y_train)

gauss_predicted = gaussian.predict(X_test)
# Uncomment below line, if you wanted to see the predicted values
#gauss_predicted
print('Confusion Matrix (Gaussian): \n', confusion_matrix(y_test,gauss_predicted))
print('---------------------------------------------')
print('Accuracy (Gaussian): \n', round(accuracy_score(y_test, gauss_predicted)*100,2), '%')
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators = 100)
random_forest.fit(X_train, y_train)

rf_predicted = random_forest.predict(X_test)
print('Confusion Matrix (Random Forest): \n', confusion_matrix(y_test,rf_predicted))
print('---------------------------------------------')
print('Accuracy (Random Forest): \n', round(accuracy_score(y_test, rf_predicted)*100,2), '%')

