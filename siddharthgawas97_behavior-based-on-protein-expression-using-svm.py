import pandas as pd
import numpy as  np
#reading data from file
data = pd.read_csv('../input/Data_Cortex_Nuclear.csv')
#Display all the columns
data.columns
#Drop unwanted Columns
data = data.drop(['MouseID','Treatment', 'Genotype', 'class'],axis=1)
#Drop all the columns which have more than or equal to 10 missing values
temp_data = (data.isnull().sum()  < 10)
columns_with_missing_lte_10 =[]
for i in range(temp_data.shape[0]):
    if temp_data.iloc[i] == True:
        columns_with_missing_lte_10.append(temp_data.index[i])
data = data[columns_with_missing_lte_10]
data.columns
#Replace Blank values with NaN
columns = data.columns
X_data = data[columns[:-1]]
y_data = data[columns[-1]]
X_data.replace('',np.NaN,inplace=True)
#Fill missing values with mean
from sklearn.preprocessing import Imputer
imputer = Imputer()
imputer.fit(X_data)
X_data = pd.DataFrame(columns=X_data.columns,data=imputer.transform(X_data))
X_data.isnull().sum()
#Train and test 
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X_data, y_data, test_size=0.2)
from sklearn.svm import SVC
from sklearn.grid_search import RandomizedSearchCV
import scipy
clf  =  SVC()
#Randomized Grid Search to optimize hyperparameters
s = RandomizedSearchCV(clf,param_distributions={'C': scipy.stats.expon(scale=100), 'gamma': scipy.stats.expon(scale=.1),
  'kernel': ['rbf','linear']},)
#Train model
s.fit(X_train,y_train)
#Test Score
print("Train score ",s.score(X_train,y_train))
#Train Score
print("Test score ",s.score(X_test,y_test))
#Best hyperparameters
s.best_params_
#Predict first 10 values
s.predict(X_test[:10])
#Actual first 10 classes
y_test[:10]
