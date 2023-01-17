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
#Load the data from .csv into a data-frame
import pandas as pd
initial_df = pd.read_csv('/kaggle/input/datasetucimlairquality/AirQualityUCI.csv')
initial_df.head()
initial_df.shape
#Print the null values of Date column to observe how the data looks.
initial_df[initial_df['Date'].isnull()]
#Observing the data, each data has a data point per hour
#However all the column of Date NaN also have a NaN
#Hence it makes sense to drop all NaN rows which amount to 114 data points out of 9471 data points = 0.11%
initial_df.dropna(subset=['Date'],inplace=True)
initial_df.info()
initial_df.shape
#Column 15 and 16 has no valid data, so need to dop them as well.
initial_df.drop(initial_df.filter(regex="Unnamed"),axis=1, inplace=True)
initial_df.describe()
initial_df.shape
#Copy into a new Pandas Data-frame for clarity purposes only
cleaned_up_df = initial_df
initial_df.shape
cleaned_up_df.shape
#Remove date and time for plotting purposes
temp_df = cleaned_up_df.copy()
temp_df.drop(columns=['Date','Time']) #This line causes error if executed mutiple times as 
#'Date' and 'Time' no longer esist after
#first run
temp_df.plot()
cleaned_up_df.shape
#From above graph it is clear that -200 is an invalid value which needs to be got rid of.
#The idea is to drop if -200 is found in more than two columns
#An attempt was made to drop all -200 values, but then the number of data points drop to from 9357 to 827, hence the logic
#of dropping rows where two columns have -200.
#filterinfDataframe = dfObj[(dfObj['Sale'] > 30) & (dfObj['Sale'] < 33) ]
sub_set_df = cleaned_up_df[(cleaned_up_df['CO_GT'] != -200) & (cleaned_up_df['PT08_S1_CO'] != -200)]
sub_set_df.shape
#Count the remaining -200s in the data-set to decide how to proceed
print("In column CO(GT),  % of invalid values", (sub_set_df['CO_GT'] == -200).sum(axis=0)/len(sub_set_df)*100)
print("In column PT08.S1(CO),  % of invalid values", (sub_set_df['PT08_S1_CO'] == -200).sum(axis=0)/len(sub_set_df)*100)
print("In column NMHC(GT),  % of invalid values", (sub_set_df['NMHC_GT'] == -200).sum(axis=0)/len(sub_set_df)*100)
print("In column C6H6(GT) % of invalid values", (sub_set_df['C6H6_GT'] == -200).sum(axis=0)/len(sub_set_df)*100)
print("In column PT08.S2(NMHC) % of invalid values", (sub_set_df['PT08_S2_NMHC'] == -200).sum(axis=0)/len(sub_set_df)*100)
print("In column NOx(GT) % of invalid values", (sub_set_df['Nox_GT'] == -200).sum(axis=0)/len(sub_set_df)*100)
print("In column PT08.S3(NOx) % of invalid values", (sub_set_df['PT08_S3_Nox'] == -200).sum(axis=0)/len(sub_set_df)*100)
print("In column NO2(GT) % of invalid values", (sub_set_df['NO2_GT'] == -200).sum(axis=0)/len(sub_set_df)*100)
print("In column PT08.S4(NO2) % of invalid values", (sub_set_df['PT08_S4_NO2'] == -200).sum(axis=0)/len(sub_set_df)*100)
print("In column PT08.S5(O3) % of invalid values", (sub_set_df['PT08_S5_O3'] == -200).sum(axis=0)/len(sub_set_df)*100)
print("In column T % of invalid values", (sub_set_df['T'] == -200).sum(axis=0)/len(sub_set_df)*100)
print("In column RH % of invalid values", (sub_set_df['RH'] == -200).sum(axis=0)/len(sub_set_df)*100)
print("In column AH % of invalid values", (sub_set_df['AH'] == -200).sum(axis=0)/len(sub_set_df)*100)
#Drop the column NMHC(GT)
sub_set_with_minimal_minus_200 = sub_set_df.drop(['NMHC_GT'],axis=1)
sub_set_with_minimal_minus_200.shape
sub_set_with_minimal_minus_200.plot()
#Common -200 values in NOx(GT) and NO2(GT) can be removed as well.
fully_clean_df = sub_set_with_minimal_minus_200[(sub_set_with_minimal_minus_200['Nox_GT'] != -200) & (sub_set_with_minimal_minus_200['NO2_GT'] != -200)]
fully_clean_df.plot()
fully_clean_df .shape
#Drop all the GT values for ML model building. 
#Note C6H6 is kept as is, as this there is no sensor value for the same.
print(fully_clean_df.shape)
ml_data_set = fully_clean_df.drop(['CO_GT','Nox_GT','NO2_GT'],axis=1)
print(ml_data_set.shape)
#Split the date and time so that ML models can use them to learn.
#For e.g. months are needed to take into account seasonality in Italy
#Spring Mar-May
#Summer June-Aug
#Autumn Sep-Nov
#Winter Dec-Feb
#The format of date is M/DD/YYYY or M-DD-YY
#First is to make all date formats uniform
import pandas as pd
ml_data_set['Month'] = pd.to_datetime(ml_data_set['Date']).dt.month
ml_data_set['Year'] =  pd.to_datetime(ml_data_set['Date']).dt.year
#Drop the Date column as month which is needed is extracted.
ml_data_set.drop(['Date'],axis=1,inplace = True)

#Now extract the hour from Time column and drop the same.
ml_data_set['Hour'] = pd.to_datetime(ml_data_set['Time']).dt.hour
ml_data_set.drop(['Time'],axis=1,inplace = True)
#CO_level is categorical, converth this using one_hot encoding
one_hot_features = ['CO_level']
one_hot_encoded_training_predictors = pd.get_dummies(ml_data_set['CO_level'])
ml_data_set = pd.concat([ml_data_set, one_hot_encoded_training_predictors] ,axis=1)
ml_data_set.drop(['CO_level'],axis=1,inplace=True)
ml_data_set.describe()
#First ML model for this problem, going with Linear Regression to predict RH
ml_data_set.describe()
X = ml_data_set.drop(['RH'],axis = 1)
y = ml_data_set['RH']
X.describe()
X.info()

#Split the data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=1234,test_size=0.3)
print(X_train.head())
y_train
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
# Train the model
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train, y_train)

print(linreg.coef_)                                            # Coefficients for Logistic Regression
print(linreg.intercept_)
y_train.shape
#Predict using the linear regression model
y_pred = linreg.predict(X_test)
y_pred.shape
# calculate these metrics by hand!
from sklearn import metrics
import numpy as np
def typical_linear_model_performance(y_pred):
    print ('MAE:', metrics.mean_absolute_error(y_test, y_pred))
    print ('MSE:', metrics.mean_squared_error(y_test, y_pred))
    print ('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 
#Find the R2 value
from sklearn.model_selection import cross_val_score
def get_cross_value_score(model):
    scores = cross_val_score(model, X_train, y_train,cv=5,scoring='r2')
    print('CV Mean: ', np.mean(scores))
    print('STD: ', np.std(scores))
    print('\n')
get_cross_value_score(linreg)
typical_linear_model_performance(y_pred)
#Add an eli5 to understand how the model behaves
import eli5
eli5.show_weights(linreg,feature_names = X_test.columns.tolist())
#Plot a SHAP as well
import shap
ind = 4
explainer = shap.LinearExplainer(linreg,data=X_test.values)
shap_values = explainer.shap_values(X_test)
shap.initjs()
shap.force_plot(
    explainer.expected_value, shap_values[ind,:], X_test.iloc[ind,:],
    feature_names=X_test.columns.tolist()
)
shap.summary_plot(shap_values,X_test)
#Build KNN model for the data-set for fun and profit
from sklearn.neighbors import KNeighborsRegressor

clf_knn = KNeighborsRegressor(n_neighbors=10)
clf_knn = clf_knn.fit(X_train,y_train)

y_pred = clf_knn.predict(X_test)
#Measure up how KNN is doing
get_cross_value_score(clf_knn)
typical_linear_model_performance(y_pred)