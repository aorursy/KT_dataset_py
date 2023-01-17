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
!pip install restcountries

!pip install pygam
#Imports all required packages

from pygam import LinearGAM, s, f

from pygam import PoissonGAM, s, te

import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

import pandas as pd

from sklearn import linear_model

from sklearn.linear_model import Lasso

import numpy as np

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_log_error

from sklearn.metrics import log_loss

#from restcountries import RestCountryApiV2 as rapi
#Read Data

train_data=pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")

test_data=pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv")

#I have read above data in local added population for all 163 countried and uplodaed same data here . As commented in above from import packages restcountries is not working 

train_data=pd.read_csv("../input/coviddatawithpopulation/train_data_with_population.csv")

test_data=pd.read_csv("../input/coviddatawithpopulation/test_data_with_population.csv")

submission_data=pd.read_csv("../input/covid19-global-forecasting-week-1/submission.csv")
#EDA

train_data.groupby(['Country','ConfirmedCases','Fatalities']).sum()
#Selecting attributes id,country,confirmed cases, fatalities,week,day,month and year

train_data_processed=train_data.iloc[:,[0,2,6,7,9,10,11,12,13]]

test_data_processed=test_data.iloc[:,[0,2,6,7,8,9,10,11]]
#Performing frequency enconding on country

fe_train=train_data_processed.groupby('Country').size()/len(train_data_processed)

fe_test=train_data_processed.groupby('Country').size()/len(train_data_processed)

train_data_processed.loc[:,'Country_Encode']=train_data_processed['Country'].map(fe_train)

train_data_processed["Country_Encode"]=((train_data_processed["Country_Encode"]-train_data_processed["Country_Encode"].min())/(train_data_processed["Country_Encode"].max()-train_data_processed["Country_Encode"].min()))*20

fe=test_data_processed.groupby('Country').size()/len(test_data_processed)

test_data_processed.loc[:,'Country_Encode']=test_data_processed['Country'].map(fe_test)

test_data_processed["Country_Encode"]=((test_data_processed["Country_Encode"]-test_data_processed["Country_Encode"].min())/(test_data_processed["Country_Encode"].max()-test_data_processed["Country_Encode"].min()))*20
#Below attributes normalizes all attributes between 0 to 1 other than Id column

def normalize(dataset):

    dataNorm=((dataset-dataset.min())/(dataset.max()-dataset.min()))*1

    dataNorm["Id"]=dataset["Id"]

    return dataNorm
#Below attributes selects only country encoded along with population and date fields

numeric_train_set=train_data_processed.iloc[:,[0,2,3,4,5,6,7,8,9]]

numeric_test_set=test_data_processed.iloc[:,[0,2,4,5,6,7,8]]
#Below attribute calls normalize function defined above to normalize all values between 0 and 1

data_train=normalize(numeric_train_set)

data_test=normalize(numeric_test_set)
#Below selection attributes to train model for fatalities

predictors=["Id","Month","WeekNumber","Day","Country_Encode","population"]

target_column=["Fatalities"]
#Divide data into train and test set

X_train_gam, X_test_gam, y_train_gam, y_test_gam = train_test_split(data_train[predictors].values, data_train[target_column].values, test_size=0.20, random_state=40)
#Below code trains the model with 1000 splines

gam = PoissonGAM(s(0, n_splines=1300) + te(3,5) +s(1)+s(2)).fit(X_train_gam, y_train_gam)
#Below code gives the view of model

plt.ion()

plt.rcParams['figure.figsize'] = (12, 8)

plt.ion()

plt.rcParams['figure.figsize'] = (12, 8)

XX = gam.generate_X_grid(term=1, meshgrid=True)

Z = gam.partial_dependence(term=1, X=XX, meshgrid=True)



ax = plt.axes(projection='3d')

ax.plot_surface(XX[0], XX[1], Z, cmap='viridis')
#Below code gives the summary of model, Summary has the rsquared value

gam.summary()
#Below code checks the accuracy on test data

predictions = gam.predict(X_test_gam)

print("Mean Square Log Error(MSLE) is "+str(np.sqrt(mean_squared_log_error( y_test_gam, predictions ))))

print("Mean Square Error(MSE) is "+str(np.sqrt(mean_squared_error(y_test_gam,predictions))))

print("RSquare is(R2) "+str(r2_score(y_test_gam, predictions)))

#We could see accuaracy of the model as 94 %

#As requested mean square log error is 0.006
#Below code plots actual vs predicted Scatter plot view

fig, ax = plt.subplots()

ax.scatter(y_test_gam, predictions)

ax.plot([y_test_gam.min(), y_test_gam.max()], [y_test_gam.min(), y_test_gam.max()], 'k--', lw=2)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

plt.show()
#Below code plots actual vs predicted Bar plot view



df = pd.DataFrame({'Actual': y_test_gam.flatten(), 'Predicted': predictions.flatten()})

df1 = df.head(25)

df1.plot(kind='bar',figsize=(16,10))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
#Below code predicts on test data

predictions = gam.predict(data_test[predictors].values)
submission_data['Fatalities']=predictions
#Below selection attributes to train model for Confirmed Cases

predictors_cc=["Id","Month","WeekNumber","Day","Country_Encode","population"]

target_column_cc=["ConfirmedCases"]
#Divide data into train and test set

X_train_gam_cc, X_test_gam_cc, y_train_gam_cc, y_test_gam_cc = train_test_split(data_train[predictors_cc].values, data_train[target_column_cc].values, test_size=0.20, random_state=40)
#Below code trains the model with 3500 splines

gam_confirmed_cases = PoissonGAM(s(0, n_splines=3500) + te(3,5) +s(1)+s(2)+s(4)).fit(X_train_gam_cc, y_train_gam_cc)
gam_confirmed_cases.summary()
#Below code checks the accuracy on test data on confirmed cases

predictions_cc = gam_confirmed_cases.predict(X_test_gam_cc)

print("Mean Square Log Error(MSLE) is "+str(np.sqrt(mean_squared_log_error( y_test_gam_cc, predictions_cc ))))

print("Mean Square Error(MSE) is "+str(np.sqrt(mean_squared_error(y_test_gam_cc,predictions_cc))))

print("R Square(R2) is "+str(r2_score(y_test_gam_cc, predictions_cc)))

#We could see accuaracy of the model as 98 %

#As requested mean square log error is 0.0033
#Beloc code predicts confirmed cases for future data

predictions_cc_sub = gam_confirmed_cases.predict(data_test[predictors].values)
submission_data['ConfirmedCases']=predictions_cc_sub

submission_data.describe()
#Below is the submission Data

submission_data=submission_data.round(4)

submission_data.to_csv('submission.csv',index=False)