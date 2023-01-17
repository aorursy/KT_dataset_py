#Library Imports

import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor 
import statsmodels.api as sm 
from sklearn.metrics import r2_score
from pylab import *
#Upload Data 
url ='../input/quality-prediction-in-a-mining-process/MiningProcess_Flotation_Plant_Database.csv'
mainData = pd.read_csv(url,decimal=',')

 
print('Shape of Main Data = ', mainData.shape)
mainData = mainData.dropna()
print('Shape of Main Data after drop null values = ', mainData.shape)

mainData.head(10)
mainData.columns
mainData.dtypes
mainData.describe()
#change date columns as datetime
mainData['date'] = pd.to_datetime(mainData['date'])
mainData['date'] = pd.to_datetime(mainData['date'])
#grouping the data according to the hours and get their average values. 
cycle_data = mainData.groupby(pd.Grouper(key='date',freq='H')).mean()
# cycle_data.insert(0,'Date',cycle_data.index)
cycle_data.reset_index(inplace = True)

#some rows have 'null' values because of timing. We need to drop them 
print('Shape of Cycle Data = ', cycle_data.shape)
cycle_data = cycle_data.dropna()
print('Shape of Cycle Data after drop null values = ', cycle_data.shape)
mainData.dtypes
data = cycle_data.copy()
data.head()
#seperate data as flotation_conditions and concentrates
flotation_conditions = data.iloc[:,1:22]
concentrates = data.iloc[:,22:]
silica_concentrate = concentrates.iloc[:,1].values
iron_concentrate = concentrates.iloc[:,0].values

print('Shape of flotation_conditions = ', flotation_conditions.shape)
'''P-value'''

b_zero = np.append(arr = np.ones((len(data.iloc[:,1].values),1)).astype(int),
                                                                     values = flotation_conditions,
                                                                     axis = 1)

silica_concentrate = concentrates.iloc[:,1].values
iron_concentrate = concentrates.iloc[:,0].values

model_iron = sm.OLS(endog = iron_concentrate,exog = b_zero).fit()
print(model_iron.summary())

model_silica = sm.OLS(endog = silica_concentrate,exog = b_zero).fit()
print(model_silica.summary())
#extract p-value from regression results
pValue_fe = model_iron.pvalues
pValue_si = model_silica.pvalues
pValue_fe_list = []
pValue_si_list = []
for i in range(len(pValue_fe)):
    if i > 0:
        pValue_fe_list.append(pValue_fe[i])
        pValue_si_list.append(pValue_si[i])
width_in_inches = 15
height_in_inches = 13
dots_per_inch = 60

plt.figure(
    figsize=(width_in_inches, height_in_inches),
    dpi=dots_per_inch)

t = flotation_conditions.columns
fe = pValue_fe_list
si = pValue_si_list
plot(t, fe, label = 'Fe', color = 'blue',marker='o')
plot(t, si, label = 'Si',color = 'red', marker='x')
plt.xticks(rotation=90)
plt.legend(loc="upper right",prop={"size":25})
ylabel('P Value',fontsize = 20)
title('P Value of Si adn Fe Cocentrates',fontsize = 25)
grid(True)
plt.figure(figsize=(60,3))
plt.savefig("p_values.png")
show()

#Confidence Intervals

m_fe = iron_concentrate.mean()
se_fe = iron_concentrate.std()/math.sqrt(len(iron_concentrate))
ci_fe = [m_fe - se_fe*1.96, m_fe + se_fe*1.96]


m_si = silica_concentrate.mean()
se_si = silica_concentrate.std()/math.sqrt(len(silica_concentrate))
ci_si = [m_si - se_si*1.96, m_si + se_si*1.96]

print  ('Confidence interval of Fe Concentrate:' ,ci_fe)
print  ('Confidence interval of Silica Concentrate:' ,ci_si)
width_in_inches = 15
height_in_inches = 13
dots_per_inch = 60

plt.figure(
    figsize=(width_in_inches, height_in_inches),
    dpi=dots_per_inch)

x_axis = data['% Iron Concentrate']
y_axis = data['% Iron Feed']
plot(x_axis, y_axis,marker='o')
plt.xticks(rotation=90)
plt.legend(loc="upper right",prop={"size":25})
xlabel('% Iron Concentrate',fontsize = 20)
ylabel('% Iron Feed',fontsize = 20)
title('% Iron Concentrate vs % Iron Feed',fontsize = 25)
grid(True)
plt.figure(figsize=(60,3))
show()
#train test split for regression training and testing 
x_train, x_test, y_train, y_test = train_test_split(flotation_conditions,
                                                    concentrates,
                                                    test_size = 0.15,
                                                    random_state = 0)
'''MLR'''
regressor_mlr = LinearRegression()
regressor_mlr.fit(x_train,y_train) #x trainden y traini öğren 

y_pred_mlr = regressor_mlr.predict(x_test)
print('R2 Score of Multi Linear Regression',r2_score(y_test,y_pred_mlr))
'''RANDOM FOREST with Train-Test Split'''

regressor_randForest = RandomForestRegressor(random_state = 0, n_estimators = 100)  
regressor_randForest.fit(x_train,y_train) 
y_pred_rf = regressor_randForest.predict(x_test)

regressor_randForest2 = RandomForestRegressor(random_state = 0, n_estimators = 100)  
regressor_randForest2.fit(flotation_conditions,concentrates)
y_pred_rf2 = regressor_randForest.predict(flotation_conditions)
 
print('R2 Score of Random Forest Regression with Train-Test Split',r2_score(y_test,y_pred_rf))
print('R2 Score of Random Forest Regression with Whole Data',r2_score(concentrates,y_pred_rf2))
'''Iron Random Forest Model'''
regressor_Fe = RandomForestRegressor(random_state = 0, n_estimators = 100)  
regressor_Fe.fit(flotation_conditions,iron_concentrate) 
y_pred_Fe = regressor_Fe.predict(flotation_conditions)

print('R2 Score of Random Forest Regression with Only Iron',r2_score(iron_concentrate,y_pred_Fe))
'''Silica Random Forest Model'''
regressor_Si = RandomForestRegressor(random_state = 0, n_estimators = 100)  
regressor_Si.fit(flotation_conditions,silica_concentrate) 
y_pred_Si = regressor_Si.predict(flotation_conditions)

print('R2 Score of Random Forest Regression with Only Silica',r2_score(silica_concentrate,y_pred_Si))
#feel free change values and see results
predictions = {'% Iron Feed':50.5,
          '% Silica Feed':13.3,
          'Starch Flow':3500.0,
          'Amina Flow':580.0,
          'Ore Pulp Flow':400.0,
          'Ore Pulp pH':10.11,
          'Ore Pulp Density':1.69,
          'Flotation Column 01 Air Flow':250.0,
          'Flotation Column 02 Air Flow':150.0,
          'Flotation Column 03 Air Flow':270.0,
          'Flotation Column 04 Air Flow':190.0,
          'Flotation Column 05 Air Flow':230.0,
          'Flotation Column 06 Air Flow':200.0,
          'Flotation Column 07 Air Flow':240.0,
          'Flotation Column 01 Level':480,
          'Flotation Column 02 Level':210.0,
          'Flotation Column 03 Level':550.0,
          'Flotation Column 04 Level':620.0,
          'Flotation Column 05 Level':610.0,
          'Flotation Column 06 Level':615.0,
          'Flotation Column 07 Level':616.0,
          }
#to see predictions, run this code
predict_values = []
for value in predictions.values(): 
    predict_values.append(value)

predict_Fe = regressor_Fe.predict([predict_values])
predict_Si = regressor_Si.predict([predict_values])

print('Predicted Fe Concentrate =',predict_Fe,'%')
print('Predicted Silica Concentrate =',predict_Si,'%')
predict_from_means = flotation_conditions.describe().mean()
predict_Fe = regressor_Fe.predict([predict_from_means])
predict_Si = regressor_Si.predict([predict_from_means])

print('Predicted Fe Concentrate from Conditions mean  =',predict_Fe,'%')
print('Predicted Silica Concentrate from Conditions mean =',predict_Si,'%')