import pandas as pd 
import numpy as np 
import matplotlib.pyplot as pypl 
import seaborn as sd
states_df = pd.read_csv('../input/daily-power-generation-in-india-20172020/State_Region_corrected.csv')
states_df.head()
power_df = pd.read_csv('../input/daily-power-generation-in-india-20172020/file.csv')
power_df.head()
power_df['Region'].unique()
states_df['Region'].unique()
states_df = states_df.replace('Northeastern','NorthEastern')
states_df.isnull().sum()
power_df.isnull().sum()
states_df.isna().sum()
power_df.isna().sum()
power_df.groupby(by=['Region']).sum()
power_df = power_df.fillna(0)
power_df.groupby(by=['Region'],as_index=False).sum()
power_df.groupby(by=['Region']).mean()
power_df.dtypes
state_colname = {'State / Union territory (UT)':'State',
                 'Area (km2)':'Area',
                 'National Share (%)':'National Share'}

power_colname = {'Thermal Generation Actual (in MU)':'Thermal Generation Actual',
                 'Thermal Generation Estimated (in MU)':'Thermal Generation Estimated',
                 'Nuclear Generation Actual (in MU)':'Nuclear Generation Actual',
                 'Nuclear Generation Estimated (in MU)':'Nuclear Generation Estimated',
                 'Hydro Generation Actual (in MU)':'Hydro Generation Actual',
                 'Hydro Generation Estimated (in MU)':'Hydro Generation Estimated'}

states_df = states_df.rename(columns=state_colname)
power_df = power_df.rename(columns=power_colname)
power_df['Thermal Generation Actual'] = power_df['Thermal Generation Actual'].str.replace(',','')
power_df['Thermal Generation Estimated'] = power_df['Thermal Generation Estimated'].str.replace(',','')
power_df['Thermal Generation Actual'] = power_df['Thermal Generation Actual'].astype(np.float64)
power_df['Thermal Generation Estimated'] = power_df['Thermal Generation Estimated'].astype(np.float64)
power_df['Date'] = pd.to_datetime(power_df['Date'])
states_df.groupby(by=['Region'])['State'].count()
states_df.groupby(by=['Region'])['State'].agg(['unique'])
mean_power = power_df.groupby(by='Date',as_index=False).mean()
mean_power
power_actcols = mean_power.loc[:,['Date','Thermal Generation Actual','Nuclear Generation Actual','Hydro Generation Actual']]
power_actcols['Total Generation Actual'] = power_actcols.iloc[:,1:3].mean(axis=1)
pypl.figure(figsize=(18,9))
#power_actcols.plot(x='Date',y='Total Actual Generation')

pypl.plot(power_actcols['Date'],power_actcols['Thermal Generation Actual'],label='Thermal Generation Actual')
pypl.plot(power_actcols['Date'],power_actcols['Nuclear Generation Actual'],label='Nuclear Generation Actual')
pypl.plot(power_actcols['Date'],power_actcols['Hydro Generation Actual'],label='Hydro Generation Actual')
pypl.plot(power_actcols['Date'],power_actcols['Total Generation Actual'],label='Total Generation Actual')

pypl.xlabel('Date')
pypl.ylabel('Power Generation mean (in MU)')

pypl.legend()
actual_power = ['Thermal Generation Actual','Nuclear Generation Actual','Hydro Generation Actual']
estimate_power = ['Thermal Generation Estimated','Nuclear Generation Estimated','Hydro Generation Estimated']

for idx,num in enumerate(actual_power): 
    #print(num)
    pypl.figure(figsize=(18,9))
    pypl.plot(mean_power['Date'],mean_power[actual_power[idx]],label=actual_power[idx])
    pypl.plot(mean_power['Date'],mean_power[estimate_power[idx]],label=estimate_power[idx])
    pypl.xlabel('Date')
    pypl.ylabel('Mean Power Generation (in MU)')
    pypl.legend()
    pypl.show()
#sd.lineplot(x='Date',y='Nuclear Generation Estimated',hue='Region',data=power_df)
mean_power_per_region = power_df.groupby(by=['Date','Region'],as_index=False).mean()
mean_power_per_region
pypl.figure(figsize=(18,9))
sd.lineplot(x='Date',y='Hydro Generation Actual',hue='Region',data=mean_power_per_region)
pypl.figure(figsize=(18,9))
sd.lineplot(x='Date',y='Nuclear Generation Actual',hue='Region',data=mean_power_per_region)
pypl.figure(figsize=(18,9))
sd.lineplot(x='Date',y='Thermal Generation Actual',hue='Region',data=mean_power_per_region)
mean_power_per_region.boxplot(by='Region',column=['Thermal Generation Actual'],figsize=(18,9))
mean_power_per_region.boxplot(by='Region',column=['Nuclear Generation Actual'],figsize=(18,9))
mean_power_per_region.boxplot(by='Region',column=['Hydro Generation Actual'],figsize=(18,9))
pypl.figure(figsize=(15,15))
sd.barplot(x='Region',y='Thermal Generation Actual',hue='Region',data=mean_power_per_region,dodge=False)
pypl.figure(figsize=(15,15))
sd.barplot(x='Region',y='Hydro Generation Actual',hue='Region',data=mean_power_per_region,dodge=False)
pypl.figure(figsize=(15,15))
sd.barplot(x='Region',y='Nuclear Generation Actual',hue='Region',data=mean_power_per_region,dodge=False)
power_share = states_df.groupby(by='Region',as_index=False).sum()
power_share
power_copy = power_df
power_copy['Year'] = power_copy['Date'].dt.year
power_copy.drop('Date',axis=1)
power_mean_year = power_copy.groupby(by=['Region'],as_index=False).mean()
power_mean_year = power_mean_year[['Region','Thermal Generation Actual','Nuclear Generation Actual','Hydro Generation Actual']]
power_mean_year
pdx = pd.merge(power_share,power_mean_year,how='outer')
pdx = pdx[pdx.Region != 'Central']
pdx['Thermal per km2'] = (pdx['Thermal Generation Actual']/pdx['Area'])
pdx['Nuclear per km2'] = (pdx['Nuclear Generation Actual']/pdx['Area'])
pdx['Hydro per km2'] = (pdx['Hydro Generation Actual']/pdx['Area'])
pdx
pdx_melt = pd.melt(pdx,id_vars=['Region'],value_vars=['Thermal per km2','Nuclear per km2','Hydro per km2'])
pdx_melt
pypl.figure(figsize=(18,9))
sd.barplot(x='Region',y='value',hue='variable',data=pdx_melt)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,Normalizer
from sklearn.linear_model import LinearRegression
mean_thermal = mean_power[['Hydro Generation Estimated','Hydro Generation Actual']]
mean_thermal
q1 = mean_thermal['Hydro Generation Actual'].quantile(0.25)
q3 = mean_thermal['Hydro Generation Actual'].quantile(0.75)
iqr = q3-q1

minimum = q1 - (1.5 * iqr)
maximum = q3 + (1.5 * iqr)

mean_thermal = mean_thermal.drop(mean_thermal[(mean_thermal['Hydro Generation Actual'] < minimum) | (mean_thermal['Hydro Generation Actual'] > maximum)].index)
q1 = mean_thermal['Hydro Generation Estimated'].quantile(0.25)
q3 = mean_thermal['Hydro Generation Estimated'].quantile(0.75)
iqr = q3-q1

minimum = q1 - (1.5 * iqr)
maximum = q3 + (1.5 * iqr)

mean_thermal = mean_thermal.drop(mean_thermal[(mean_thermal['Hydro Generation Estimated'] < minimum) | (mean_thermal['Hydro Generation Estimated'] > maximum)].index)
mean_thermal.boxplot()
x = mean_thermal[['Hydro Generation Estimated']].values
y = mean_thermal[['Hydro Generation Actual']].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
linreg = LinearRegression()
model = linreg.fit(x_train,y_train)
y_pred = model.predict(x_test)
pypl.figure(figsize=(18,9))

pypl.scatter(x_test,y_test)
pypl.plot(x_test,y_pred,color='r')
f'Training score : {linreg.score(x_train,y_train)}'
from sklearn.metrics import r2_score

score = r2_score(y_test,y_pred)

print(f'Test score : {score}')
f'Slope : {model.coef_}'
f'Intercept : {model.intercept_}'