#imprt Libraries

import pandas as pd 

import numpy as np

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

import statsmodels.api as sm

import matplotlib.pyplot as plt 

from statsmodels.sandbox.regression.predstd import wls_prediction_std
#import Data

data_train=pd.read_csv('../input/used-cars-price-prediction/train-data.csv')



#Data Look

data_train.head()
data_train=data_train.iloc[:,1:]
data_train.info()
#selection of important columns

data_train_c=data_train.iloc[:,[0,1,2,3,4,5,7,8,9,10]]
#firstly we need to Clean the data for This Dataframe

data_train_c.replace({' ':np.nan,'null':np.nan},inplace=True)

data_train_c=data_train_c.dropna()
# How To check Above Instruction

for i in data_train_c.columns:

     print(data_train_c[i].unique()) #unique value for each column
# Here you can see 'null bhp' occur so it's replace by np.nan and then drop this raw

data_train_c.replace({'null bhp':np.nan,'null km/kg':np.nan,'null CC':np.nan},inplace=True)

data_train_c=data_train_c.dropna()

data_train_c.head()
#remove extenation  km/kg ,bhp, CC from below column and convert string into float Value

l1=['Mileage','Engine','Power']

for i in l1:

    data_train_c[i]=data_train_c[i].str.split(" ").apply(lambda x:x[0])

    data_train_c[i] = data_train_c[i].str.strip()

    data_train_c[i] = data_train_c[i].astype(float)

#Data Look 

data_train_c.head()
data_train_c.Fuel_Type.unique()
#Fule_Type column is Categorical data so we need to convert into numeric order for Better Price prediction 

data_train_c.Fuel_Type.replace({'Petrol':1,'CNG':3,'Diesel':2,'LPG':4},inplace=True)
data_train_c.info()
data_train_c.Transmission.unique()
# Above thing apply in Transmission column

data_train_c.Transmission.replace({'Manual':1,'Automatic':2},inplace=True)
data_train_c.head()
data_train_c.info()
'''Before we apply first step  look our  data range  menas  we need to check range  of column  values 

so we will be visualize easily our plot with Price column value'''

for i in data_train_c.columns:

    print(i)

    print(data_train_c[i].describe())
''' Here 'Kilometers_Driven' in High value data so we need to convert high value to lower value

therfore we easily compare with  price value '''

data_train_c['KM_Drlog_form'] = np.log(data_train_c['Kilometers_Driven'])



data_train_c['KM_Drlog_form'].dropna()
data_train_c
data_train_c['Company']=data_train_c['Name'].str.split(" ").apply(lambda x: x[0])
data_train_c1=data_train_c.copy()
data_train_c1['Price']=data_train['Price']
data_train_c1
%matplotlib inline
#Scatter plot

g = sns.PairGrid(data_train_c1,



                 x_vars=data_train_c1.columns[1:11],



                 y_vars=['Price'])

g = g.map(plt.scatter)
data_train_c1.head()
#remove one outlier  because this outlier distract our model we can see this thing our above sctter plot



data_train_c1.loc[data_train_c1['Kilometers_Driven']!=data_train_c1['Kilometers_Driven'].max()]

data_train_c1['Kilometers_Driven'].dropna()

data_train_c1['KM_Drlog_form'] = np.log(data_train_c1['Kilometers_Driven'])

data_train_c1.head()
data_train_c1.columns
#Heatmap relationship all columns

sns.heatmap(data=data_train_c1.iloc[1:11].corr(), annot = True)
data_train_c1.corr()
#Fitting Model and Generate results

model=sm.OLS.from_formula("Price~Power+Year",data=data_train_c1)  # without Categoricl variable

# Power > Engine in term of correlationship with Price  so we do not need to include both into the  prediction (select: Power )

# { Mileage,KM_Drlog_form,'Year'}  related with each other so we can take one variable from them. (select: Year )

res=model.fit()

print(res.summary())

#Note: Here All selected Variable Based on the corrletion and R-squred Value 

# R-squared 0.680
print('Parameters: ', res.params)

print('R2: ', res.rsquared)
#Fitting Model and Generate results

model=sm.OLS.from_formula("Price~Power+Year+C(Company)",data=data_train_c1) # with Categorical variable

# Power > Engine in term of correlationship with Price  so we do not need to include both into the  prediction (select: Power )

# { Mileage,KM_Drlog_form,'Year'}  related with each other so we can take one variable from them. (select: Year )

res=model.fit()

print(res.summary())

#Note: Here All selected Variable Based on the corrletion and R-squred Value 

#
print('Parameters: ', res.params)

print('R2: ', res.rsquared)
# Here you can see R-squared 0.765 if we include the company(Categorical Variable ) then we get the two many Dimension if we don't 

# do that thing then we got R-squared got 0.680 
Pre_Price=res.predict()
prstd, iv_l, iv_u = wls_prediction_std(res)
#Common Variable for Comapare our Orginal Price vs Predicted Price 

Time= np.linspace(0, 1000, 100)
# Here We Can see Variation Upper bound and Lower Bound line

fig, ax = plt.subplots(figsize=(30,10 ))

ax.plot(Time, data_train_c1['Price'][:100], 'o', label="True")

ax.plot(Time, res.fittedvalues[:100], '*-', label="OLS")

ax.plot(Time, iv_u[:100], 'r--',label='Lower_Bound')

ax.plot(Time, iv_l[:100], 'g--',label='Upper_Bound')

ax.legend(loc='best')

plt.xlabel('Common_Variable')

plt.ylabel('Predict vs Orginal(Price)')
pp = sns.scatterplot(res.fittedvalues, res.resid)

pp.set_xlabel("Fitted values")

_ = pp.set_ylabel("Residuals")
data_test=pd.read_csv('../input/used-cars-price-prediction/test-data.csv')
data_test.head()
data_train.head()

# data Cleaning function

def Data_clean(data_train):



    data_train = data_train.iloc[:, 1:]



    # selection of important columns

    data_train_c = data_train.iloc[:, [0,1,2, 3, 4, 5, 7, 8, 9, 10]]



    # firstly we need to Clean the data for This Dataframe

    data_train_c.replace({' ': np.nan, 'null': np.nan}, inplace=True)

    data_train_c = data_train_c.dropna()



    # Here you can see 'null bhp' occur so it's replace by np.nan and then drop this raw

    data_train_c.replace(

        {'null bhp': np.nan, 'null km/kg': np.nan, 'null CC': np.nan}, inplace=True)

    data_train_c = data_train_c.dropna()



    # remove extenation  km/kg ,bhp, CC from below column and convert string into float Value

    l1 = ['Mileage', 'Engine', 'Power']

    for i in l1:

        data_train_c[i] = data_train_c[i].str.split(" ").apply(lambda x: x[0])

        data_train_c[i] = data_train_c[i].str.strip()

        data_train_c[i] = data_train_c[i].astype(float)



    # Fule_Type column is Categorical data so we need to convert into numeric order for Better Price prediction

    data_train_c.Fuel_Type.replace(

        {'Petrol': 1, 'CNG': 3, 'Diesel': 2, 'LPG': 4}, inplace=True)



    # Above thing apply in Transmission column

    data_train_c.Transmission.replace(

        {'Manual': 1, 'Automatic': 2}, inplace=True)



    ''' Here 'Kilometers_Driven' in High value data so we need to convert high value to lower value

    therfore we easily compare with  price value '''

    data_train_c['KM_Drlog_form'] = np.log(data_train_c['Kilometers_Driven'])

    

    data_train_c['Company']=data_train_c1['Name'].str.split(" ").apply(lambda x: x[0])

    



    return data_train_c
data_test=Data_clean(data_test)
data_test['Pre_Price']=res.predict(data_test) # res from the above data training function 
data_test