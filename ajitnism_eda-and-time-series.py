import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
data = pd.read_csv("../input/sales-forecasting/train.csv")
data.head()
data.shape
data = data.drop_duplicates()
data.shape
data.info()
data.drop('Row ID',axis=1,inplace=True)
data.describe(include='all')
data.columns
data.hist(['Sales'])
data['Sales']
data.nunique()
def plotbarcharts(dataset,columns):
    %matplotlib inline
    fig,subplot = plt.subplots(nrows=1,ncols=len(columns),figsize=(18,5))
    fig.suptitle('Bar Chart for' + str(columns))
    for columnname,plotnumber in zip(columns,range(len(columns))):
        dataset.groupby(columnname).size().plot(kind='bar',ax=subplot[plotnumber])
columnsList1 = ['Ship Mode','Region']
columnsList2 = ['Region','Category','Sub-Category']
data[columnsList1].head()
data[columnsList2].head()
plotBarChart(data,columnsList1)
plotBarChart(data,columnsList2)
data.groupby(['State']).size().plot(kind='bar',figsize=(18,8))
data.isna().sum()
data['Order Date'] = pd.to_datetime(data['Order Date'], format='%d/%m/%Y')

# Function to get month from a date
def Function_get_month(inpDate):
    return(inpDate.month)

# Function to get Year from a date
def Function_get_year(inpDate):
    return(inpDate.year)


# Creating new columns
data['Month']=data['Order Date'].apply(Function_get_month)
data['Year']=data['Order Date'].apply(Function_get_year)

data.head()
data['Year'].unique()
data['Month'].unique()
data.groupby(['Month']).size().plot(kind='bar')
data.groupby(['Year']).size().plot(kind='bar')
data.set_index("Order Date", inplace = True)
data['Sales'].plot()

# Aggregating the sales quantity for each month for all categories
pd.crosstab(columns=data['Month'],
            index=data['Year'],
            values=data['Sales'],
            aggfunc='sum')
import matplotlib.pyplot as plt
SalesQuantitiy=pd.crosstab(columns=data['Year'],
            index=data['Month'],
            values=data['Sales'],
            aggfunc='sum').melt()['value']

MonthNames=['Jan','Feb','Mar','Apr','May', 'Jun', 'Jul', 'Aug', 'Sep','Oct','Nov','Dec']*4

# Plotting the sales
%matplotlib inline
SalesQuantitiy.plot(kind='line', figsize=(16,5), title='Total Sales Quantity per month')
# Setting the x-axis labels
plotLabels=plt.xticks(np.arange(0,48,1),MonthNames, rotation=30)
SalesQuantitiy.values
from statsmodels.tsa.seasonal import seasonal_decompose
series = SalesQuantitiy.values
result = seasonal_decompose(series, model='additive', freq=12)
#print(result.trend)
#print(result.seasonal)
#print(result.resid)
#print(result.observed)
result.plot()
CurrentFig=plt.gcf()
CurrentFig.set_size_inches(11,8)
plt.show()
# Importing the algorithm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')


# Train the model on the full dataset 
SarimaxModel = model = SARIMAX(SalesQuantitiy,  
                        order = (5, 1, 10),  
                        seasonal_order =(1, 0, 0, 12))
SalesModel = SarimaxModel.fit()
  
# Forecast for the next 6 months
forecast = SalesModel.predict(start = 0,
                          end = (len(SalesQuantitiy)) + 6,
                          typ = 'levels').rename('Forecast')
print("Next Six Month Forecast:",forecast[-6:])

# Plot the forecast values
SalesQuantitiy.plot(figsize = (18, 5), legend = True, title='Time Series Sales Forecasts')
forecast.plot(legend = True, figsize=(18,5))

# Measuring the accuracy of the model
MAPE=np.mean(abs(SalesQuantitiy-forecast)/SalesQuantitiy)*100
print('#### Accuracy of model:', round(100-MAPE,2), '####')

# Printing month names in X-Axis
MonthNames=MonthNames+MonthNames[0:6]
plotLabels=plt.xticks(np.arange(0,54,1),MonthNames, rotation=30)
data.columns
data.groupby(['Ship Mode']).sum()['Sales'].plot(kind='bar')
data.groupby(['Ship Mode']).sum()['Sales']
# Filtering only Technology data
StandardClassSalesData=data[data['Ship Mode']=='Standard Class']
# Aggregating the sales quantity for each month for all categories
pd.crosstab(columns=StandardClassSalesData['Month'],
            index=StandardClassSalesData['Year'],
            values=StandardClassSalesData['Sales'],
            aggfunc='sum')
import matplotlib.pyplot as plt
SalesQuantity=pd.crosstab(columns=StandardClassSalesData['Year'],
            index=StandardClassSalesData['Month'],
            values=StandardClassSalesData['Sales'],
            aggfunc='sum').melt()['value']

MonthNames=['Jan','Feb','Mar','Apr','May', 'Jun', 'Jul', 'Aug', 'Sep','Oct','Nov','Dec']*4

# Plotting the sales
SalesQuantity.plot(kind='line', figsize=(16,5), title='Total Sales Quantity per month for Standard Class')
# Setting the x-axis labels
plotLabels=plt.xticks(np.arange(0,48,1),MonthNames, rotation=30)
# Importing the algorithm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')


# Train the model on the full dataset 
SarimaxModel = model = SARIMAX(SalesQuantity, 
                        order = (6, 0, 1),  
                        seasonal_order =(1, 0, 0, 12))

SalesModel = SarimaxModel.fit()
  
# Forecast for the next 6 months
forecast = SalesModel.predict(start = 0,
                          end = (len(SalesQuantity)) + 6,
                          typ = 'levels').rename('Forecast')

# Plot the forecast values
SalesQuantity.plot(figsize = (20, 5), legend = True, title='Time Series Sales Forecasts for Standard Class')
forecast.plot(legend = True, figsize=(20,5))

print("Next Six Month Forecast:",forecast[-6:])

# Measuring the accuracy of the model
MAPE=np.mean(abs(SalesQuantity-forecast)/SalesQuantity)*100
print('#### Accuracy of model:', round(100-MAPE,2), '####')

# Printing month names in X-Axis
MonthNames=MonthNames+MonthNames[0:6]
plotLabels=plt.xticks(np.arange(0,54,1),MonthNames, rotation=30)
data.groupby(['State']).sum()['Sales'].plot(kind='bar', figsize=(18,5))
data.groupby(['Segment']).sum()['Sales'].plot(kind='bar')
data.groupby(['Category']).sum()['Sales'].plot(kind='bar')
data.groupby(['Sub-Category']).sum()['Sales'].plot(kind='bar')
data.groupby(['Sub-Category']).sum()['Sales']
data.groupby(['Category']).sum()['Sales']
# Filtering only Technology data
TechnologySalesData=data[data['Category']=='Technology']
# Aggregating the sales quantity for each month for all categories
pd.crosstab(columns=TechnologySalesData['Month'],
            index=TechnologySalesData['Year'],
            values=TechnologySalesData['Sales'],
            aggfunc='sum')
import matplotlib.pyplot as plt
SalesQuantity=pd.crosstab(columns=TechnologySalesData['Year'],
            index=TechnologySalesData['Month'],
            values=TechnologySalesData['Sales'],
            aggfunc='sum').melt()['value']

MonthNames=['Jan','Feb','Mar','Apr','May', 'Jun', 'Jul', 'Aug', 'Sep','Oct','Nov','Dec']*4

# Plotting the sales
SalesQuantity.plot(kind='line', figsize=(16,5), title='Total Sales Quantity per month for Technology Category')
# Setting the x-axis labels
plotLabels=plt.xticks(np.arange(0,48,1),MonthNames, rotation=30)
# Importing the algorithm
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')


# Train the model on the full dataset 
SarimaxModel = model = SARIMAX(SalesQuantity, 
                        order = (6, 0, 1),  
                        seasonal_order =(1, 0, 0, 12))

SalesModel = SarimaxModel.fit()
  
# Forecast for the next 6 months
forecast = SalesModel.predict(start = 0,
                          end = (len(SalesQuantity)) + 6,
                          typ = 'levels').rename('Forecast')

# Plot the forecast values
SalesQuantity.plot(figsize = (20, 5), legend = True, title='Time Series Sales Forecasts for Technology Category')
forecast.plot(legend = True, figsize=(20,5))

print("Next Six Month Forecast:",forecast[-6:])

# Measuring the accuracy of the model
MAPE=np.mean(abs(SalesQuantity-forecast)/SalesQuantity)*100
print('#### Accuracy of model:', round(100-MAPE,2), '####')

# Printing month names in X-Axis
MonthNames=MonthNames+MonthNames[0:6]
plotLabels=plt.xticks(np.arange(0,54,1),MonthNames, rotation=30)
