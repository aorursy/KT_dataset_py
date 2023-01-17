import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm, skew #for some statistics
from scipy import stats #qqplot
import statsmodels.api as sm
from matplotlib import rcParams
import matplotlib.pyplot as plt
data=pd.read_csv('/kaggle/input/productdemandforecasting/Historical Product Demand.csv')
prod=pd.unique(data['Product_Code']).tolist()
cate=pd.unique(data['Product_Category']).tolist()
ware=pd.unique(data['Warehouse']).tolist()
data.isnull().sum()
data['Order_Demand'] = data['Order_Demand'].str.replace('(',"")
data['Order_Demand'] = data['Order_Demand'].str.replace(')',"")
data['Order_Demand'] = data['Order_Demand'].astype('int64')
print(len(prod),len(cate),len(ware))
data.head()
food=['Dairy_products','baked_products','sugar_baked','eggs','meat','poultry','fish','flour','seafood','legumes ','cooked_food','cooked_vegetables','cooked_meat','leftovers','other_fruits','other_vegetable','Citrus','Stone_fruit','Tropical','peas','berries','melons','green_vegetables','Cruciferous','acidic_fruits','Marrow','root','allium','Soy Products','fresh_drinks','processed_food','desserts','bevarages']
food.sort()
for i in range(data.shape[1]):
    data=data.replace(to_replace=cate[i],value=food[i])
order=[]
for i in range(data.shape[0]):
    if data.iloc[i,0]=='Product_0979' and data.iloc[i,1]=='Whse_J' and data.iloc[i,2]=='Citrus':
        order.append([data.iloc[i,3],data.iloc[i,4]])
df=pd.DataFrame(data=order,columns=['Date','Order_Demand'])
df['Date']=pd.DatetimeIndex(df['Date'])
final=df.pivot_table(index='Date',aggfunc=sum)
rcParams['figure.figsize'] = 50,14
sns.barplot(x=pd.DatetimeIndex(df['Date']).year, y=df['Order_Demand'])
#np.random.uniform(low=1.3, high=9.2, size=(33,))
import seaborn as sns
rcParams['figure.figsize'] = 5,3
sns.distplot(df['Order_Demand'], fit=norm)
#Get the QQ-plot
fig = plt.figure()
res = stats.probplot(df['Order_Demand'], plot=plt)
plt.show()
sns.lineplot(x=df['Date'], y=df['Order_Demand'])
y =final.resample('M').sum()
y.index.freq = "M"
final= pd.DataFrame(y["Order_Demand"]) 
final.head()
! pip install pmdarima
from pmdarima import auto_arima
model= auto_arima(final['Order_Demand'], start_p=1, start_q=1,
                          max_p=5, max_q=5, m=12,
                          start_P=0, seasonal=True,
                          d=None, D=1, trace=True,
                          error_action='ignore',   # we don't want to know if an order does not work
                          suppress_warnings=True,  # we don't want convergence warnings
                          stepwise=True)           # set to stepwise

model.summary()
dtrain=final['Order_Demand'].iloc[:45]
dtest=final['Order_Demand'].iloc[45:]
from statsmodels.tsa.statespace.sarimax import SARIMAX
mod= SARIMAX(dtrain,order=(2,0,2),seasonal_order=(2,1,0,12))
result=mod.fit()
result.summary()
predictions = result.predict(start=len(dtrain), end=len(dtrain)+len(dtest)-1, dynamic=False, typ='levels').rename('predicted')
ax=dtest.plot(legend=True,figsize=(12,6))
predictions.plot(legend=True)
from sklearn.metrics import mean_squared_error

error = np.sqrt(mean_squared_error(dtest, predictions))
print(f'SARIMA(2,0,2)(2,1,0,12) RMSE Error: {error:11.10}')
print('Std of Test data:                  ', final['Order_Demand'].std())
model = SARIMAX(final['Order_Demand'],order=(2,0,2),seasonal_order=(2,1,0,12))
results = model.fit()
fcast = results.predict(len(final['Order_Demand']),len(final['Order_Demand'])+3,typ='levels').rename('next_3_months_predicted')
ax =final['Order_Demand'].plot(legend=True,figsize=(12,6))
fcast.plot(legend=True)
ax.autoscale(axis='x',tight=True)
'''temperature=[7, 2, 6, 2, 6, 7, 3, 1, 6, 6, 4, 3, 7, 6, 4, 4, 5, 7, 3, 7, 5, 7,5, 2, 5, 4, 1, 3, 3, 3, 4, 2, 3]
temp=pd.DataFrame(data=data['Product_Category'],columns='Temperature')
for i in range(data.shape[1]):
    temp=temp.replace(to_replace=food[i],value=temperature[i])
data=data.join(temp)'''
#output is the graph in grey area By plotting the x axis(date) against y axis(demand) we can get the predicted demand