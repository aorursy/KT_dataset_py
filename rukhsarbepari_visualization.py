import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
df
df.info()
df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'],format = '%Y-%m-%d %H:%M')  # Converting Datetime COlumn from object to datetime format
df['DATE'] = pd.to_datetime(df['DATE_TIME'],format = '%Y-%m-%d %H:%M').dt.date # Splitting DateTime into Date
df['DATE'] = pd.to_datetime(df['DATE']) # Convert Date into DateTime format
df.info()
df.nunique()
df['DATE']
df.columns
import matplotlib.pyplot as plt
#plt.plot(x,y)
plt.figure(figsize=(12,8))
plt.plot(df['DATE_TIME'],df['AMBIENT_TEMPERATURE'],label ='AMBIENT',c='cyan')
plt.plot(df['DATE_TIME'],df['MODULE_TEMPERATURE'],label ='MODULE',c='orange')
plt.plot(df['DATE_TIME'],df['MODULE_TEMPERATURE']-df['AMBIENT_TEMPERATURE'],label ='DIFFERENCE',c='k')
plt.grid()
plt.margins(0.05)
plt.legend()
import matplotlib.pyplot as plt
#plt.plot(x,y)
plt.figure(figsize=(20,10))
plt.plot(df['DATE_TIME'],df['AMBIENT_TEMPERATURE'].rolling(window=20).mean(),label ='AMBIENT',c='cyan')
plt.plot(df['DATE_TIME'],df['MODULE_TEMPERATURE'].rolling(window=20).mean(),label ='MODULE',c='orange')
plt.plot(df['DATE_TIME'],(df['MODULE_TEMPERATURE']-df['AMBIENT_TEMPERATURE']).rolling(window=20).mean(),label ='DIFFERENCE',c='k')
plt.grid()
plt.margins(0.05)
plt.legend()
plt.figure(figsize=(20,10))
plt.plot(df['AMBIENT_TEMPERATURE'],df['MODULE_TEMPERATURE'],marker = 'o',linestyle='')
plt.show()
plt.figure(figsize=(20,10))
plt.scatter(df['AMBIENT_TEMPERATURE'],df['MODULE_TEMPERATURE'],c=df['AMBIENT_TEMPERATURE'],alpha =0.5)
#alpha is for transparency (0 for full transparent, and 1 for solid color)
plt.show()
df['DATE']
dates = df['DATE'].unique()
dates
dates[0]
print(dates[0])
#irradiation =0 it means it is night time
  #           >0 it means it is day time
data = df[df['DATE'] == dates[0]][df['IRRADIATION']>0]
data
data = df[df['DATE'] == dates[0]][df['IRRADIATION']>0]
data
plt.plot(data['AMBIENT_TEMPERATURE'],data['MODULE_TEMPERATURE'])
data = df[df['DATE'] == dates[0]][df['IRRADIATION']>0]
data
plt.plot(data['AMBIENT_TEMPERATURE'],data['MODULE_TEMPERATURE'],marker = 'o',linestyle='',label = pd.to_datetime(dates[0],format = '%Y-%m-%d').date())
plt.legend()
plt.figure(figsize=(20,10))
for date in dates:
    data = df[df['DATE'] == date][df['IRRADIATION']>0]
    plt.plot(data['AMBIENT_TEMPERATURE'],data['MODULE_TEMPERATURE'],marker = 'o',linestyle='',label = pd.to_datetime(date,format = '%Y-%m-%d').date())
plt.legend()
df1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
df1
df1.info()
#15-05-2020 00:00 
df1['DATE_TIME'] = pd.to_datetime(df1['DATE_TIME'],format = '%d-%m-%Y %H:%M')  # Converting Datetime COlumn from object to datetime format
df1['DATE'] = pd.to_datetime(df1['DATE_TIME'],format = '%d-%m-%Y %H:%M').dt.date # Splitting DateTime into Date
df1['DATE'] = pd.to_datetime(df1['DATE']) # Convert Date into DateTime format
df1.info()
inv_lst= df1['SOURCE_KEY'].unique()
inv_lst

df1['SOURCE_KEY'].nunique()
df1.groupby('SOURCE_KEY')['TOTAL_YIELD'].max()





#Plot bar graph of sourcekey vs total yield for a particular inverter
plt.figure(figsize= (20,10))
plt.bar(inv_lst,df1.groupby('SOURCE_KEY')['TOTAL_YIELD'].max())
plt.xticks(rotation = 90)
plt.grid()
plt.show()


# Add x lable,y label ,legend ,play around with colours
df.info()

df
df1.info()
r_left= pd.merge(df1,df,on= 'DATE_TIME',how='left')
r_left
r_left.info()
r_left.isnull().sum()
# i want to know how many missing values are present in Ambient_temperature
r_left['AMBIENT_TEMPERATURE'].isnull().value_counts()
# display null data for evry column 
r_left.isnull().value_counts()

null_data = r_left[r_left.isnull().any(axis = 1)]   # you have 1 assignmnet program based on this
null_data
#plot a graph of Irradiation vs DC Power
plt.plot(r_left['IRRADIATION'],r_left['DC_POWER'],c='orange',marker ='o',linestyle='',alpha = 0.05,label ='DC POWER')
plt.legend()
plt.xlabel('irradiation')
plt.ylabel('dc power')
plt.show()
#1.plot the graph Module temperature vs DC Power
r_left.info()
dates = r_left['DATE_x'].unique()
dates
ndates = r_left['DATE_x'].nunique()
ndates
r_left[r_left['DATE_x']==dates[0]][r_left['IRRADIATION']>0.1]
data = r_left[r_left['DATE_x']== dates[1]][r_left['IRRADIATION']>0.1]
plt.plot(data['MODULE_TEMPERATURE'],data['DC_POWER'],marker ='o',linestyle='',label = pd.to_datetime(dates[1],format='%Y-%m-%d').date)
plt.legend()
r_left
r_left.info()
r_left['IRRADIATION']= r_left['IRRADIATION'].fillna(0)
r_left['AMBIENT_TEMPERATURE']= r_left['AMBIENT_TEMPERATURE'].fillna(0)
r_left['MODULE_TEMPERATURE']= r_left['MODULE_TEMPERATURE'].fillna(0)


r_left.info()
r_left.isnull().count()
# extract one column
X = r_left.iloc[:,12:13].values   #Irradiation
y =r_left.iloc[:,3].values        #DC POWER
X.ndim
y.ndim
plt.scatter(X,y)
X.shape
y.shape
# Linear Regression
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)
X_train.shape
X_test.shape
y_train.shape

y_test.shape
from sklearn.linear_model import LinearRegression
lin_reg =LinearRegression()
lin_reg.fit(X_train,y_train)
y_pred =lin_reg.predict(X_test)
y_pred
y_test
#plot input test vs output test
#plot input test vs predicted test

plt.scatter(X_test,y_test,color ='gray')
plt.scatter(X_test,y_pred,color ='red')
plt.show()
lin_reg.coef_   #slope  m
lin_reg.intercept_  #y intercept
lin_reg.predict([[0.9]])
 # y = mx+c
13223.52483583*0.9+75.70664797947302 
