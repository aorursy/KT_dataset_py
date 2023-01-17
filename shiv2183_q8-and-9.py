# Importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib import pyplot
#Reading into dataframe df
df=pd.read_csv("../input/DC_Properties.csv")
df=df.iloc[:,1:len(df.columns)]
len(df.iloc[:,1])
#Finding percentage null in each column & 
#missing_value=df.isnull().sum()*100/len(df.iloc[:,1])
#print(missing_value)

#Removing rows for which price is null
unknown_data = df[df['PRICE'].isnull()==True]
print(len(unknown_data))
unknown_data.head()
#Removing rows for which price is known
df = df[df['PRICE'].isnull()==False]
df.head(2)
#Finding percentage null in each column
missing_value=df.isnull().sum()*100/len(df.iloc[:,1])
print(missing_value)
#Removing column names with more than 20% missing data
non_null_data=[]
for index in range(0, len(missing_value)):
        if(missing_value[index]<20):
            non_null_data.append(missing_value.index[index])

#Non-null dataframe
df=df[non_null_data]

#Finding percentage null in each column & 
missing_value=df.isnull().sum()*100/len(df.iloc[:,1])
print(missing_value)
#Removing rows with null values
df=df.dropna(subset=non_null_data)

#Finding percentage null in each column
missing_value=df.isnull().sum()*100/len(df.iloc[:,1])
print(missing_value)
#No. of duplicate data rows
print(len(df))
df.duplicated().sum()
#Remove duplicate rows
df=df.drop_duplicates()
len(df)
#Modifying data types
df['EYB']=pd.to_datetime(df['EYB'], format='%Y').dt.year
df['SALEDATE']=pd.to_datetime(df['SALEDATE']).dt.year
df.head(2)
df['AYB']=pd.to_datetime(np.int16(df['AYB']), format='%Y')
df['AYB']=df['AYB'].dt.year
#Drop Columns
df=df.drop(['GIS_LAST_MOD_DTTM'], axis=1)#Single Value Data
df=df.drop(['X','Y'], axis=1)#Repetitive Data
df=df.drop(['BLDG_NUM'],axis=1)#Single Value Data
df=df.drop(['SQUARE'],axis=1)#No trend in data
df=df.drop(['AYB'],axis=1)#No trend
df=df.drop(['CENSUS_TRACT'],axis=1)#Nominal data
df=df.drop(['LATITUDE','LONGITUDE'],axis=1)#Zipcode denotes the location
df=df.reset_index(drop=True)
df=df[df['FIREPLACES']<200]
#Function to find index no of outliers

def outlierindex(a):
    li=[]
    #find q1
    q1=np.percentile(a, 25)
    #find q2
    q3=np.percentile(a, 75)
    #find iqr
    iqr=q3-q1
    #Outlier range
    #lower fence=q1-1.5iqr
    lf=q1-1.5*iqr
    #upper fence=q3+1.5∗iqr
    uf=q3+1.5*iqr
    #index of outliers
    for i in range(0, len(a)):
        if((a[i]<=lf) or (a[i]>=uf)):
        #print(a[i])
            li.append(i)
    return (li)

#Plotting boxplot for numeric data
nndf=df.select_dtypes(exclude=['object'])
for i in range(0, len(nndf.columns)):
    nndf[nndf.columns[i]].plot.box()
plt.show()
df.info()
df.groupby('QUADRANT')['PRICE'].mean().index
df.groupby('HEAT')['PRICE'].mean().plot.bar()
encoding={'HEAT':{'Air Exchng':1, 'Air-Oil':2, 'Elec Base Brd':1, 'Electric Rad':2, 'Evp Cool':1,
       'Forced Air':2, 'Gravity Furnac':1, 'Hot Water Rad':1, 'Ht Pump':3, 'Ind Unit':1,
       'No Data':0, 'Wall Furnace':1, 'Warm Cool':2, 'Water Base Brd':1}}
df.replace(encoding, inplace=True)
encoding={'AC':{'0':1, 'N':1, 'Y':2}}
df.replace(encoding, inplace=True)
encoding={'QUALIFIED':{'Q':1, 'U':2}}
df.replace(encoding, inplace=True)
encoding={'USECODE':{11:1, 12:1, 13:1, 15:1, 16:1, 17:1, 19:2, 23:1, 24:1, 39:2, 116:1, 117:1}}
df.replace(encoding, inplace=True)
encoding={'SOURCE':{'Condominium':2, 'Residential':1}}
df.replace(encoding, inplace=True)
encoding={'ZIPCODE':{20001.0:3, 20002.0:2, 20003.0:3, 20004.0:2, 20005.0:5, 20006.0:1, 20007.0:4,
              20008.0:4, 20009.0:3, 20010.0:3, 20011.0:2, 20012.0:2, 20015.0:4, 20016.0:5,
              20017.0:2, 20018.0:2, 20019.0:1, 20020.0:1, 20024.0:2, 20032.0:1, 20036.0:2,
              20037.0:3, 20052.0:3, 20392.0:2}}
df.replace(encoding, inplace=True)
encoding={'ASSESSMENT_NBHD':{'16th Street Heights':2, 'American University':2, 'Anacostia':1,
       'Barry Farms':1, 'Berkley':4, 'Brentwood':1, 'Brightwood':1, 'Brookland':1,
       'Burleith':2, 'Capitol Hill':2, 'Central-tri 1':2, 'Central-tri 3':4,
       'Chevy Chase':2, 'Chillum':1, 'Cleveland Park':4, 'Colonial Village':2,
       'Columbia Heights':1, 'Congress Heights':1, 'Crestwood':2, 'Deanwood':1,
       'Eckington':1, 'Foggy Bottom':1, 'Forest Hills':2, 'Fort Dupont Park':1,
       'Fort Lincoln':1, 'Foxhall':2, 'Garfield':2, 'Georgetown':3, 'Glover Park':1,
       'Hawthorne':2, 'Hillcrest':1, 'Kalorama':2, 'Kent':3, 'Ledroit Park':2,
       'Lily Ponds':1, 'Marshall Heights':1, 'Massachusetts Avenue Heights':4,
       'Michigan Park':1, 'Mt. Pleasant':2, 'North Cleveland Park':2,
       'Observatory Circle':2, 'Old City 1':2, 'Old City 2':2, 'Palisades':2,
       'Petworth':1, 'Randle Heights':1, 'Riggs Park':1, 'Shepherd Heights':2,
       'Southwest Waterfront':1, 'Spring Valley':3, 'Takoma Park':1, 'Trinidad':1,
       'Wakefield':2, 'Wesley Heights':2, 'Woodley':3, 'Woodridge':1}}
df.replace(encoding, inplace=True)
encoding={'WARD':{'Ward 1':2, 'Ward 2':3, 'Ward 3':3, 'Ward 4':2, 'Ward 5':2, 'Ward 6':2, 'Ward 7':1,
       'Ward 8':1}}
df.replace(encoding, inplace=True)
encoding={'QUADRANT':{'NE':1, 'NW':2, 'SE':1, 'SW':1}}
df.replace(encoding, inplace=True)
df.head()
df.groupby('HEAT')['PRICE'].mean().plot.bar()
df['LOGPRICE']=np.log(df['PRICE'])
sns.heatmap(df.corr(method='spearman'))
# Model 1-['LOGPRICE-All other Variables in DF dataframe]
ndf=df
X=ndf.drop(['PRICE','LOGPRICE'], axis=1)
Y=ndf['LOGPRICE']
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8 , random_state=100)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,Y_train)
print(lm.intercept_)
print(lm.coef_)
coeff_df = pd.DataFrame(lm.coef_,X_test.columns,columns=['Coefficient'])
coeff_df
Y_pred = lm.predict(X_train)
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y_train, Y_pred)
r_squared = r2_score(Y_train, Y_pred)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)
nndf=pd.DataFrame(Y_train)
nndf['PRED']=Y_pred
nndf=nndf.reset_index(drop=True)
#Plotting the actual vs predicted values
plt.plot(nndf.index,nndf['LOGPRICE'])
plt.plot(nndf.index,nndf['PRED'], color='red')
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Actual (Blue) vs Predicted (Red)')
plt.show()
import statsmodels.api as sm

X_train_sm = X_train 
X_train_sm = sm.add_constant(X_train_sm)

lm_sm = sm.OLS(Y_train,X_train_sm).fit()

lm_sm.params
print(lm_sm.summary())
Y_pred = lm.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y_test, Y_pred)
r_squared = r2_score(Y_test, Y_pred)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)
nndf=pd.DataFrame(Y_test)
nndf['PRED']=Y_pred
nndf=nndf.reset_index(drop=True)
#Plotting the actual vs predicted values
plt.plot(nndf.index,nndf['LOGPRICE'])
plt.plot(nndf.index,nndf['PRED'], color='red')
plt.xlabel('Index')
plt.ylabel('Values-test')
plt.title('Actual (Blue) vs Predicted (Red)')
plt.show()
plt.scatter(nndf['LOGPRICE'],nndf['PRED'])
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
coeff_df.Coefficient.sort_values().plot.bar()
from sklearn.linear_model import Ridge


ridgeReg = Ridge(alpha=500, normalize=True)
ridgeReg.fit(X_train,Y_train)
coeff_df = pd.DataFrame(ridgeReg.coef_,X_test.columns,columns=['Coefficient'])
coeff_df.Coefficient.sort_values().plot.bar()
Y_Pred=ridgeReg.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y_test, Y_Pred)
r_squared = r2_score(Y_test, Y_Pred)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)
ridgeReg = Ridge(alpha=0.05, normalize=True)
ridgeReg.fit(X_train,Y_train)
coeff_df = pd.DataFrame(ridgeReg.coef_,X_test.columns,columns=['Coefficient'])
coeff_df.Coefficient.sort_values().plot.bar()
Y_Pred=ridgeReg.predict(X_test)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y_test, Y_Pred)
r_squared = r2_score(Y_test, Y_Pred)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=200, random_state=0)
regressor.fit(X_train, Y_train)
Y_Pred=regressor.predict(X_train)
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y_train, Y_Pred)
r_squared = r2_score(Y_train, Y_Pred)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)
regressor.score(X_train, Y_train)
Y_Pred=regressor.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(Y_test, Y_Pred)
r_squared = r2_score(Y_test, Y_Pred)
print('Mean_Squared_Error :' ,mse)
print('r_square_value :',r_squared)
nndf=pd.DataFrame(Y_test)
nndf['PRED']=Y_Pred
nndf=nndf.reset_index(drop=True)
#Plotting the actual vs predicted values
plt.plot(nndf.index,nndf['LOGPRICE'])
plt.plot(nndf.index,nndf['PRED'], color='red')
plt.xlabel('Index')
plt.ylabel('Values-test')
plt.title('Actual (Blue) vs Predicted (Red)')
plt.show()
#Predicted vs Actual
plt.scatter(nndf['LOGPRICE'],nndf['PRED'])
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
from sklearn.externals import joblib
joblib.dump(lm, 'filename.pkl') 