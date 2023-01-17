import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')
test = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')

Y1=train['ConfirmedCases']
Y2=train['Fatalities']
test.columns
train['Complete_Date'] = train['Date'].astype('datetime64[ns]')
test['Complete_Date'] = test['Date'].astype('datetime64[ns]')

month = [int(el[5:7]) for el in list(train['Date'].values)]
day = [int(el[8:10]) for el in list(train['Date'].values)]

month_test = [int(el[5:7]) for el in list(test['Date'].values)]
day_test = [int(el[8:10]) for el in list(test['Date'].values)]

df_month= pd.DataFrame(month, columns= ['Month'])
df_day= pd.DataFrame(day, columns= ['Day'])

df_month_test= pd.DataFrame(month_test, columns= ['Month'])
df_day_test= pd.DataFrame(day_test, columns= ['Day'])

train=pd.concat([train, df_month], axis=1)
test=pd.concat([test, df_month_test], axis=1)

train=pd.concat([train, df_day], axis=1)
test=pd.concat([test, df_day_test], axis=1)

train['Date']=train['Month']*100+train['Day']
test['Date']=test['Month']*100+test['Day']
train['Province_State'].fillna('',inplace=True)
test['Province_State'].fillna('',inplace=True)

train['Province_State']=train['Province_State'].astype(str)
test['Province_State']=test['Province_State'].astype(str)

y= train['Country_Region']+train['Province_State']
y= pd.DataFrame(y, columns= ['Place'])

y_test= test['Country_Region']+test['Province_State']
y_test= pd.DataFrame(y_test, columns= ['Place'])

train=pd.concat([train, y], axis=1)
test=pd.concat([test, y_test], axis=1)
Country_df=train["Place"]
ConfirmedCases_df=train["ConfirmedCases"]
Country_df.to_numpy()
ConfirmedCases_df.to_numpy()
Country=Country_df[0]
NbDay = pd.DataFrame(columns=['NbDay'])
day=0
count=0
for x in train["Month"]:
    if (ConfirmedCases_df[count]==0):      
        NbDay = NbDay.append({'NbDay': int(0)}, ignore_index=True)
        count=count+1 
    else:
        if (Country_df[count]==Country):
            day=day+1
            NbDay = NbDay.append({'NbDay': int(day)}, ignore_index=True)
            count=count+1
        else:
            Country=Country_df[count]
            day=1
            NbDay = NbDay.append({'NbDay': int(day)}, ignore_index=True)
            count=count+1
train=pd.concat([train, NbDay], axis=1)

# Adding NbDay feature to the test data
NbDay_test_array=np.zeros(test.shape[0])
i=0
df=test["Place"]
Place_array=df.to_numpy()
for t in test.Date:
    place=Place_array[i]
    if t==402:
        row=train.loc[(train['Place'] == place) & (train['Date'] ==t)]
        row=row.to_numpy()
        NbDay_test_array[i]= row[0][10]
    else: 
        NbDay_test_array[i]=0
    i=i+1

NbDay=pd.DataFrame(NbDay_test_array, columns=['NbDay1'])
test=pd.concat([test,NbDay], axis=1)

Country_df=test["Place"]
NbDay_df=test['NbDay1']
Country_df.to_numpy()
day_array=NbDay_df.to_numpy()
Country=Country_df[0]
NbDay = pd.DataFrame(columns=['NbDay'])
day=0
count=0
for t in test["Date"]:
    if (t==402):
        day=day_array[count] 
        NbDay = NbDay.append({'NbDay': int(day)}, ignore_index=True)  
        count=count+1
    else:
        day=day+1
        NbDay = NbDay.append({'NbDay': int(day)}, ignore_index=True)
        count=count+1
test=pd.concat([test,NbDay], axis=1)
train=train[['Place','Country_Region','NbDay','ConfirmedCases','Fatalities']]
test=test[['Place','Country_Region','NbDay','ForecastId']]
country_array=train['Place'].to_numpy()

def distinct_values(country_array):
    liste=[]
    liste.append(country_array[0])
    for i in range(1,len(country_array)): 
        if country_array[i]!=country_array[i-1]:
            liste.append(country_array[i])
    return liste

Countries_liste=distinct_values(country_array)

len(Countries_liste)
# Adding some exponential features
def exponentiate_alpha(column,v):
    array=column.to_numpy()
    string='NbDay'+str(v)
    array=np.power(v,array)
    frame=pd.DataFrame(array, columns=[string])
    return frame

def product(column1,column2,number):
    array=column1.to_numpy()
    array2=column2.to_numpy()
    
    string='Product'+str(number)
    array=np.multiply(array,array2)
    frame=pd.DataFrame(array, columns=[string])
    return frame
df1=exponentiate_alpha(train['NbDay'],1.0001)
df2=product(df1,train['NbDay'],1)
train['NbDay_exp']=df1
train['Product']=df12
train.columns
import xgboost

model = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=5,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42,
                 objective='reg:squarederror',
                 eval_metric='rmse')
ConfirmedCasesPredictions=[]

i=1
for country in Countries_liste:
    
    #print('For country ',i,': *******************', country, '')
    
    # Train
    train_=train[train['Place']==country][['NbDay']]
    y_=train[train['Place']==country]['ConfirmedCases']
    
    train_=train_.astype(float)

    
    
    # Test
    test_=test[test['Place']==country][['NbDay']]
    
    test_=test_.astype(float)

    
    # Train the model using the training sets
    model.fit(train_, y_)

    
    # Make predictions using the testing set
    y_pred = model.predict(test_)
    y_pred = list(y_pred)

    ConfirmedCasesPredictions+=y_pred
    
    i=i+1
    #print('___________________________________________')
ConfirmedCases=np.array(ConfirmedCasesPredictions)
ConfirmedCases=pd.DataFrame(ConfirmedCases, columns=['ConfirmedCases'])
test['ConfirmedCases']=ConfirmedCases
ConfirmedFatalities=[]

#i=1
for country in Countries_liste:
    
    #print('For country ',i,': *******************', country, '')
    
    # Train
    train_=train[train['Place']==country][['NbDay','ConfirmedCases']]
    y_=train[train['Place']==country]['Fatalities']
    
    train_=train_.astype(float)
    
    # Test
    test_=test[test['Place']==country][['NbDay','ConfirmedCases']]

    test_=test_.astype(float)

    
    # Train the model using the training sets
    model.fit(train_, y_)

    
    # Make predictions using the testing set
    y_pred = model.predict(test_)
    
    y_pred = list(y_pred)
    ConfirmedFatalities+=y_pred
    
    #i=i+1
    #print('___________________________________________')
Fatalities=np.array(ConfirmedFatalities)
Fatalities=pd.DataFrame(Fatalities, columns=['Fatalities'])
test['Fatalities']=Fatalities
# Submission


sub = pd.DataFrame()
sub['ForecastId'] = test['ForecastId']
sub['ConfirmedCases'] = test['ConfirmedCases']
sub['Fatalities'] = test['Fatalities']
sub.to_csv('submission.csv', index=False)