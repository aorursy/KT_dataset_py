import os

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

from sklearn.metrics import mean_squared_log_error

from sklearn.model_selection import train_test_split

from sklearn import datasets, linear_model

from sklearn.metrics import mean_squared_error, r2_score

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')



Y1=train['ConfirmedCases']

Y2=train['Fatalities']
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
train=train[['Place','NbDay','ConfirmedCases','Fatalities']]

test=test[['Place','NbDay']]



train_data = train

test_data = test
country_array=train_data['Place'].to_numpy()



def distinct_values(country_array):

    liste=[]

    liste.append(country_array[0])

    for i in range(1,len(country_array)): 

        if country_array[i]!=country_array[i-1]:

            liste.append(country_array[i])

    return liste





Countries_liste=distinct_values(country_array)



len(Countries_liste)
def exponentiate_alpha(column,v):



    

    array=column.to_numpy()

    

    string='NbDay'+str(v)

    

    array=np.power(v,array)

        

    frame=pd.DataFrame(array, columns=[string])

    

        

    return frame



liste_mse_countries=[]

liste_r2_countries=[]

results=[]



i=1

for country in Countries_liste:

    

    

    train_NbDay=train[train['Place']==country]['NbDay']

    y_NbDay=train[train['Place']==country]['ConfirmedCases']

    

    test_NbDay=test[test['Place']==country]['NbDay']

    

    alpha=[1+i*0.01 for i in range(1,101)]



    liste_mse_countries=[]

    liste_r2_countries=[]

    liste_mse=[]

    liste_r2=[]

    liste_rmsle=[]

    

    

    i=i+1

    

    for v in alpha: 

    

        

        X1=exponentiate_alpha(train_NbDay,v)

    

        

        X_train,X_test,y_train,y_test = train_test_split(X1,y_NbDay,test_size = 0.3, shuffle= False)

    

        # Create linear regression object

        regr = linear_model.LinearRegression()



        # Train the model using the training sets

        regr.fit(X_train, y_train)



        # Make predictions using the testing set

        y_pred = regr.predict(X_test)

        y_pred = np.maximum(y_pred, 0)

    

        liste_rmsle.append(np.sqrt(mean_squared_log_error( y_test, y_pred )))

        



        liste_r2.append(r2_score(y_test, y_pred))



    argmaximum = np.argmax(liste_r2)

    

    maximum = liste_r2[argmaximum]

    minimum = liste_rmsle[argmaximum]



    

    results.append([country,maximum,minimum,alpha[argmaximum]])



dic_alpha1={}

for liste in results: 

    dic_alpha1[liste[0]]=liste[3]
liste_mse_countries=[]

liste_r2_countries=[]

results2=[]



i=1

for country in Countries_liste:

    

    

    train_NbDay=train[train['Place']==country]['NbDay']

    y_NbDay=train[train['Place']==country]['Fatalities']

    

    test_NbDay=test[test['Place']==country]['NbDay']

    

    alpha=[1+i*0.01 for i in range(1,101)]



    liste_mse_countries=[]

    liste_r2_countries=[]

    liste_mse=[]

    liste_r2=[]

    liste_rmsle=[]

    

    

    i=i+1

    

    for v in alpha: 

    

        

        X1=exponentiate_alpha(train_NbDay,v)

    

        

        X_train,X_test,y_train,y_test = train_test_split(X1,y_NbDay,test_size = 0.3, shuffle= False)

    

        # Create linear regression object

        regr = linear_model.LinearRegression()



        # Train the model using the training sets

        regr.fit(X_train, y_train)





        y_pred = regr.predict(X_test)

        y_pred = np.maximum(y_pred, 0)

    



        liste_rmsle.append(np.sqrt(mean_squared_log_error( y_test, y_pred )))



        liste_r2.append(r2_score(y_test, y_pred))



    argmaximum = np.argmax(liste_r2)

    

    maximum = liste_r2[argmaximum]

    minimum = liste_rmsle[argmaximum]



    

    results2.append([country,maximum,minimum,alpha[argmaximum]])

    

dic_alpha2={}

for liste in results2: 

    dic_alpha2[liste[0]]=liste[3]
ConfirmedCasesPredictions=[]



i=1

for country in Countries_liste:

        

    # Train

    train_NbDay=train[train['Place']==country]['NbDay']

    y_NbDay=train[train['Place']==country]['ConfirmedCases']

    

    

    # Test

    test_NbDay=test[test['Place']==country]['NbDay']

    

    

    #Best alpha1

    v= dic_alpha1[country]

    

    #Modifiying NbDay for test and train    

    X1=exponentiate_alpha(train_NbDay,v)

    X2=exponentiate_alpha(test_NbDay,v)



    

    # Create linear regression object

    regr = linear_model.LinearRegression()



    

    # Train the model using the training sets

    regr.fit(X1, y_NbDay)



    

    # Make predictions using the testing set

    y_pred = regr.predict(X2)

    y_pred = list(np.maximum(y_pred, 0))

    ConfirmedCasesPredictions+=y_pred

    

    i=i+1

    



    

FatalitiesPredictions=[]



i=1

for country in Countries_liste:

        

    # Train

    train_NbDay=train[train['Place']==country]['NbDay']

    y_NbDay=train[train['Place']==country]['Fatalities']

    

    

    # Test

    test_NbDay=test[test['Place']==country]['NbDay']

    

    

    #Best alpha1

    v= dic_alpha2[country]

    

    #Modifiying NbDay for test and train    

    X1=exponentiate_alpha(train_NbDay,v)

    X2=exponentiate_alpha(test_NbDay,v)



    

    # Create linear regression object

    regr = linear_model.LinearRegression()



    

    # Train the model using the training sets

    regr.fit(X1, y_NbDay)



    

    # Make predictions using the testing set

    y_pred = regr.predict(X2)

    y_pred = list(np.maximum(y_pred, 0))

    FatalitiesPredictions+=y_pred

    

    i=i+1
ConfirmedCases=np.array(ConfirmedCasesPredictions)

Fatalities=np.array(FatalitiesPredictions)



ConfirmedCases=pd.DataFrame(ConfirmedCases, columns=['ConfirmedCases'])

Fatalities=pd.DataFrame(Fatalities, columns=['Fatalities'])



# Submission



t = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')



Id=t['ForecastId']





sub = pd.DataFrame()

sub['ForecastId'] = Id

sub['ConfirmedCases'] = ConfirmedCases

sub['Fatalities'] = Fatalities

sub.to_csv('submission.csv', index=False)