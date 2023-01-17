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
df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')

population = pd.read_csv('/kaggle/input/population/population_by_country_2020.csv')
merged_inner = pd.merge(left=df, right=population, left_on='Country_Region', right_on='Country_Region')
merged_inner = merged_inner[['Id', 'Province_State', 'Country_Region', 'Date', 'ConfirmedCases',

       'Fatalities','Population (2020)','Density (P/Km²)','Land Area (Km²)','Med. Age']]



merged_inner



df = merged_inner
df.Province_State.fillna("NaN",inplace=True)

null_lst = df[df.isnull().any(axis=1)]['Country_Region'].unique()



not_null = set(df['Country_Region'].unique()) - set(null_lst)



not_null = list(not_null)



for i in null_lst:

    pop = df[df['Country_Region'] == i]['Population (2020)'].iloc[0]

    globals()["pop_"+str(i)] = []

    for j in not_null:

        if df[df['Country_Region'] == j]['Population (2020)'].iloc[0] > pop - 500000 and df[df['Country_Region'] == j]['Population (2020)'].iloc[0] < pop + 500000:

            globals()["pop_"+str(i)].append(j)

            print("pop_"+str(i))
l = []

for i in null_lst:

    

    x = 0

    for j in globals()["pop_"+str(i)]:

        x += df[df['Country_Region'] == j]['Med. Age'].iloc[0]

    z = round(x/len(globals()["pop_"+str(i)]),2)

    l.append(z)

#     globals()["age_"+str(i)] = round(x/len(globals()["pop_"+str(i)]),2)



l



j = 0

for i in null_lst: 

    df.loc[df['Country_Region'] == i,'Med. Age'] = l[j]

    j += 1
X = df[df['Country_Region'] == 'US'].groupby('Province_State').sum()



z = X.iloc[:,1].reset_index()



z



df[df['Country_Region'] == 'US']



import matplotlib.pyplot as plt

plt.figure(figsize=(30,10))

plt.hist(z.Province_State,weights=z.ConfirmedCases)

plt.show()







# from sklearn.preprocessing import LabelEncoder

# le = LabelEncoder()



# df['Province_State'] = le.fit_transform(df['Province_State'])

# df['Country_Region'] = le.fit_transform(df['Country_Region'])



df['Date'] = pd.to_datetime(df['Date'])



df['Weeks'] = df.Date.dt.week



X = df[['Province_State','Country_Region','Weeks','Population (2020)','Density (P/Km²)','Land Area (Km²)','Med. Age']]

y = df.ConfirmedCases



X



X['ConfirmedCases'] = y

import seaborn as sn

plt.figure(figsize=(20,20))

corrMatrix = X.corr()

sn.heatmap(corrMatrix, annot=True)

plt.show()



X.drop(['ConfirmedCases','Density (P/Km²)'],axis=1,inplace=True)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)



from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder().fit(X_train['Province_State'])

le2 = LabelEncoder().fit(X_train['Country_Region'])



X_train['Province_State'] = le1.transform(X_train['Province_State'])

X_train['Country_Region'] = le2.transform(X_train['Country_Region'])

X_test['Province_State'] = le1.transform(X_test['Province_State'])

X_test['Country_Region'] = le2.transform(X_test['Country_Region'])
from sklearn.preprocessing import StandardScaler

ss = StandardScaler().fit(X_train)

X_train = ss.transform(X_train)

X_test = ss.transform(X_test)

from sklearn.ensemble import RandomForestRegressor

dt = RandomForestRegressor(n_estimators=400)

dt.fit(X_train,y_train)

dt.score(X_test,y_test)
X1 = df[['Province_State','Country_Region','Weeks','Population (2020)','Density (P/Km²)','Land Area (Km²)','Med. Age','ConfirmedCases']]

y1 = df.Fatalities



X1['Fatalities'] = y1

import seaborn as sn

plt.figure(figsize=(20,20))

corrMatrix = X1.corr()

sn.heatmap(corrMatrix, annot=True)

plt.show()



X1.drop(['Fatalities','Density (P/Km²)'],axis=1,inplace=True)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.33, random_state=42)



from sklearn.preprocessing import LabelEncoder

le1_f = LabelEncoder().fit(X_train['Province_State'])

le2_f = LabelEncoder().fit(X_train['Country_Region'])



X_train['Province_State'] = le1_f.transform(X_train['Province_State'])

X_train['Country_Region'] = le2_f.transform(X_train['Country_Region'])

X_test['Province_State'] = le1_f.transform(X_test['Province_State'])

X_test['Country_Region'] = le2_f.transform(X_test['Country_Region'])



from sklearn.preprocessing import StandardScaler

ss_f = StandardScaler().fit(X_train)

X_train = ss_f.transform(X_train)

X_test = ss_f.transform(X_test)







from sklearn.ensemble import RandomForestRegressor

dt_f = RandomForestRegressor(n_estimators=400)

dt_f.fit(X_train,y_train)

print(dt_f.score(X_test,y_test))



test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv')



merged_inner = pd.merge(left=test, right=population, left_on='Country_Region', right_on='Country_Region')



merged_inner = merged_inner[['ForecastId', 'Province_State', 'Country_Region', 'Date',

       'Population (2020)','Density (P/Km²)','Land Area (Km²)','Med. Age']]







test = merged_inner



test



test.Province_State.fillna("NaN",inplace=True)



for i in list(test.columns):

    print(i,":",test[i].isnull().sum())



null_lst1 = test[test.isnull().any(axis=1)]['Country_Region'].unique()
j = 0

for i in null_lst: 

    test.loc[test['Country_Region'] == i,'Med. Age'] = l[j]

    j += 1
test.drop(['Density (P/Km²)'],axis=1,inplace=True)



test['Date'] = pd.to_datetime(test['Date'])

test['Weeks'] = test.Date.dt.week



test_f = test

test = test[['Province_State','Country_Region','Weeks','Population (2020)','Land Area (Km²)','Med. Age']]



from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder().fit(test['Province_State'])

le2 = LabelEncoder().fit(test['Country_Region'])



test['Province_State'] = le1.transform(test['Province_State'])

test['Country_Region'] = le2.transform(test['Country_Region'])





# from sklearn.preprocessing import StandardScaler



test = ss.transform(test)



test



ypred = dt.predict(test)



ypred = [round(x,0) for x in list(ypred)]



test_f['ConfirmedCases'] = ypred



test_f



test = test_f[['Province_State','Country_Region','Weeks','Population (2020)','Land Area (Km²)','Med. Age','ConfirmedCases']]



test





from sklearn.preprocessing import LabelEncoder

le1_f = LabelEncoder().fit(test['Province_State'])

le2_f = LabelEncoder().fit(test['Country_Region'])



test['Province_State'] = le1_f.transform(test['Province_State'])

test['Country_Region'] = le2_f.transform(test['Country_Region'])







test = ss_f.transform(test)







ypred = dt_f.predict(test)



ypred = [round(i,0) for i in ypred]



test_f['Fatalities'] = ypred



test_f
submission = test_f[['ForecastId','ConfirmedCases','Fatalities']]



submission.to_csv('submission.csv',index=False)