import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

pd.set_option('display.float_format', lambda x: '%.4f' % x)

import seaborn as sns

sns.set_context("paper", font_scale=1.3)

sns.set_style('white')

import warnings

warnings.filterwarnings('ignore')

from sklearn import preprocessing

import math

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# dx = pd.read_csv('/kaggle/input/sars-outbreak-2003-complete-dataset/sars_2003_complete_dataset_clean.csv')

# e1 = set(dx.Country)

# print(dx.shape)
df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv",index_col='Id')

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv")

e2= set(df['Country/Region'])

df['ConfirmedCases']=df['ConfirmedCases'].astype(int)

df['Fatalities']=df['Fatalities'].astype(int)

df["Date"] = pd.to_datetime(df["Date"])

df['Weekday']= df.apply(lambda row: row["Date"].weekday(),axis=1)

df["Weekday"] = (df["Weekday"] < 5).astype(int)

print(df.shape)

# df.head()
# e = set(e2).intersection(e1) 
# df_samp = df[df['Country/Region'].isin(e)==True]

# df_samp=df_samp.reset_index()

# df_samp=df_samp.drop(['Id'],axis=1)

# print(df_samp.shape)

# df_samp.head()
df['ConfirmedCases']=df['ConfirmedCases'].astype(int)

df['Fatalities']=df['Fatalities'].astype(int)

df["Date"] = pd.to_datetime(df["Date"])



df['Weekday']= df.apply(lambda row: row["Date"].weekday(),axis=1)

df["Weekday"] = (df["Weekday"] < 5).astype(int)
df.isnull().sum()
df.head()
df1=df.loc[:,['Date','Fatalities']]

df1.set_index('Date',inplace=True)

df1.plot(figsize=(12,5))

plt.ylabel('Global Fatalities')

plt.legend().set_visible(False)

plt.tight_layout()

plt.title('Fatalities Time Series')

sns.despine(top=True)

plt.show();
# set(df[df['Province/State'].isnull()]['Country/Region'])
# df.iloc[:,1:].isnull().sum()
confirmed_total_date = df.groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date = df.groupby(['Date']).agg({'Fatalities':['sum']})

total_date = confirmed_total_date.join(fatalities_total_date)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))

total_date.plot(ax=ax1)

ax1.set_title("Global confirmed cases", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)

fatalities_total_date.plot(ax=ax2, color='orange')

ax2.set_title("Global Fatality cases", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)
# df.head()

# df_temp = df[['Lat','Long','Date','ConfirmedCases','Fatalities','Weekday']]

# # df_temp.groupby(["Date"]).first()

# df.groupby(['Date'], as_index=False).agg({'ConfirmedCases': 'sum', 'Fatalities': 'sum', 'Weekday': 'first', 'Lat': 'first','Long': 'first'})
df.groupby(['Date','Country/Region']).first()
#To have a public leaderboard for this forecasting task, we will be using data from 7 days before to 7 days after competition launch. 

#Only use data on or prior to 2020-03-11 for predictions on the public leaderboard period. 

#Use up to and including the most recent data for predictions on the private leaderboard period.



df_public = df[df["Date"]<"2020-03-12"]

df_public.groupby(['Date','Lat','Long'], as_index=False).agg({'ConfirmedCases': 'sum', 'Fatalities': 'sum', 'Weekday': 'first'})
#I will use this function later



def preprocessing(dataframe):

    z=dataframe['Date']-df['Date'].min()

    for i in z.index:

        z[i]=int(str(z[i]).split()[0])



    data=dataframe

    x =data[['Lat', 'Long', 'Date','Weekday']]

    y1 = data[['ConfirmedCases']]

    y2 = data[['Fatalities']]

    x_test = test[['Lat', 'Long', 'Date']]

    x_test["Date"] = pd.to_datetime(x_test["Date"])

    return z,x,y1,y2
z=df_public['Date']-df_public['Date'].min()

for i in z.index:

    z[i]=int(str(z[i]).split()[0])
data=df_public

x =data[['Lat', 'Long', 'Date','Weekday']]

y1 = data[['ConfirmedCases']]

y2 = data[['Fatalities']]

x_test = test[['Lat', 'Long', 'Date']]

x_test["Date"] = pd.to_datetime(x_test["Date"])
x_test["Weekday"]= x_test.apply(lambda row: row["Date"].weekday(),axis=1)

x_test["Weekday"] = (x_test["Weekday"] < 5).astype(int)
c=z.max()+1

y=x_test['Date']-x_test['Date'].min()

for i in y.index:

    y[i]=int(str(y[i]).split()[0])+c
x['Date']=z

x_test['Date']=y
x_test2=x_test.drop(['Weekday'],axis=1)

x2=x.drop(['Weekday'],axis=1)
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold, cross_val_score, train_test_split
#For smaller dataset

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(x)

x_test_scaled = scaler.transform(x_test)



X_train, X_test, y_train, y_test = train_test_split(X_scaled, y1, test_size=0.2, random_state=44)
models = []

models.append(("RF",RandomForestClassifier()))

models.append(("Dtree",DecisionTreeClassifier()))

models.append(("KNN",KNeighborsClassifier()))
for name,model in models:

    kfold = KFold(n_splits=2, random_state=22)

    cv_result = cross_val_score(model,X_train,y_train, cv = kfold,scoring = "accuracy")

    print(name, cv_result)
#For Fatalities

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y2, test_size=0.2, random_state=44)
for name,model in models:

    kfold = KFold(n_splits=2, random_state=22)

    cv_result = cross_val_score(model,X_train,y_train, cv = kfold,scoring = "accuracy")

    print(name, cv_result)
z,x,y1,y2 = preprocessing(df)
x['Date']=z
def do_your_thing(x,y):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(x)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=44)

    

    models = []

    models.append(("RF",RandomForestClassifier()))

    models.append(("Dtree",DecisionTreeClassifier()))

    models.append(("KNN",KNeighborsClassifier()))



    for name,model in models:

        kfold = KFold(n_splits=2, random_state=22)

        cv_result = cross_val_score(model,X_train,y_train, cv = kfold,scoring = "accuracy")

        print(name, cv_result)    
#Taking entire dataset

#For COnfirmed Cases

print("-----------Confirmed------\n")

do_your_thing(x,y1)

print("-----------Fatality------\n")

#For Fatality

do_your_thing(x,y2)
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold



kfold = KFold(n_splits=2, shuffle = True, random_state = 0)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(x)



X_train1, X_test1, y_train1, y_test1 = train_test_split(X_scaled, y1, test_size=0.2, random_state=44)



best_score = 0

for n_estimators in [90, 100, 110, 120, 200]:

    for max_features in [0.6, 0.7, 0.8, 0.9, 1]:

            print(n_estimators,"--",max_features,"--",'\n')

            rf = RandomForestClassifier(n_estimators = n_estimators, criterion = 'entropy', max_features = max_features)

            scores = cross_val_score(rf, X_train1, y_train1, cv=kfold)

            score = np.mean(scores)

            print("Score is ",score,"\n")

            if score > best_score:

                best_score = score

                best_parameters = {'n_estimators': n_estimators, 'max_features': max_features,'criterion':'entropy'}

                print("BREACHED!!-->",best_parameters,"\n")
print("Params: ", best_parameters)
scaler = StandardScaler()

X_scaled = scaler.fit_transform(x)

rf = RandomForestClassifier(**best_parameters)

rf.fit(X_scaled,y1)
test_scaled = scaler.transform(x_test)
#Notice the small x

y_pred_confirmed = rf.predict(test_scaled)
predictions = pd.DataFrame({'ForecastId':test['ForecastId'],'ConfirmedCases':y_pred_confirmed})

predictions.head()
scaler = StandardScaler()

X_scaled = scaler.fit_transform(x)



X_train1, X_test1, y_train1, y_test1 = train_test_split(X_scaled, y2, test_size=0.2, random_state=44)



best_score = 0

for n_estimators in [90, 100, 110, 120, 200]:

    for max_features in [0.6, 0.7, 0.8, 0.9]:

            print(n_estimators,"--",max_features,'\n')

            rf = RandomForestClassifier(n_estimators = n_estimators, criterion = 'entropy', max_features = max_features)

            scores = cross_val_score(rf, X_train1, y_train1, cv=kfold)

            score = np.mean(scores)

            print("Score is ",score,"\n")

            if score > best_score:

                best_score = score

                best_parameters = {'n_estimators': n_estimators, 'max_features': max_features,'criterion':'entropy'}

                print("BREACHED!!-->",best_parameters,"\n")
print("Params: ", best_parameters)
scaler = StandardScaler()

X_scaled = scaler.fit_transform(x)



rf = RandomForestClassifier(**best_parameters)

rf.fit(X_scaled,y2)
test_scaled = scaler.transform(x_test)
#Notice the small x

y_pred_fatal = rf.predict(test_scaled)
predictions = pd.DataFrame({'ForecastId':test['ForecastId'],'ConfirmedCases':y_pred_confirmed,'Fatalities':y_pred_fatal})

predictions.head()
# X_train1, X_test1, y_train1, y_test1 = train_test_split(x, y1, test_size=0.2, random_state=44)

# X_train2, X_test2, y_train2, y_test2 = train_test_split(x, y2, test_size=0.2, random_state=44)



# rf_model1 = RandomForestClassifier().fit(X_train1, y_train1)

# rf_model2 = RandomForestClassifier().fit(X_train2, y_train2)



# print('---------------------Confirmed------------------------\n')

# print('Accuracy of RF classifier on training set: {:.2f}'

#        .format(rf_model.score(X_train1, y_train1)))

# print('Accuracy of RF classifier on test set: {:.2f}'

#        .format(rf_model.score(X_test1[X_train1.columns], y_test1)))



# print('-----------------------Fatal------------------------\n')

# print('Accuracy of RF classifier on training set: {:.2f}'

#        .format(rf_model.score(X_train2, y_train2)))

# print('Accuracy of RF classifier on test set: {:.2f}'

#        .format(rf_model.score(X_test2[X_train2.columns], y_test2)))
# #Notice the small x

# y_pred_fatal = rf_model.predict(x_test[X_train.columns])
# predictions = pd.DataFrame({'ForecastId':test['ForecastId'],'ConfirmedCases':y_pred,'Fatalities':y_pred_fatal})

# predictions.head()
predictions.to_csv('submission.csv', header=True, index=False)
# model.fit(x,y1)

# pred1 = model.predict(x_test)

# pred1 = pd.DataFrame(pred1)

# pred1.columns = ["ConfirmedCases_prediction"]



# model.fit(x,y2)

# pred2 = model.predict(x_test)

# pred2 = pd.DataFrame(pred1)

# pred2.columns = ["Death_prediction"]
# model_1 = DecisionTreeRegressor() # raw

# model_2 = LinearRegression() # raw

# model_3 = RandomForestRegressor(n_estimators=50, random_state=0) # tunned

# model_4 = RandomForestRegressor(max_leaf_nodes=10, random_state=0) # tunned



# # Tunned models list

# models = [model_1, model_2, model_3, model_4]



# # lets define a counter for mean absolute error

# def MAE(model, x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val):

#     model.fit(X_train, y_train)

#     prediction1 = model.predict(X_val)

#     print("MEAN ABSOLUTE ERROR: ", mean_absolute_error(y_val, prediction1))



# # lets check the MAE

# for i in models:

#     MAE(i)
# temp=df.loc[:,['Lat','Long','ConfirmedCases','Fatalities']]
# plots=df.groupby(['Country/Region'])
# fig, ax = plt.subplots(figsize=(15,7))

# plots.plot(x='Date',y='ConfirmedCases',ax=ax,legend=False)
# ax= sns.scatterplot(x='Date',y='ConfirmedCases',data=df)

# ax.set(xlim = ('2020-01', '2020-03'))
# sns.relplot(x="Date",y="ConfirmedCases",kind="line",data=df)