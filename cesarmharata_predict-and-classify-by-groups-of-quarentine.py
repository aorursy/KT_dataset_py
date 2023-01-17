import seaborn as sns

from sklearn.linear_model import LinearRegression

import pandas as pd

from sklearn import metrics

import statsmodels.api as sm

import matplotlib.pyplot as plt

from datetime import datetime, date

import os



train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

train.head()
fig, ax1 = plt.subplots(figsize=(15, 7))



df = pd.melt(train, id_vars="Date", value_vars=["ConfirmedCases","Fatalities"], value_name="number")

total = pd.DataFrame({

    'Date': df['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d')) ,

    'type': df['variable'],

    'number': df['number'].apply(lambda x: float(x)) 

})



train_cross = pd.crosstab(index=total['Date'],columns=total.type,values=total.number,aggfunc='sum',margins=True)

train_cross['Date'] = train_cross.index

train_cross = train_cross.drop(["All"])

sns.lineplot(x="Date" ,y="ConfirmedCases", marker='.',data=train_cross, label = "Cases Confirmed")

sns.lineplot(x="Date" ,y="Fatalities", marker='.',data=train_cross, label="Death")

plt.ylabel('N COVID-19')

plt.xlabel('Time Line')

plt.legend(fontsize=12)
#create model

train_cross['date_ordinal'] = train_cross['Date'].apply(lambda x: x.toordinal())

train_cross['days'] = train_cross['Date'].apply(lambda x: x.toordinal())- date(2019,12,31).toordinal()

rlConfirmed = LinearRegression()

rlFatalities =LinearRegression()



march_train = train_cross.loc[date(year=2020,month=3,day=1):date(year=2020,month=3,day=30)]

# total_confirmed = train_cross.loc[total['type'] == "ConfirmedCases"]

rlConfirmed.fit( march_train[['date_ordinal']] , march_train['ConfirmedCases'])



# total_fatalities = total.loc[total['type'] == "Fatalities"]

rlFatalities.fit( march_train[['date_ordinal']] , march_train['Fatalities'])

print(rlConfirmed.coef_,rlConfirmed.intercept_)

sns.lmplot('ConfirmedCases','days',data=march_train)
#create april range

april = pd.date_range('2020-04-01', periods=30, freq='D')

april_df = pd.DataFrame({

    'Date': april.map(datetime.toordinal)

})



april_predict_confirmed = rlConfirmed.predict(april_df)

april_predict_fatalities = rlFatalities.predict(april_df)
train_cross.tail()
fig, ax1 = plt.subplots(figsize=(15, 7))

april_df_predict = pd.DataFrame({

    'Date': april.map(datetime.date),

    'ConfirmedCases': april_predict_confirmed + train_cross.ConfirmedCases.std(),

    'Fatalities' : april_predict_fatalities + train_cross.Fatalities.std()

})



sns.lineplot(x="Date" ,y="ConfirmedCases", marker='.',data=april_df_predict, label = "Cases Confirmed")

sns.lineplot(x="Date" ,y="Fatalities", marker='.',data=april_df_predict, label="Death")

plt.ylabel('N COVID-19 PREDICT')

plt.xlabel('Time Line')

plt.legend(fontsize=10)
sns.lineplot(x="Date" ,y="ConfirmedCases", marker='.',data=april_df_predict, label = "Cases Confirmed")
sns.lineplot(x="Date" ,y="Fatalities", marker='.',data=april_df_predict, label="Death")
from sklearn.cluster import KMeans

k = KMeans(n_clusters=5)

k.fit(train[['ConfirmedCases']],  train[['Fatalities']])

sns.set(rc={'figure.figsize':(10,8)})

sns.scatterplot(train['ConfirmedCases'],train['Fatalities'], hue=k.labels_, palette=sns.color_palette('Set1',5), sizes=(20, 200))