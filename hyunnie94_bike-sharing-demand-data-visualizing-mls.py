import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
bike_train = pd.read_csv(r'../input/bike-sharing-demand/train.csv')
bike_train
bike_train.columns
bike_train.index.values
bike_train.info()
bike_train["Date"] = bike_train.datetime.apply(lambda x : x.split()[0])

bike_train["Date"]   #날짜 split
bike_train["Time"] = bike_train.datetime.apply(lambda x : x[11:13].split(':')[0])

bike_train["Time"] = pd.to_numeric(bike_train.Time)

bike_train["Time"]  #시간 split
bike_train["Year"] = bike_train.Date.apply(lambda x : x.split('-')[0])

bike_train["Year"] = pd.to_numeric(bike_train.Year)

bike_train["Year"]  # date 중 연도 split
bike_train["Month"] = bike_train.Date.apply(lambda x : x.split('-')[1])

bike_train["Month"] = pd.to_numeric(bike_train.Month)

bike_train["Month"]  #date 중 월 split
bike_train["Day"] = bike_train.Date.apply(lambda x : x.split('-')[2])

bike_train["Day"] = pd.to_numeric(bike_train.Day)

bike_train["Day"]   # date 중 일 split
pd.concat([bike_train["Date"],bike_train["Time"], bike_train["Year"], bike_train["Month"], bike_train["Day"] ], axis=1)
bike_train = bike_train.drop('Date', axis=1)
bike_train   
bike_train.info()
sns.factorplot(x='Time', y='count', data=bike_train, kind='bar',size=5,aspect=1.5)
bike1 = bike_train[bike_train['Month']==1].groupby('Day')['count'].mean()

bike2 = bike_train[bike_train['Month']==2].groupby('Day')['count'].mean()

bike3 = bike_train[bike_train['Month']==3].groupby('Day')['count'].mean()

bike4 = bike_train[bike_train['Month']==4].groupby('Day')['count'].mean()

bike5 = bike_train[bike_train['Month']==5].groupby('Day')['count'].mean()

bike6 = bike_train[bike_train['Month']==6].groupby('Day')['count'].mean()

bike7 = bike_train[bike_train['Month']==7].groupby('Day')['count'].mean()

bike8 = bike_train[bike_train['Month']==8].groupby('Day')['count'].mean()

bike9 = bike_train[bike_train['Month']==9].groupby('Day')['count'].mean()

bike10 = bike_train[bike_train['Month']==10].groupby('Day')['count'].mean()

bike11 = bike_train[bike_train['Month']==11].groupby('Day')['count'].mean()

bike12 = bike_train[bike_train['Month']==12].groupby('Day')['count'].mean()

bike13 = bike_train.groupby('Day')['count'].mean()
bike_train_Month = pd.DataFrame({'Jan':bike1,'Fab':bike2,'Mar':bike3,'Apr':bike4,'May':bike5,'Jun':bike6,'Jul':bike7,'Aug':bike8,'Sep':bike9,'Oct':bike10,'Nov':bike11,'Dec':bike12})
bike_train_Month
plt.style.use('seaborn-darkgrid')   #Initialize the figure

palette = plt.get_cmap('rainbow')      # create a color palette

fig = plt.figure(figsize=(16,12))

fig.suptitle("Bike demands by month and date", fontsize='x-large')

# multiple line plot

num=0

for column in bike_train_Month:

    num+=1 

    plt.subplot(4,3, num)

    

    plt.plot(bike_train_Month[column], marker='', color= palette(num), linewidth=1.9, alpha=0.9, label=column)

    

    plt.xlim(0, 20)

    plt.ylim(50, 300)

    plt.xticks([0,2,4,6,8,10,12,14,16,18,20])

    if num in range(10) :

        plt.tick_params(labelbottom='off')

    if num not in [1,4,7,10] :

        plt.tick_params(labelleft='off')

        

    plt.title(column, loc='left', fontsize=12, fontweight=0, color=palette(num))



    # Axis title

    plt.xlabel('Day')

    plt.ylabel('count')
plt.style.use('seaborn-darkgrid')

palette = plt.get_cmap('Set3')

fig = plt.figure(figsize=(12,8))

plt.xlabel("Day")

plt.ylabel("count")

plt.title("Bike demends by month", loc='left', fontsize=20, fontweight=0, color='black')



num=0

# multiple line plot

for column in bike_train_Month:

    num+=1

    plt.plot(bike_train_Month[column], marker='', color= palette(num), linewidth=2, alpha=0.9)

    plt.plot(bike13, marker='', color= 'blue', linewidth=4, alpha=0.9)

num=0

for i in bike_train_Month.values[12][0:]:

    num+=1

    name=list(bike_train_Month)[num]

    plt.text(10.2, i, name, horizontalalignment='left', size='small', color='black')



plt.text(10.2, bike13, 'Average demend', horizontalalignment='left', size='small', color='blue')

weekend = bike_train.loc[(bike_train['holiday']== 0) & (bike_train['workingday'] == 0), : ]

weekend

#공휴일은 아니지만 주말
weekday = bike_train.loc[(bike_train['holiday']== 0) & (bike_train['workingday'] == 1), : ]

weekday

# 주중. 일하는 날
holiday = bike_train.loc[(bike_train['holiday']== 1) & (bike_train['workingday'] == 0), : ]

holiday

# 주말이고 공휴일
holidayInworkday = bike_train.loc[(bike_train['holiday']== 1) & (bike_train['workingday'] == 1), : ]

holidayInworkday    # 데이터 없음. 공휴일이면서 일하는 날 일 수 없음.. 그렇다고 한다..
bikeholiday =  pd.DataFrame({'Time':bike_train['Time'],'weekend':weekend['count'], 'weekday':weekday['count'], 'holiday':holiday['count']})
# blue one

plt.plot('Time', 'count', data= weekend.groupby('Time')['count'].max().reset_index(), linestyle='-', marker='o')

# yellow one

plt.plot('Time', 'count', data= weekday.groupby('Time')['count'].max().reset_index(), linestyle='-', marker='o')

# green one

plt.plot('Time', 'count', data= holiday.groupby('Time')['count'].max().reset_index(), linestyle='-', marker='o')

plt.title('Daily bicycle demand based on holidays',fontsize=12, color='black')

plt.xlabel("Time")

plt.ylabel('count')
plt.plot('temp', 'count', data=bike_train, linestyle='', marker='o', markersize=0.7)

plt.xlabel("Temperature")

plt.ylabel("count")
plt.plot('atemp', 'count', data=bike_train, linestyle='', marker='o', markersize=0.7)

plt.xlabel("Haptic Temperature")

plt.ylabel("count")
plt.plot('humidity', 'count', data=bike_train, linestyle='', marker='o', markersize=0.7)

plt.xlabel("Humidity")

plt.ylabel("count")
plt.plot('windspeed', 'count', data=bike_train, linestyle='', marker='o', markersize=0.7)

plt.xlabel("Wind speed")

plt.ylabel("count")
bike_train = bike_train.drop("datetime", axis=1)
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
X = bike_train[['temp', 'holiday',  'atemp', 'humidity', 'Month', 'Day', 'weather', 'season', 'workingday', 'windspeed']]

y = bike_train['count']

X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.3, random_state=42  )

forest = RandomForestClassifier(n_estimators=100, random_state=42)

forest.fit(X_train, y_train)
print("Accuracy : ", forest.score(X_test, y_test))
from sklearn.ensemble import RandomForestRegressor

X = bike_train[['temp', 'holiday', 'atemp', 'humidity', 'Month', 'Day', 'weather', 'season', 'workingday', 'windspeed' ]]

y = bike_train['count']

X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.3, random_state=42 )

forest_reg = RandomForestRegressor(n_estimators=100)

forest_reg.fit(X_train, y_train)
print("Accuracy :", forest_reg.score(X_test, y_test))
from sklearn.linear_model import LinearRegression

X = bike_train[['temp', 'holiday', 'atemp', 'humidity', 'Month', 'Day', 'weather', 'season', 'workingday', 'windspeed' ]]

y = bike_train['count']

lin_reg = LinearRegression()

lin_reg.fit(X_train, y_train)
print("Accuracy :", lin_reg.score(X_test, y_test))
from sklearn.preprocessing import StandardScaler

X = bike_train[['temp', 'holiday', 'atemp', 'humidity', 'Month', 'Day', 'weather', 'season', 'workingday', 'windspeed' ]]

y = bike_train['count']





scaler = StandardScaler()

scaler.fit(X)

X_scaled = scaler.transform(X)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(X_scaled)



X_pca = pca.transform(X_scaled)

print("원본 데이터 형태: {}".format(str(X_scaled.shape)))

print("축소된 데이터 형태: {}".format(str(X_pca.shape)))
plt.figure(figsize=(8,8))

plt.plot(X_pca[y>500,0], X_pca[y>500,1], 'bo')

plt.plot(X_pca[y<30,0], X_pca[y<30,1], 'ys')

plt.xlabel('1st principal component')

plt.ylabel('2nd principal component')
pca.explained_variance_ratio_