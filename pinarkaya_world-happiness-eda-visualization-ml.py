# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor# data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
d2015=pd.read_csv("/kaggle/input/world-happiness/2015.csv")
d2016=pd.read_csv("/kaggle/input/world-happiness/2016.csv")
d2017=pd.read_csv("/kaggle/input/world-happiness/2017.csv")
d2018=pd.read_csv("/kaggle/input/world-happiness/2018.csv")
d2019=pd.read_csv("/kaggle/input/world-happiness/2019.csv")
d2015["year"]=2015
d2016["year"]=2016
d2017["year"]=2017
d2018["year"]=2018
d2019["year"]=2019
d2015.head(1)
d2016.head(1)
d2017.head(1)
d2018.head(1)
d2019.head(1)
plt.figure(figsize=(10,5))
sns.kdeplot(d2015['Health (Life Expectancy)'],color='red')
sns.kdeplot(d2016['Health (Life Expectancy)'],color='blue')
sns.kdeplot(d2017['Health..Life.Expectancy.'],color='limegreen')
sns.kdeplot(d2018['Healthy life expectancy'],color='orange')
sns.kdeplot(d2019['Healthy life expectancy'],color='pink')
plt.title('Health',size=20)
plt.show()
plt.figure(figsize=(10,5))
sns.kdeplot(d2015['Family'],color='red')
sns.kdeplot(d2016['Family'],color='blue')
sns.kdeplot(d2017['Family'],color='limegreen')
sns.kdeplot(d2018['Social support'],color='orange')
sns.kdeplot(d2019['Social support'],color='pink')
plt.title('Social support',size=20)
plt.show()
plt.figure(figsize=(10,5))
sns.kdeplot(d2015['Economy (GDP per Capita)'],color='red')
sns.kdeplot(d2016['Economy (GDP per Capita)'],color='blue')
sns.kdeplot(d2017['Economy..GDP.per.Capita.'],color='limegreen')
sns.kdeplot(d2018['GDP per capita'],color='orange')
sns.kdeplot(d2019['GDP per capita'],color='pink')
plt.title('Economy',size=20)
plt.show()
plt.figure(figsize=(10,5))
sns.kdeplot(d2015['Freedom'],color='red')
sns.kdeplot(d2016['Freedom'],color='blue')
sns.kdeplot(d2017['Freedom'],color='limegreen')
sns.kdeplot(d2018['Freedom to make life choices'],color='orange')
sns.kdeplot(d2019['Freedom to make life choices'],color='pink')
plt.title('Freedom ',size=20)
plt.show()
d2015.rename(columns={"Economy (GDP per Capita)":"Economy",
                     "Family":"Social support",
                     "Health (Life Expectancy)":"Health",
                     "Happiness Score":"Score"},inplace=True)
d2016.rename(columns={"Economy (GDP per Capita)":"Economy",
                     "Health (Life Expectancy)":"Health",
                     "Family":"Social support",
                     "Happiness Score":"Score"},inplace=True)
d2017.rename(columns={"Economy..GDP.per.Capita.":"Economy",
                     "Health..Life.Expectancy.":"Health",
                     "Family":"Social support",
                     "Happiness.Rank":"Happiness Rank",
                     "Happiness.Score":"Score"},inplace=True)
d2018.rename(columns={"Country or region":"Country",
                      "GDP per capita":"Economy",
                     "Healthy life expectancy":"Health",
                     "Freedom to make life choices":"Freedom",
                     "Overall rank":"Happiness Rank"},inplace=True)
d2019.rename(columns={"Country or region":"Country",
                      "GDP per capita":"Economy",
                     "Healthy life expectancy":"Health",
                     "Freedom to make life choices":"Freedom",
                     "Overall rank":"Happiness Rank"},inplace=True)
data=pd.concat([d2015,d2016,d2017,d2018,d2019],join="inner")
data.head()
plt.figure(figsize=(8,5))
tr=data[data["Country"]=="Turkey"]
sns.lineplot(x="year", y="Score",data=tr,label='Turkey');
plt.figure(figsize=(16,6))
plt.title("Türkiye İçin Değişkenlerin Yıllara Göre Değişimi")
sns.lineplot(x=tr['year'], y=tr['Economy'], data=tr,label="Economy")
sns.lineplot(x=tr['year'], y=tr['Social support'], data=tr,label="Social support")
sns.lineplot(x=tr['year'], y=tr['Freedom'], data=tr, label="Freedom")
sns.lineplot(x=tr['year'], y=tr['Generosity'], data=tr, label="Generosity");
sns.lineplot(x=tr['year'], y=tr['Health'], data=tr, label="Health")
plt.xlabel("Yıl")
plt.ylabel("seviye");
trcorr=tr.corr()
trcorr
trcorr_=trcorr.drop(['Happiness Rank','Economy','Social support','Freedom','Generosity',"Health","year"], axis=1)
trcorr_=trcorr_.drop(['Happiness Rank',"Score","year"], axis=0)
trcorr_.sort_values("Score",ascending=False)
colors = ['#ff6666', '#468499', '#ff7f50', '#ffdab9', 
          '#00ced1']
fig1, ax1 = plt.subplots(figsize=(10,10))

ax1.pie(np.sqrt(trcorr_.Score*trcorr_.Score),labels=trcorr_.index,colors=colors, 
        autopct='%1.1f%%');
ax1.set_title("Turkey Data",color="black",size=17);
d2019.corr()
plt.figure(figsize=(10,8))
sns.heatmap(d2019.corr(), annot=True);
d2019[d2019['Score']==np.max(d2019['Score'])]['Country']
d2019[d2019['Score']==np.max(d2019['Score'])]['Score']
plt.figure(figsize=(8,5))
fin=data[data["Country"]=="Finland"]
sns.lineplot(x="year", y="Score",data=fin,label='Finland');
fincorr=fin.corr()
fincorr
fincorr_=fincorr.drop(['Happiness Rank','Economy','Social support','Freedom','Generosity',"Health","year"], axis=1)
fincorr_=fincorr_.drop(['Happiness Rank',"Score","year"], axis=0)
fincorr_.sort_values("Score",ascending=False)
colors = ['#ff6666', '#468499', '#ff7f50', '#ffdab9', 
          '#00ced1']
fig1, ax1 = plt.subplots(figsize=(10,10))

ax1.pie(np.sqrt(fincorr_.Score*fincorr_.Score),labels=fincorr_.index,colors=colors, 
        autopct='%1.1f%%');
ax1.set_title("Finland data",color="black",size=17)
maxEconomy=np.max(d2019["Economy"])
maxEconomy
FinEconomy=d2019[d2019["Country"]=="Finland"]["Economy"]
FinEconomy
maxSupport=np.max(d2019["Social support"])
maxSupport
FinSupport=d2019[d2019["Country"]=="Finland"]["Social support"]
FinSupport
x=data.iloc[:,3:]
y=data["Score"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
print(y_train.shape)
print(y_test.shape)
model=RandomForestRegressor(random_state=45)
model.fit(x_train,y_train)
pred=model.predict(x_test)
np.sqrt(mean_squared_error(y_test,pred))
params={"max_depth":list(range(5,10)),
       "max_features":[5,10],
       "n_estimators":[200,500,1000,]}
rf_model=RandomForestRegressor(random_state=42)
rf_model.fit(x_train,y_train)
cv_model=GridSearchCV(rf_model,params,cv=10,n_jobs=-1)
cv_model.fit(x_train,y_train)
cv_model.best_params_
son_model=RandomForestRegressor(max_depth=9,
                                max_features=5,
                                n_estimators=1000)
son_model.fit(x_train,y_train)
predicted=son_model.predict(x_test)
np.sqrt(mean_squared_error(y_test,predicted))
lmodel=LinearRegression().fit(x_train,y_train)
predicted=lmodel.predict(x_train)
np.sqrt(mean_squared_error(y_train,predicted))
model.score(x_train,y_train)
cross_val_score(lmodel,x_train,y_train,cv=10,scoring="r2").mean()
np.sqrt(-cross_val_score(lmodel,
                        x_train,
                        y_train,
                        cv=10,
                        scoring="neg_mean_squared_error")).mean()
