# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('seaborn-colorblind')

# Which style do you want to use

#plt.style.available



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from pylab import rcParams

rcParams['figure.figsize'] = 14,6
from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import r2_score

# number of last days for regression

start_point =  20
def MA(df,col,N):

    return df.iloc[:,col].rolling(window=N).mean()
data = pd.read_excel('/kaggle/input/covid-19-turkey/covid19.xlsx',index_col="Day")

covid = data.drop(["Intubated","Intensive Care"],axis=1)

hospital = data.drop(["Cases","Deaths","Recovered","Test"],axis=1)

data.head()
total_cases = np.array(covid.Cases).cumsum()

total_recovered = np.array(covid.Recovered).cumsum()

total_death = np.array(covid.Deaths).cumsum()

total_test = np.array(covid.Test).cumsum()

active_cases = total_cases - total_recovered - total_death

daily_active_cases = np.array(covid.Cases - covid.Recovered - covid.Deaths)

death_rate = total_death / total_cases

recovered_rate = total_recovered / total_cases
hospital["in Mild Condition"] = active_cases - (np.array(hospital["Intubated"])+np.array(hospital["Intensive Care"]))
hospital.dropna(inplace=True)

hospital.tail().style.background_gradient(cmap='Reds')
covid.info()
sns.heatmap(covid.drop("Test",axis=1),yticklabels=False)

plt.show()
sns.heatmap(hospital,yticklabels=False)

plt.show()
sns.swarmplot(data=covid.drop(["Test",],axis=1))

sns.boxplot(data=covid.drop(["Test",],axis=1),color="w")

plt.title("Swarm Plot for Covid Data")

plt.show()
covid.Cases.plot(label="Cases")

MA(covid,0,7).plot(label="7-Days Moving Average")

plt.title("Daily Cases")

plt.legend()

plt.show()



covid.Recovered.plot(label="Recovered")

MA(covid,2,7).plot(label="7-Days Moving Average")

plt.title("Daily Recovered")

plt.legend()

plt.show()



covid.Deaths.plot(label="Deaths")

MA(covid,1,7).plot(label="7-Days Moving Average")

plt.title("Daily Deaths")

plt.legend()

plt.show()



MA(covid,3,7).plot(label="7-Days Moving Average")

plt.bar(covid.index,covid.Test,label="Daily Test",color="y")

plt.title("Daily Test")

plt.legend()

plt.show()
# pie chart of total

labels = ['Active Cases', 'Total Recovered', 'Total Death']

sizes = [active_cases[len(active_cases)-1],covid.Recovered.sum(),covid.Deaths.sum()]

explode = (0, 0, 0.1)

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False,explode=explode, startangle=0)

ax1.axis('equal')

plt.title("Pie Chart for Cases of Covid 19 in Turkey ")

plt.show()
plt.bar(x=covid.index,height=total_cases,label="Total Cases")

plt.bar(x=covid.index,height=total_recovered,label="Total Recovered")

plt.bar(x=covid.index,height=total_death,label="Total Deaths")

plt.xticks(rotation=30)

plt.title("Covid19 in Turkey")

plt.legend()

plt.show()
plt.plot(death_rate,'.-')

plt.title("Death Rate")

plt.show()



plt.plot(daily_active_cases)

plt.plot([0,len(covid)],[0,len(covid)],c='r')

plt.title("Daily Active Cases")

plt.show()
# pie chart of cases

labels = ['Intubated', 'Intensive Care', 'in Mild Condition']

sizes = [hospital["Intubated"][len(hospital)-1],

         hospital["Intensive Care"][len(hospital)-1],

         hospital["in Mild Condition"][len(hospital)-1]]

explode = (0, 0, 0.1)

fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=False,explode=explode, startangle=0)

ax1.axis('equal')

plt.title("Pie Chart for Active Cases of Covid 19 in Turkey")

plt.show()
plt.bar(x=hospital.index,height=active_cases[15:],label="Active Cases")

plt.bar(x=hospital.index,height=hospital["Intensive Care"],label="Intensive Care")

plt.bar(x=hospital.index,height=hospital["Intubated"],label="Intubated")

plt.legend()

plt.show()
plt.plot(hospital["Intensive Care"],label="Intensive Care")

plt.plot(hospital.Intubated,label="Intubated")

plt.title("Hospital Data")

plt.legend()

plt.show()
plt.plot(covid.Cases/covid.Test,label="Case/Test")

plt.plot((covid.Cases/covid.Test).rolling(window=7).mean(),label="7-Days Moving Average",alpha=0.3)

plt.title("Cases / Test Rate")

plt.xlabel("Days")

plt.ylabel("Rate")

plt.xticks(rotation=20)

plt.legend()

plt.show()
X = np.arange(0,start_point).reshape(-1,1)

y = covid.iloc[len(covid)-start_point:,0].values
X_MA7 = np.arange(0,start_point).reshape(-1,1)

y_MA7 = covid.iloc[len(covid)-(start_point+6):,0].rolling(window=7).mean()[6:]
plt.scatter(X,y,label="Actual Value",s=100)

plt.legend()

plt.title("Input:X and Target:y")

plt.show()
plt.scatter(X_MA7,y_MA7,label="7 Days - Moving Averange",s=100)

plt.legend()

plt.title("Input:X and Target:y")

plt.show()
model = LinearRegression()

model.fit(X_MA7,y_MA7)
pred = model.predict(X_MA7)

LR = r2_score(y_MA7,pred)
plt.scatter(X_MA7,y_MA7,label="Actual",s=100)

plt.plot(X_MA7,pred,label="Regression",color='r')

plt.legend()

plt.show()

print("New case for next 3 days:",model.predict([[start_point+1],[start_point+2],[start_point+3]]))
model2 = KNeighborsRegressor(n_neighbors=2)

model2.fit(X_MA7,y_MA7)
pred2 = model2.predict(X_MA7)

KNR = r2_score(y_MA7,pred2)
plt.scatter(X_MA7,y_MA7,label="Actual",s=100)

plt.plot(X_MA7,pred2,label="KNeighborsRegressor",color='r')

plt.legend()

plt.show()

print("New case for next 3 days:",model2.predict([[start_point+1],[start_point+2],[start_point+3]]))
model3 = MLPRegressor(hidden_layer_sizes=(100,),activation='logistic')

model3.solver = 'lbfgs'

model3.max_iter=10000

model3.fit(X_MA7,y_MA7)
pred3 = model3.predict(X_MA7)

MLPR = r2_score(y_MA7,pred3)
plt.scatter(X_MA7,y_MA7,label="Actual",s=100)

plt.plot(X_MA7,pred3,label="MLPRegressor",color='r')

plt.legend()

plt.show()

print("New case for next 3 days:",model3.predict([[start_point+1],[start_point+2],[start_point+3]]))
estimation = {"R2 Score":[LR,KNR,MLPR]}

estimation = pd.DataFrame(estimation,index=["LinearRegression","KNeighborsRegressor","MLPRegressor"])

estimation.style.background_gradient(cmap='Greens')