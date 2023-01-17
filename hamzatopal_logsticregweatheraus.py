import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data vizul

import matplotlib.pyplot as plt  # data vizul

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/weatherAUS.csv")
data.head()
data.info()
data.count().sort_values() # first 4 future half - ilk 4 sütunu siliyoruz çünkü yarısı boş.
data.shape
data=data.drop(columns=["Sunshine","Evaporation","Cloud3pm","Cloud9am","Location","Date","RISK_MM"],axis=1) 
data=data.dropna(how="any") # delete all blank var - boş satırları sildik.
data.shape
#delete outliers - Verimizden aykırı uyuşmayan verileri siliyoruz.(zscore)

from scipy import stats

z = np.abs(stats.zscore(data._get_numeric_data()))

print(z)

data= data[(z < 3).all(axis=1)]

print(data.shape)
data.corr() # corelation
#heatmap

f,ax = plt.subplots(figsize=(12, 12))

sns.heatmap(data.corr(), annot=True, linewidths=0.3,linecolor="red", fmt= '.2f',ax=ax)

plt.show()

plt.figure(figsize=(12,12))

sns.countplot(data=data,x="MaxTemp",order=data.MaxTemp.value_counts().iloc[:12].index)

plt.show()
plt.figure(figsize=(12,12))

sns.countplot(data=data,x="Temp9am",order=data.Temp9am.value_counts().iloc[:12].index)

plt.show()
plt.figure(figsize=(12,12))

sns.countplot(data=data,x="Temp3pm",order=data.Temp3pm.value_counts().iloc[:12].index)

plt.show()
plt.figure(figsize=(6,6))

sns.countplot(data=data,x="RainToday")

plt.show()
plt.figure(figsize=(10,10))

sns.FacetGrid(data, hue="RainTomorrow", height=6).map(sns.kdeplot, "MinTemp").add_legend()

plt.ioff() 

plt.show()
plt.figure(figsize=(10,10))

sns.FacetGrid(data, hue="RainTomorrow", height=6).map(sns.kdeplot, "MaxTemp").add_legend()

plt.ioff() 

plt.show()
plt.figure(figsize=(6,6))

sns.boxplot(data=data,x="RainTomorrow",y="Rainfall")

plt.show()
data.RainToday.isnull().sum()
data['RainToday'].replace({'No': 0, 'Yes': 1},inplace = True)

data['RainTomorrow'].replace({'No': 0, 'Yes': 1},inplace = True)
data.columns
#categorical dato to numerical

testData = data['RainTomorrow']

data = data.drop(columns=['RainTomorrow'])

trainData = pd.get_dummies(data,columns=['WindGustDir', 'WindDir3pm', 'WindDir9am']) #get_dummies
trainData.shape # (107868, 61)

testData.shape # (107868,)
testData=testData.values.reshape(-1,1)
testData.shape # (107868, 1) # because sklearn format.
#transpose(T) train(x,y)  test(y,z) y eşit olmalıdır o sebeble T yaparız.

# %%train test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(trainData, testData, test_size=0.15, random_state=42)



X_train = X_train.T

X_test = X_test.T





print("x train: ",X_train.shape)

print("x test: ",X_test.shape)

print("y train: ",y_train.shape)

print("y test: ",y_test.shape)

# sklearn

from sklearn import preprocessing

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

logReg=LogisticRegression()

logReg = LogisticRegression(random_state = 42,max_iter= 150)

print("test accuracy: {} ".format(logReg.fit(X_train.T, y_train).score(X_test.T, y_test)))

print("train accuracy: {} ".format(logReg.fit(X_train.T, y_train).score(X_train.T, y_train)))
a=(X_test.T).iloc[99:106,:]
a
pre=logReg.predict(a)
print(pre) #Haftlık tahmin.