import matplotlib as plt
import sklearn as ski
import pandas as pd
import numpy as np
import seaborn as sea

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

pd.options.display.max_rows = None

df = pd.read_excel("../input/statewisetestingdetails/StatewiseTestingDetails.xlsx")
df.drop(['Negative'], axis=1,inplace=True)
df.head(20)

df.dtypes

df['Negative'] = df['TotalSamples'] - df['Positive']
df.head(20)
df['Positive'] = df['Positive'].replace(np.nan, 0)
df['Negative'] = df['Negative'].replace(np.nan, 0)
df.head(20)
total_statewise_sample_captured= df.groupby("State")["TotalSamples"].sum().sort_values(ascending=False).to_frame()
total_statewise_sample_captured
total_statewise_positive = df.groupby("State")["Positive"].sum().sort_values(ascending=False).to_frame()
total_statewise_positive
total_statewise_negative = df.groupby("State")["Negative"].sum().sort_values(ascending=False).to_frame()
total_statewise_negative
plt.pyplot.figure(figsize=(20,10))
x = df['TotalSamples']
y = df['State']
plt.pyplot.scatter(x,y)
plt.pyplot.figure(figsize=(20,10))
x = df['TotalSamples']
y = df['Positive']
plt.pyplot.scatter(x,y)
plt.pyplot.title("A Plot Between Total Samples and Positive Casses")
plt.pyplot.ylabel("Total Samples")
plt.pyplot.xlabel("Positive")
plt.pyplot.figure(figsize=(20,10))
y = df['TotalSamples']
x = df['Negative']
plt.pyplot.scatter(x,y)
plt.pyplot.title("A Plot Between Total Samples and Negative Casses")
plt.pyplot.ylabel("Total Samples")
plt.pyplot.xlabel("Negative")
plt.pyplot.figure(figsize=(20,10))
x = df['Positive']
y = df['Negative']
plt.pyplot.scatter(x,y)
plt.pyplot.title("A Plot Between Positive and Negative Casses")
plt.pyplot.xlabel("Positive")
plt.pyplot.ylabel("Negative")
import seaborn as sea
sea.regplot(x= "Positive",y="Negative", data = df)
plt.pyplot.ylim(0)
sea.regplot(x= "TotalSamples",y="Negative", data = df)
plt.pyplot.ylim(0)
sea.regplot(x= "TotalSamples",y="Positive", data = df)
plt.pyplot.ylim(0)
pearson_coefficeint = df.corr(method='pearson')
pearson_coefficeint
#plt.pyplot.figure(figsize=(5,5))

sea.heatmap(pearson_coefficeint, cmap='RdBu_r',annot=True)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x = df[['Positive','Negative']]
y = df['TotalSamples']

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2)
#plt.pyplot.scatter(x_train,y_train,label="Training Data")


print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)
x_train.head(20)
y_train.head(20)
x_test.head(20)



y_test.head(20)
LR=LinearRegression()
LR.fit(x_train, y_train)



LR.predict(x_test)
LR.score(x_test, y_test)