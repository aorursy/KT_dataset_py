# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv") 

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv") 

sub=pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')
train.head()
data = train.drop(labels=['Id','Province_State','Country_Region','Date'], axis=1)
print(data.head(2))

print(test.head(2))
data2 = train.drop(labels=['Province_State','Country_Region','Date'], axis=1)
print(data2.head(2))
pd.merge(data2,data)
print(data.shape)

print(data2.shape)

print(data2.head(2))

df_new = data2.rename(columns={'Id': 'ForecastId'})
print(df_new.head())

print(df_new.info())
df_new.isna()
df_new.shape
Group_features = ['ConfirmedCases','Fatalities']

fig, ax = plt.subplots(figsize=(10,10)) 

sns.heatmap(df_new[Group_features].corr(), annot = True, fmt = '.2f')

plt.show()
corr = df_new[Group_features].corr()

sns.set(style="white")

# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(10, 10))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
sns.pairplot(df_new)
plt.figure(figsize=(10,5))

sns.kdeplot(df_new['Fatalities'],color='red')

sns.kdeplot(df_new['ConfirmedCases'],color='blue')

plt.title(' ConfirmedCases - Fatalities',size=20)

plt.show()
plt.figure(figsize=(10,8))

df = df_new

sns.lineplot(x="ConfirmedCases", y="Fatalities",data=df_new,label='ConfirmedCases')



plt.title("ConfirmedCases")
# line plot 

# first of all check columns 'data.columns'

# budget vs revenue 

df_new.ConfirmedCases.plot(kind = 'line', color = 'r', label = ' ConfirmedCases', linewidth=1,alpha =0.8  ,grid = True, linestyle = ':')

df_new.Fatalities.plot(color = 'b',label = 'Fatalities',linewidth=1,alpha = 0.8,grid = True, linestyle = '-.')

plt.legend(loc='upper right')

plt.xlabel(' ConfirmedCases ')

plt.ylabel(' Fatalities ')

plt.title('ConfirmedCases vs Fatalities')

plt.show()
# x ConfirmedCases , y Fatalities

df_new.plot(kind='scatter',x='ConfirmedCases',y='Fatalities',alpha=0.5,color='red')

plt.xlabel('ConfirmedCases')

plt.ylabel('Fatalities')

plt.title('ConfirmedCases Count / Fatalities Scatter plot')

plt.show()
sns.set(style="white")

df = df_new.loc[:,['ConfirmedCases','Fatalities']]

g = sns.PairGrid(df, diag_sharey=False)

g.map_lower(sns.kdeplot, cmap="Blues_d")

g.map_upper(plt.scatter)

g.map_diag(sns.kdeplot, lw=3)
# histogram

# values of ConfirmedCases 



plt.plot(df_new.ConfirmedCases,df_new.Fatalities)

plt.xlabel('ConfirmedCases')

plt.ylabel('Fatalities')

plt.show()
# histogram

# values of Fatalities 

df_new.Fatalities.plot(kind = 'hist',bins = 10,figsize = (10,10),color='r')

plt.show()
fig, ax = plt.subplots()

for a in [df_new.ConfirmedCases, df_new.Fatalities]:

    sns.distplot(a, bins=range(1, 110, 10), ax=ax, kde=False)

ax.set_xlim([0, 100])
plt.hist([df_new.Fatalities, df_new.ConfirmedCases], color=['r','b'], alpha=0.5)
df_new = df_new[:-22536]
X = df_new['ConfirmedCases'].values

y = df_new['Fatalities'].values
print(X.shape)

print(y.shape)
X = X.reshape(-1,1)

y = y.reshape(-1,1)
print(X.shape)

print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=42)
# !!! DO NOT FORGET TO LIBRARIES





# Create a k-NN classifier with 7 neighbors: knn

knn = KNeighborsClassifier(n_neighbors=12)



# Fit the classifier to the training data

knn.fit(X_train,y_train)



# Print the accuracy

print('Score', knn.score(X_test, y_test))

# !!! DO NOT FORGET TO LIBRARIES

reg = LinearRegression()



reg.fit(X_train,y_train)

preds = reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test,preds))

print(rmse)

print('Score',reg.score(X_test,y_test))
# !!! DO NOT FORGET TO LIBRARIES



log_reg = LogisticRegression(random_state=0)

log_reg.fit(X_train, y_train)

log_reg.predict(X_test)

print("Score :",log_reg.score(X_test, y_test))
# !!! DO NOT FORGET TO LIBRARIES





# Instantiate model with 1000 decision trees

rf = RandomForestRegressor(n_estimators = 100, random_state = 42)

# Train the model on training data

rf.fit(X_train, y_train);

# Use the forest's predict method on the test data

predictions = rf.predict(X_test)

# Calculate the absolute errors

errors = abs(predictions - y_test)



# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

#Score

print("Score :",rf.score(X_test, y_test))
