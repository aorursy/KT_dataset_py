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

import seaborn as sns

import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
data= pd.read_csv("/kaggle/input/us-accidents/US_Accidents_May19.csv")
data.head()
data.info()
data.isnull().sum()
fig=plt.gcf()

fig.set_size_inches(20,20)

fig=sns.heatmap(data.corr(),annot=True,linewidths=1,linecolor='k',square=True,mask=False, 

                vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)


fig=plt.plot()

clr = ("blue", "green", "red", "orange", "purple",'black','pink','gray','darkgreen','brown')

data.State.value_counts().sort_values(ascending=False)[:10].sort_values().plot(kind='barh',color=clr)

fig, ax=plt.subplots()

data['Weather_Condition'].value_counts().sort_values(ascending=False).head(10).plot.bar(width=0.5,edgecolor='k',align='center')

plt.xlabel('Weather_Condition')

plt.ylabel('Number of Accidents')

ax.tick_params()

plt.title('Top 10 Weather Condition for accidents')

plt.ioff()
#Converting the date and time in the standard format.

data['time'] = pd.to_datetime(data.Start_Time, format='%Y-%m-%d %H:%M:%S')

data = data.set_index('time')

data.head()
#Adding an extra column as Day of the week to get the weekday name.

data['Start_Time'] = pd.to_datetime(data['Start_Time'], format="%Y/%m/%d %H:%M:%S")

data['Day'] = data['Start_Time'].dt.weekday_name

data.head()
#Plotting the graph 

fig, ax=plt.subplots()

data['Day'].value_counts().plot.bar(width=0.5,edgecolor='k',align='center')

plt.xlabel('Day of the Week')

plt.ylabel('Number of accidents')

ax.tick_params(labelsize=20)

plt.title('Accidents per day')

plt.ioff()
features=['Source','TMC','Severity','Start_Lng','Start_Lat','Distance(mi)','Side','City','County',

             'State','Timezone','Temperature(F)','Humidity(%)','Pressure(in)', 'Visibility(mi)',

             'Wind_Direction','Weather_Condition','Amenity','Bump','Crossing','Give_Way','Junction',

             'No_Exit','Railway','Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal',

             'Turning_Loop','Sunrise_Sunset','Day']
df=data[features].copy()
df.dropna(subset=df.columns[df.isnull().mean()!=0], how='any', axis=0, inplace=True)
# Select the state of California

state='CA'

df_state=df.loc[df.State==state].copy()

df_state.drop('State',axis=1, inplace=True)

df_state.info()
# Map of accidents, color code by county



sns.scatterplot(x='Start_Lng', y='Start_Lat', data=df_state, hue='County', legend=False, s=20)

plt.show()
# Select San Francisco as the county

county='San Francisco'

df_county=df_state.loc[df_state.County==county].copy()

df_county.drop('County',axis=1, inplace=True)

df_county.info()
#Dealing with categorical variables

#Categorical variables are converted into dummy indicator variables.

df_dummy = pd.get_dummies(df_county,drop_first=True)

target='Severity'

y=df_dummy[target]

x=df_dummy.drop(target,axis=1)
#Splitting using the train-test split.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
lreg=LogisticRegression(random_state=0)

result=lreg.fit(x_train,y_train)

result



y_pred1=lreg.predict(x_test)

acc1=accuracy_score(y_test, y_pred1)

acc1


knn = KNeighborsClassifier(n_neighbors=10)

knn.fit(x_train,y_train)

y_pred2 = knn.predict(x_test)



# Get the accuracy score

acc2=accuracy_score(y_test, y_pred2)

acc2


dt = DecisionTreeClassifier(max_depth=8, criterion='entropy', random_state=1)





# Fit dt_entropy to the training set

dt.fit(x_train, y_train)



# Use dt_entropy to predict test set labels

y_pred3= dt.predict(x_test)



# Evaluate accuracy_entropy

acc3 = accuracy_score(y_test, y_pred3)

acc3

dt_gini = DecisionTreeClassifier(max_depth=8, criterion='gini', random_state=1)





# Fit dt_entropy to the training set

dt_gini.fit(x_train, y_train)



# Use dt_entropy to predict test set labels

y_pred4= dt_gini.predict(x_test)



# Evaluate accuracy_entropy

accuracy_gini = accuracy_score(y_test, y_pred4)

accuracy_gini




rfc=RandomForestClassifier(n_estimators=100)



#Train the model using the training sets y_pred=clf.predict(X_test)

rfc.fit(x_train,y_train)



y_pred5=rfc.predict(x_test)





# Get the accuracy score

acc5=accuracy_score(y_test, y_pred5)



acc5

y_pred5