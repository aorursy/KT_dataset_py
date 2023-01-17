import numpy as np

import pandas as pd

import seaborn as sns

from pandas import set_option

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

fifa = pd.read_csv('players_20.csv')

fifa
peek = fifa.head(10)

print(peek)
fifa.describe()
fifa.shape
fifa.info()
fifa.columns
fifa.dtypes
class_counts = fifa.groupby('overall').size()

print(class_counts)
fifa.isnull().sum().sort_values(ascending=False)
def missing_values_table(fifa):

        mis_val = fifa.isnull().sum()

        mis_val_percent = 100 * fifa.isnull().sum() / len(fifa)

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        print ("Your selected dataframe has " + str(fifa.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        return mis_val_table_ren_columns
missing_values_table(fifa)
fifa.hist('age')

fifa.hist('overall')

fifa.hist('height_cm')

fifa.hist('weight_kg')

fifa.hist('value_eur')



plt.show()
pdscatter = plt.scatter(fifa['overall'],fifa['value_eur'])
# Data to plot



England = len(fifa[fifa['nationality'] == 'England'])

Germany = len(fifa[fifa['nationality'] == 'Germany'])

Spain = len(fifa[fifa['nationality'] == 'Spain'])

Argentina = len(fifa[fifa['nationality'] == 'Argentina'])

France = len(fifa[fifa['nationality'] == 'France'])

Brazil = len(fifa[fifa['nationality'] == 'Brazil'])

Italy = len(fifa[fifa['nationality'] == 'Italy'])

Portugal = len(fifa[fifa['nationality'] == 'Portugal'])

Uruguay = len(fifa[fifa['nationality'] == 'Uruguay'])

Netherlands = len(fifa[fifa['nationality'] == 'Netherlands'])



labels = 'England','Germany','Spain','Argentina','France','Brazil','Italy','Portugal','Uruguay','Netherlands'

sizes = [England,Germany,Spain,Argentina,France,Brazil,Italy,Portugal,Uruguay,Netherlands]

plt.figure(figsize=(6,6))



# Plot

plt.pie(sizes, explode=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05), labels=labels, colors=sns.color_palette("Blues"),

autopct='%1.1f%%', shadow=True, startangle=90)

sns.set_context("paper", font_scale=1.2)

plt.title('Ratio of players by different Nationality', fontsize=16)

plt.show()
sns.set(style="darkgrid")

fig, axs = plt.subplots(nrows=2, figsize=(16, 20))

sns.countplot(fifa['team_position'], palette="RdPu", ax=axs[0])

axs[0].set_title('Number of players per position', fontsize=16)

sns.distplot(fifa['overall'],color="Purple", ax=axs[1])

axs[1].set_title('Distribution of players by Overall', fontsize=16)
fifa.corr()
set_option('display.width', 100)

set_option('precision', 3)

correlations = fifa.corr(method='pearson')

print(correlations)
from matplotlib import pyplot

from pandas import read_csv

import numpy

filename = 'players_20.csv'

names = ['overall'] 

data = read_csv(filename, names=names)

correlations = fifa.corr()

# plot correlation matrix

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(correlations, vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0,9,1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(names)

ax.set_yticklabels(names)

plt.show()
fifa.skew()
missing_values_table(fifa)
X_train = fifa.drop(['overall', 'value_eur'], axis = 1)

X_test = fifa.drop(['overall', 'value_eur'], axis = 1)
# Import label encoder 

from sklearn import preprocessing 

  

# label_encoder object knows how to understand word labels. 

label_encoder = preprocessing.LabelEncoder() 

  

# Encode labels in column 'species'. 

fifa['nationality']= label_encoder.fit_transform(fifa['nationality']) 

  

fifa['nationality'].unique() 
test = fifa.sort_values('overall', ascending=False)

test = test.drop(columns=['player_url'])

test
fifa["overall"] = fifa["overall"].astype('category')

fifa.dtypes
fifa = pd.read_csv('players_20.csv')

df = pd.DataFrame({'x':[1,1,1,2,1,2,2], 'y':['a','b','c','d','e','f','g']})

df = df.groupby('x')['y'].apply(lambda i: ', '.join(i)).reset_index()
from sklearn.model_selection import train_test_split
x=fifa.iloc[:,4] 
x.head()
x.isnull().any()
y=fifa.iloc[:,11]
y.head()
y.isnull().any()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
type(x_train)

type(y_train)
x_train=np.array(x_train)

y_train=np.array(y_train)
type(x_train)

type(y_train)
x_train=x_train.reshape(-1,1)

y_train=y_train.reshape(-1,1)
regressor.fit(x_train,y_train)
x_test=np.array(x_test)
x_test=x_test.reshape(-1,1)
y_pred= regressor.predict(x_test)
plt.scatter(x_train,y_train,color="green")

plt.xlabel("Age of Player")

plt.ylabel("Potential of Player")

plt.plot(x_train, regressor.predict(x_train),color="black") # To draw line of regression

plt.show()
youth_special = fifa[(fifa['overall']>75) & (fifa['potential'] - fifa['overall']>=10)].sort_values(by='overall',ascending=False)

cols = ['short_name','club','dob','overall','potential','team_position','value_eur']

youth_special[cols]