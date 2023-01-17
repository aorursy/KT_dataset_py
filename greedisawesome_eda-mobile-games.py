import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#data visualization

import seaborn as sns 

import matplotlib.pyplot as plt

%matplotlib inline



from scipy import stats

from sklearn.metrics import mean_squared_error as MSE



from sklearn.model_selection import train_test_split

from sklearn import preprocessing

from sklearn import linear_model



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#importing data and cheking it

data = pd.read_csv('/kaggle/input/17k-apple-app-store-strategy-games/appstore_games.csv')
data.head()
data.info()
data.isnull().sum()
data.describe()
data['Average User Rating'].count() 
data['Average User Rating'].isnull().sum()
# droping out unnesessary data, bringing to right type and format (Age rating to numerical, Release date to datetime).

# also removing any duplicated row, based on ID as it is an unique for each application

# Significant problem with Name that for now I cant resolve - it contains some Unicode characters that better to be removed. 

#for example print(data['Name'][700])

data.drop_duplicates(subset = 'ID', keep = 'first', inplace = True)

data = data.drop(['URL', 'Subtitle','Icon URL','Description'], axis=1)

data['Age Rating'] = data['Age Rating'].str.replace('+','')

data['Age Rating'] = pd.to_numeric(data['Age Rating'])

data['Size'] = data['Size']/1024/1024 #size now in MB

data['Size'] = pd.qcut(data['Size'], q=5, labels=False) #splitting Size into 5 equal bins

data['Release_Date'] = pd.to_datetime(data['Original Release Date'], format ='%d/%m/%Y')

data['Current Version Release Date'] = pd.to_datetime(data['Current Version Release Date'], format ='%d/%m/%Y')

data['Release_Year'] = data['Release_Date'].dt.year

data['Release_Month'] = data['Release_Date'].dt.month

data = data.drop(['Original Release Date'], axis = 1)
data.info()
data.head()
data['Primary Genre'].value_counts()
#splitting stings with genres

genres = data['Genres'].str.split(', ')

genres
#counting single appearance of genre in a list wih a Python

#from collections import Counter

#list = []

#for i in data['Genres']:

#    for genre in i.split(', '):

#        list.append(genre)

#d = Counter(list)

#df = pd.Series(d).to_frame('Frequency')

#df_sorted = df.sort_values(by='Frequency', ascending=False)

#df_sorted['Prc'] = df_sorted['Frequency'] / df_sorted['Frequency'].sum()

#df_sorted





#doing the same with Pandas functionality

Genres = pd.DataFrame(data['Genres'].str.split(', ',expand=True))

Genres = pd.DataFrame(Genres.values.ravel(), columns = ["Genres"])

Genres = pd.DataFrame(Genres['Genres'].value_counts().reset_index())

Genres.columns = ['Genres', 'Count']

#plottring top-10 genres

plt.figure(figsize=(10,8))

sns.barplot(x='Genres', y ='Count', data=Genres.head(10)).set_title("Top-10 genres");
data = data.drop(['Primary Genre', 'Genres'], axis = 1)
data['Languages'].value_counts()
languages = pd.DataFrame(data['Languages'].str.split(', ',expand=True))

languages = pd.DataFrame(languages.values.ravel(), columns = ["Languages"])

languages = pd.DataFrame(languages['Languages'].value_counts().reset_index())

languages.columns = ['Language', 'Count']

languages.head(10)
sns.barplot(x="Language", y="Count", data=languages.head(10));
#We are going to transform original Language feature into 2 new features: have english translation/ doesn't have english translation 

#& single/ multiple languages avaliable. After performing this original feature is going to be droped.

data['Eng_lng'] = 0

data['Single_lng'] = 0



data.loc[data['Languages'].str.contains('EN') == True,'Eng_lng'] = 1

data.loc[data['Languages'].str.len() == 2, 'Single_lng'] = 1



data = data.drop('Languages', axis = 1)
#to check if previous operation succeeded

#data[data['Eng_lng'] == 0]

#data[data['Single_lng'] == 0]
#transforming Price feature to nominal feature, where 0 is free-to-play game, and 1 is paid game

data.loc[data['Price'] == 0, 'Price'] = 0

data.loc[data['Price'] > 0, 'Price'] = 1
#Transforming In-app purchases to nominal feature, where 0 is purchases not avaliable and 1 is in-app purchases are avaliable

data['In-app Purchases'] = data['In-app Purchases'].fillna(0)

data.loc[data['In-app Purchases'] != 0, 'In-app Purchases'] = '1'
data['Release_Year'].value_counts().reset_index()
plt.figure(figsize=(11,8))

sns.lineplot(x=data['Release_Year'].value_counts().reset_index().iloc[:,0],

             y=data['Release_Year'].value_counts().reset_index().iloc[:,1]).set_title("Amount of published games by Year")
plt.figure(figsize=(11,8))

sns.lineplot(x=data['Release_Month'].value_counts().reset_index().iloc[:,0],

             y=data['Release_Month'].value_counts().reset_index().iloc[:,1]).set_title("Amount of published games by Month")
sns.countplot(data['Average User Rating'])
data.groupby(['Release_Year'])['Average User Rating'].aggregate('mean')
sns.lineplot(x=data['Release_Year'], y=data['Average User Rating'])
data.groupby(['Release_Year'])['User Rating Count'].aggregate('mean')
sns.lineplot(x=data['Release_Year'], y=data['User Rating Count'])
data.groupby(['Release_Year'])['Price'].aggregate('mean')
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(11, 6))

sns.lineplot(x=data['Release_Year'], y=data['Average User Rating'], hue=data['Price'], ax=axes[0, 0])

axes[0, 0].set_title('Price')

sns.lineplot(x=data['Release_Year'], y=data['Average User Rating'], hue=data['In-app Purchases'], ax=axes[0,1])

axes[0,1].set_title('In-app Purchases')

sns.lineplot(x=data['Release_Year'], y=data['Average User Rating'], hue=data['Eng_lng'], ax=axes[1,0])

axes[1, 0].set_title('Eng_lng')

sns.lineplot(x=data['Release_Year'], y=data['Average User Rating'], hue=data['Single_lng'], ax=axes[1, 1])

axes[1, 1].set_title('Single_lng')
data.sort_values(by=['Average User Rating','User Rating Count'], ascending=False).head(10)
data.sort_values(by=['User Rating Count'], ascending=False).head(10)
#first we are sorting rows based on our conditions and then grouping them and finding the highest value in group. 

#First action is needed because idmax() is returning first highest value, if there is multiple values that fits our condition 

data = data.sort_values(by=['Average User Rating','User Rating Count'], ascending=False)

data.loc[data.groupby(['Release_Year'])['Average User Rating'].idxmax()]
data.loc[data.groupby(['Age Rating'])['Average User Rating'].idxmax()]
data.loc[data.groupby(['Price'])['Average User Rating'].idxmax()]
data.shape
#According to information from owner of dataset, for all games with less then 5 ratings is was replaced to NaN. We can replace it to 0

#For machine learning purposes we filling in all Nan values with 0

data.fillna(0, inplace = True)
data.head()
data.corr()
sns.heatmap(data.corr())
data.head()
#We are not including 'ID' in a analysis because its unique to each application thus it doesnt provide any predictive information.

df_x = data[['User Rating Count','Price','In-app Purchases','Age Rating','Size','Release_Year','Release_Month','Eng_lng','Single_lng']]

df_y = data['Average User Rating']
#Activating LinerRegression model, splitting data for train and test sets, fitting model on train data.

reg = linear_model.LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.33, random_state=1)

reg.fit(x_train,y_train)
#Getting results on effectivness of resulting model

print('Train score: ', reg.score(x_train, y_train), '/n Test score:', reg.score(x_test, y_test))

print(np.sum(reg.coef_!=0))

print(reg.coef_)
#prediction for testing set

y_pred_test = reg.predict(x_test)

print(np.sqrt(MSE(y_test, y_pred_test)))

prediction_results = pd.DataFrame({'Actual': y_test,'Predicted':y_pred_test}) 

prediction_results



#predicting for train set

y_pred_train = reg.predict(x_train)

print(np.sqrt(MSE(y_train, y_pred_train)))

prediction_results = pd.DataFrame({'Actual': y_train,'Predicted':y_pred_train}) 

prediction_results
#checking coefficients for features that our model uses

featuresDF = pd.DataFrame(df_x.columns.values, columns=['Features'])

featuresDF['w'] = reg.coef_

featuresDF