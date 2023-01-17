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
import matplotlib.pyplot as plt

import seaborn as sns
data_play = pd.read_csv(os.path.join(dirname, filenames[2]))
data_play
dataplay = data_play.copy(deep =True)
dataplay.head()
def basic_infos(data):

    print("Dataset shape is: ", data.shape,"\n")

    print("Dataset columns are: ",data.columns,"\n")

    print("Dataset dimensions are:",data.ndim,"\n")

    print("Dataset information is:\n",data.info(),"\n")

    categorical, numerical = [], []

    for i in data.columns:

        if dataplay[i].dtype==object:

            categorical.append(i)

        else:

            numerical.append(i)

    print("Categorical datatype columns are: ", [i for i in categorical],"\n")

    print("Numercial datatype columns are: ", [i for i in numerical],"\n")
basic_infos(dataplay)
data_play.iloc[10472]
dataplay.loc[10472, "Category"] = "LIFESTYLE"

dataplay.loc[10472, "Rating"] = 1.9

dataplay.loc[10472, "Reviews"] = 19

dataplay.loc[10472, "Size"] = "3.0M"

dataplay.loc[10472, "Installs"] = "1000+"

dataplay.loc[10472, "Type"] = "Free"

dataplay.loc[10472, "Price"] = "0"

dataplay.loc[10472, "Content Rating"] = "Everyone"

dataplay.loc[10472, "Genres"] = "Lifestyle"

dataplay.loc[10472, "Last Updated"] = "February 11, 2018"

dataplay.loc[10472,"Current Ver"] = "1.0.19"

dataplay.loc[10472, "Android Ver"] = "4.0 and up"
basic_infos(dataplay)
dataplay['Reviews'] = dataplay['Reviews'].astype(int)
basic_infos(dataplay)
dataplay['Size'].unique()
dataplay['Price'].unique()
def replace_in_in_price(price):

    if price == '0':

        price = 0

        return price

    elif '$' in price:

        price = price.replace("$","")

        return float(price)
dataplay['Price'] = dataplay["Price"].apply(lambda x: replace_in_in_price(x))
dataplay['Price'].dtype
dataplay['Last Updated'] = pd.to_datetime(dataplay['Last Updated'])
dataplay.info()
print(dataplay.isnull().sum())
sns.boxplot(dataplay['Rating'])
dataplay["Rating"] = dataplay['Rating'].fillna(dataplay['Rating'].median())
dataplay['Type'] = dataplay['Type'].fillna(dataplay['Type'].mode()[0])
dataplay['Android Ver'] = dataplay['Android Ver'].fillna(dataplay['Android Ver'].mode()[0])
dataplay.isnull().sum()
dataplay.head()
dataplay['Category'].value_counts()
plt.figure(figsize=(30,6))

plt.hist(dataplay['Category'], bins = len(dataplay['Category'].value_counts()), edgecolor="#FF4040")

plt.xticks(rotation=-90)

plt.show()
#plt.figure(figsize=(100,80))

labels=['FAMILY', 'GAME', 'TOOLS', 'MEDICAL', 'BUSINESS', 'PRODUCTIVITY',\

       'PERSOALIZATION', 'COMMUNICATION', 'SPORTS', 'LIFESTYLE', 'FINANCE', 'HEALTH_AND_FITNESS'\

       'PHOTOGRAPHY', 'SOCIAL', 'NEWS_AND_MAGAZINES', 'SHOPPING', 'TRAVEL_AND_LOCAL', 'DATING', 'BOOKS_AND_REFERENCE', 'VIDEO_PLAYER',\

       'EDUCATION', 'EDUCATION', 'ENTERTAINMENT', 'MAPS_AND_NAVIGATION', 'FOODS_AND_DRINK', 'HOUSE_AND_HOME', 'LIBRARIES_AND_DEMO', 'AUTO_AND_VEHICLES',\

       'WHEATHER', 'ART_AND_DESIGN', 'EVENTS', 'COMICS', 'PARENTING', 'BEAUTY']

sizes = dataplay['Category'].value_counts()

fig, ax=plt.subplots()



patches, texts = ax.pie(sizes,shadow=True, startangle=90)



labels = ['{0} - {1:1.2f}'.format(i,j) for i, j in zip(labels,sizes)]

sort_legend = False



plt.legend(patches, labels, loc= 'best', bbox_to_anchor=(-0.1, 1.), fontsize = 10)

ax.axis('equal')

fig = plt.gcf()

fig.set_size_inches(15,15)

plt.show()
dataplay['Rating'].value_counts()
dataplay['Rating'].value_counts().index.tolist() # for labels
plt.figure(figsize=(20,7))

sns.countplot(dataplay['Rating'], label = "Rating")

plt.legend(loc= 'best')

plt.xticks(rotation= -45)

plt.show()
import squarify #for making treemap, we need squarify

plt.figure(figsize=(20,8))

label = dataplay['Rating'].value_counts().index.tolist()

colors = [plt.cm.Spectral(i/float(len(labels))) for i in range(len(labels))]

squarify.plot(sizes = dataplay['Rating'].value_counts(), label = label, color = colors, alpha = 0.8)
dataplay['Size'].value_counts() 
plt.figure(figsize=(20,7))

plt.hist(dataplay['Size'].value_counts(), bins = len(dataplay['Size'].value_counts()))

plt.show()
dataplay['Installs'].value_counts()
plt.figure(figsize=(20,7))

sns.countplot(dataplay['Installs'], edgecolor = "#7FFF00")

plt.xticks(rotation = -45)

plt.show()
dataplay['Price'].value_counts()
plt.figure(figsize=(20,7))

sns.countplot(dataplay['Price'], label= "price")

plt.xticks(rotation=-45)

plt.show()
fig, ax1 = plt.subplots(figsize= (10,7))

fig.patch.set_facecolor('black') # For background

plt.rcParams['text.color'] = 'white' # for changing the text color

lables = dataplay['Price'].value_counts().index.to_list()

size = dataplay['Price'].value_counts()

my_circle = plt.Circle((0,0), 0.9, color = 'black') # for making the circle

percent = 100*np.array(size)/np.array(size).sum() #to show % of every category



#theme = plt.get_cmap('hsv')

#ax1.set_prop_cycle("color", [theme(1. *i / len(size)) for i in range(len(size))])



patches, text = ax1.pie(size) # Making the pie chart



labels = ['{0} - {1:1.2f}'.format(i,j) for i, j in zip(lables,percent)]

sort_legend = False

ax1.axis('equal')



plt.legend(patches, labels, loc= 'best', bbox_to_anchor=(-0.1, 1.), fontsize = 10)

p = plt.gcf()

p.gca().add_artist(my_circle)

plt.show()
dataplay['Content Rating'].value_counts()
plt.figure(figsize=(20,7))

sns.countplot(dataplay['Content Rating'])

plt.xticks(rotation=-90)

plt.show()
fig, ax=plt.subplots()

plt.rcParams['text.color'] = 'black'

labels = dataplay['Content Rating'].value_counts().index.to_list()

sizes = dataplay['Content Rating'].value_counts()

percent = 100*np.array(sizes)/np.array(sizes).sum() #to show % of every category

patches, texts = ax.pie(sizes, shadow=True, startangle=90)

labels = ['{0} - {1:1.2f}%'.format(i,j) for i, j in zip(labels,percent)]

sort_legend = False

ax1.axis('equal')



plt.legend(patches, labels, loc= 'best', bbox_to_anchor=(-0.1, 1.), fontsize = 10)

ax.axis('equal')

plt.show()
dataplay['Genres'].value_counts()
plt.figure(figsize=(50,8))

sns.countplot(dataplay['Genres'])

plt.xticks(rotation = -45)

plt.show()
dataplay['Current Ver'].value_counts()
dataplay['Android Ver'].value_counts()
plt.figure(figsize=(20,8))

sns.countplot(dataplay['Android Ver'])

plt.xticks(rotation = -90)

plt.show()
sns.boxplot(dataplay['Rating'])
# code for changing size

def change_sixe(d):

    if "M" in d:

        d = d.replace("M","")

        try:

            d = int(d)*10**6

            return d

        except ValueError:

            d = int(float(d)*10**6)

            return d

    elif "k" in d:

        d = d.replace("k","")

        try:

            d = int(d)*1000

            return d

        except ValueError:

            d = int(float(d)*1000)

            return d

    else:

        d = 0

        return d

# M being changed to 10**6 and k being changed to 1000. First these letters are being replaced by empty character and then the coversation is done

# Direct coverstion to int cannot happen as as many strings after removal of M or k will be of float type, that's why try-except is used for handling

# We are having many apps with size value as Varies with device, and for our computation we are taking it to be 0
# lets try to apply and get to see if things are going right

dataplay['Size'] = dataplay['Size'].apply( lambda x: change_sixe(x))
dataplay.head()
# Function to remoce , and + from Installs

def remove_plus_and_comma(x):

    x = x.replace(",","")

    x = x.replace("+","")

    return int(x)
dataplay['Installs'] = dataplay['Installs'].apply(lambda x: remove_plus_and_comma(x))
dataplay.head()
dummy_C = pd.get_dummies(dataplay['Category'])

del dummy_C[dummy_C.columns[-1]]#To avoid dummy variable trap

dataplay = pd.concat([dataplay , dummy_C], axis = 1)
dummy_T = pd.get_dummies(dataplay['Type'])

del dummy_T[dummy_T.columns[-1]]#To avoid dummy variable trap

dataplay = pd.concat([dataplay, dummy_T], axis = 1)
dummy_G = pd.get_dummies(dataplay["Genres"])

del dummy_G[dummy_G.columns[-1]]#To avoid dummy variable trap

dataplay = pd.concat([dataplay, dummy_G], axis = 1)
dummy = pd.get_dummies(dataplay['Content Rating'])

del dummy[dummy.columns[-1]]

dataplay = pd.concat([dataplay, dummy], axis = 1)
datamodel = dataplay.drop(["Category", "Type", "Content Rating", "Genres", "App", "Last Updated", "Current Ver", "Android Ver"], axis =1)
datamodel
from sklearn.model_selection import train_test_split
X = datamodel.loc[:, datamodel.columns != "Rating"]
y = datamodel['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 10)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model = lr.fit(X_train, y_train)
y_predict = model.predict(X_test)
y_predict
predict_dataframe = pd.DataFrame(data={"Predicted": y_predict, "Actual": y_test})
predict_dataframe
predict_dataframe[:20].plot(kind = "bar", figsize = (20,8))
model.score(X_test, y_test)
plt.plot(predict_dataframe["Predicted"][:20], "*")

plt.plot(predict_dataframe['Actual'][:20], "^")

plt.show()
fig, ax = plt.subplots()

ax.scatter(y_test, y_predict)

ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw = 4 )

ax.set_xlabel("Actual")

ax.set_ylabel("Predicted")

plt.show()
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Ridge
ridge = Ridge()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]}

ridge_regressor = GridSearchCV(ridge, parameters, scoring = 'neg_mean_squared_error', cv =5)
modelR = ridge_regressor.fit(X_train, y_train)
y_predict_R = modelR.predict(X_test)
rigid_df = pd.DataFrame(data = {"Predicted": y_predict_R, "Actual": y_test})
rigid_df
modelR.score(X_test, y_test)
from sklearn.svm import SVR
svr = SVR()
model_svr = svr.fit(X_train, y_train)
y_predict_svr = model_svr.predict(X_test)
svr_df = pd.DataFrame(data = {"Predicted": y_predict_svr, "Actual": y_test})
svr_df
fig, ax = plt.subplots()

ax.scatter(y_test, y_predict_svr)

ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "k--", lw = 4 )

ax.set_xlabel("Actual")

ax.set_ylabel("Predicted")

plt.show()
plt.plot(svr_df["Predicted"][:20], "*")

plt.plot(svr_df['Actual'][:20], "^")

plt.show()
model_svr.score(X_test, y_test)
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
modef_rfr = rfr.fit(X_train, y_train)
y_predict_rfr = modef_rfr.predict(X_test)
rfr_df = pd.DataFrame(data={"Predicted": y_predict_rfr, "Actual": y_test})
rfr_df
plt.plot(rfr_df["Predicted"][:20], "*")

plt.plot(rfr_df['Actual'][:20], "^")

plt.show()
modef_rfr.score(X_test, y_test)
print("Linear Regression Score: ", model.score(X_test, y_test))

print("Rigid Regression Score: ", modelR.score(X_test, y_test))

print("Support Vector Regression Score: ", model_svr.score(X_test, y_test))

print("Random Forest Regressor Score: ", modef_rfr.score(X_test, y_test))