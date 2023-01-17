# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime, timedelta

#import plotly.offline as py

#py.init_notebook_mode(connected=True)

#import plotly.graph_objs as go



import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # this is used for the plot the graph 

import seaborn as sns # used for plot interactive graph.



import matplotlib.pyplot as plt

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



import warnings

warnings.filterwarnings('ignore')

from pylab import rcParams

# figure size in inches



%matplotlib inline
## Read file



data = pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv")

user_r=pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore_user_reviews.csv")

data

data['Installs'].unique()

data['Installs'].value_counts()

data['Installs'].replace()
# Converting the Installs column to 

#data['Installs']=data['Installs'].str.replace(r",","").str.replace(r"+","")

#data.drop(data.index[10472], inplace=True)

#data['Installs']=pd.to_numeric(data['Installs'])



data.Installs=data.Installs.apply(lambda x: x.strip('+'))

data.Installs=data.Installs.apply(lambda x: x.replace(',',''))

data.Installs=data.Installs.replace('Free',np.nan)

data.Installs.value_counts()

data["Size"].value_counts()

data["Size"].unique()

data['Installs'].value_counts()

# Converting the Rating column to integer form

data["Rating"]=pd.to_numeric(data['Rating'])

data['Rating'].value_counts()

#data['Rating'].unique()





user_r
data.info()

data.describe()

data.columns
#Checking for any null values

data.isnull().info()
data.Reviews.str.isnumeric().sum()
#data[~data.Reviews.str.isnumeric()]
data=data.drop(data.index[10472])
data[10471:].head(2)
#data.iloc[10472, data.columns.get_loc('Reviews')] = "3000000"

#data['Reviews']=data['Reviews'][10472].replace("3.0M","3000000")

print(data["Reviews"].tail())

data['Reviews'].unique()

#Here we see that the data type is string for the "Number of reviews",so we need to change it to numeric type

data['Reviews']=pd.to_numeric(data['Reviews'])

print(data["Reviews"].tail())
# Which



a=data.groupby(["Category"])["App","Reviews"].mean().reset_index()

print(a)
# Which Category of apps have the best Ratings

a=data.groupby(["Category"])["App","Rating"].mean().reset_index()

print(a)
import seaborn as sns

# Number of apps per category



g=sns.countplot(x="Category",data=data,palette="Set1")

# rating distibution 

%matplotlib inline

rcParams['figure.figsize'] = 11.7,8.27

sns.distplot(data["Rating"])
%matplotlib inline

rcParams['figure.figsize'] = 11.7,8.27

g=sns.kdeplot(data["Rating"],color='Red',shade=True)

g.set_xlabel("Rating")

g.set_ylabel("Frequency")

rcParams['figure.figsize']=20,15



# Let"s try to plot some categorical data using catplot function of the seaborn



z=sns.boxplot(x='Category',y='Rating',data=data)

z.set_xticklabels(z.get_xticklabels(),rotation=55)

y=sns.lineplot(x='Category',y='Rating',data=data)

z.set_xticklabels(z.get_xticklabels(),rotation=55)
# Number of Categories present

print("The number of categories present are :",len(data["Category"].unique()))
# Number of apps in each category

g1 = sns.countplot(x="Category",data=data, palette = "Set1")

g1.set_xticklabels(g1.get_xticklabels(),rotation=55)

plt.title('Count of app in each category',size = 20)
# Review distribution

 

rcParams['figure.figsize'] = 11.7,8.27

g = sns.kdeplot(data.Reviews, color="Green", shade = True)

g.set_xlabel("Reviews")

g.set_ylabel("Frequency")

plt.title('Distribution of Reveiw',size = 20)
plt.figure(figsize = (10,10))

sns.regplot(x="Reviews", y="Rating", color = 'darkorange',data=data[data['Reviews']<1000000]);

plt.title('Rating VS Reveiws',size = 20)
# It is in object format so we need to deal with it



data['Size'].value_counts()



# we have "varies with device" a lot so we need to change it a bit

data['Size'].replace("Varies with device",np.nan,inplace=True)

data
#data.Size = (data.Size.replace(r'[kM]+$', '', regex=True).astype(float) * \data.Size.str.extract(r'[\d\.]+([KM]+)', expand=False).fillna(1)

#            .replace(['k','M'], [10**3, 10**6]).astype(int))
data.Size=data.Size.str.replace('k','e+3')

data.Size=data.Size.str.replace('M','e+6')

data.Size=pd.to_numeric(data["Size"])

data.Size.head()
data.Size.unique()
data.hist(column='Size')

plt.xlabel('Size')

plt.ylabel('Frequency')
data.Rating.value_counts()
print("Range:",data.Rating.min(),"-",data.Rating.max())
data.Rating.dtype
print(data.Rating.isna().sum(),"null values out of ",len(data.Rating))
data.Rating.replace(np.nan,data.Rating.mean())
data.Rating.unique()
data.Rating.hist()

plt.xlabel("Rating")

plt.ylabel("Frequecy")
data.Type.value_counts()
data.Type.isna().sum()

data[data.Type.isna()]
data=data.drop(data.index[9148])
#Check if the row is removed

print(data[9146:].head(4))
data.Price.value_counts()
data.Price.unique()
data.Price.apply(lambda x: x.strip('$'))
#data.Price=pd.to_numeric(data.Price)

#data.Price.hist()

#plt.xlabel("Price")

#plt.ylabel("Frequency")
#temp=data[data.Price > 350]

#temp=data.Price.apply(lambda x:True if x >(350) else False)

#data[temp].head(3)
data.Category.unique()
data=data.rename(columns={"Content Rating":"Content_Rating"})



data.columns
data.Content_Rating.unique()
data.Content_Rating.value_counts().plot(kind='bar')

plt.yscale('log')
data.Genres.unique()
data.Genres.value_counts().plot(kind='bar')
sep=";"

rest=data.Genres.apply(lambda x:x.split(sep)[0])

data['Pri_Genres']=rest

data.Pri_Genres.unique()
data['Pri_Genres'].head()
rest=data.Genres.apply(lambda x:x.split(sep)[-1])

rest.unique()

data['Sec_Genres']=rest

data.Sec_Genres.unique()
grouped = data.groupby(['Pri_Genres','Sec_Genres'])

grouped.size().head(15)
data=data.rename(columns={'Last Updated':'Last_Updated'})

data.columns
data.Last_Updated.unique()
from datetime import datetime,date

temp1=pd.to_datetime(data['Last_Updated'])

temp1.head()
data["Last_Updated_Days"]=temp1.apply(lambda x:date.today()-datetime.date(x))

data.Last_Updated_Days.head()
data=data.rename(columns={'Android Ver':'Android_Version'})

data.columns
data.Android_Version.unique()