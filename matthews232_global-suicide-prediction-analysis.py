# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Libraries for data visualizations

import matplotlib.pyplot as plt #matplotlib

import matplotlib as mpl 

import seaborn as sns; sns.set() #seaborn

import chart_studio.plotly as py # plotly library to make visualizations

import plotly.graph_objs as go

from sklearn import linear_model

import warnings

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings('ignore') #ignoring warnings

from sklearn.tree import DecisionTreeClassifier

from sklearn import tree

import graphviz

from sklearn.cluster import KMeans

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def create_label_encoder_dict(df):

    from sklearn.preprocessing import LabelEncoder

    

    label_encoder_dict = {}

    for column in df.columns:

        # Only create encoder for categorical data types

        if not np.issubdtype(df[column].dtype, np.number) and column != 'Age':

            label_encoder_dict[column]= LabelEncoder().fit(df[column])

    return label_encoder_dict


data_frame = pd.read_csv("../input/suicide-rates-overview-1985-to-2016/master.csv")
data_frame = data_frame.sort_values(['year'], ascending = True) #Sorting the data in ascending order

print(data_frame.shape) #showing the number of rows and columns in the dataset
# Checking the total unique countries in the dataset

print("Number of Countries in Dataset :", data_frame['country'].nunique())
data_frame.head(15) #Viewing the first 15 rows in the dataset. Having an idea of how everything is
data_frame.columns #Viewing the columns in the dataset
#Renaming of columns to more friendly names

data_frame.rename(columns={'HDI for year': 'HDI_for_year', 'country-year':'country_year', 

                           'suicides/100k pop': 'suicides_100k_pop', ' gdp_for_year ($) ':'gdp_for_year', 

                           'gdp_per_capita ($)':'gdp_per_capita'}, inplace=True);

data_frame.columns #checking if columns got renamed successfully
data_frame.isnull().sum() # checkinng the null values in the dataset
#Creating a copy of the dataset to make changes to

data_frame2 = data_frame
#Dropping unnecessary columns...

data_frame2 = data_frame2.drop(['HDI_for_year'], axis = 1)#Too much missing values plus don't know what HDI is
#Dropping unnecessary columns contd...

data_frame2 = data_frame2.drop(['country_year'], axis = 1)#Not necessary
data_frame2.columns #checking if column got dropped successfully

#data_frame2.shape


data_frame2.isnull().sum() #Rechecking if null values are in dataset
data_frame2.info() #checking the data types of the columns in data frame
#Transformation of data in dataset



# label encoding for sex

from sklearn.preprocessing import LabelEncoder #importing label encoder from sklearn



# creating the encoder

le = LabelEncoder()

data_frame2['sex'] = le.fit_transform(data_frame2['sex'])



data_frame2['sex'].value_counts()
# label encoding for generation

# replacing categorical values in the generation column

print("Number of Generations in Dataset :", data_frame2['generation'].nunique())



data_frame2['generation'] = data_frame2['generation'].replace('Generation X', 0)

data_frame2['generation'] = data_frame2['generation'].replace('Silent', 1)

data_frame2['generation'] = data_frame2['generation'].replace('G.I. Generation', 2)

data_frame2['generation'] = data_frame2['generation'].replace('Boomers', 3)

data_frame2['generation'] = data_frame2['generation'].replace('Millenials', 4)

data_frame2['generation'] = data_frame2['generation'].replace('Generation Z', 5)



# label encoding for generation

# replacing categorical values in the age column

#print("Number of Age Groups in Dataset :", data_frame['age'].nunique())



data_frame2['age'] = data_frame2['age'].replace('5-14 years', 0)

data_frame2['age'] = data_frame2['age'].replace('15-24 years', 1)

data_frame2['age'] = data_frame2['age'].replace('25-34 years', 2)

data_frame2['age'] = data_frame2['age'].replace('35-54 years', 3)

data_frame2['age'] = data_frame2['age'].replace('55-74 years', 4)

data_frame2['age'] = data_frame2['age'].replace('75+ years', 5)



data_frame2['age_bin']=data_frame2['age']

del data_frame2['age']



metadata_age=[(0,'5-14 years'),(1,'15-24 years'),(2,'25-34 years'),(3,'35-54 years'),(4,'55-74 years'),(5,'75+ years')]

metadata_age = pd.DataFrame(metadata_age, columns = ['age_bin' , 'Age'])



metadata_age
#replacing categorical values in the country column

#Country numerical codes were used via pycountry library which usesISO 3166 international standard.

import pycountry



input_countries = data_frame2['country'] #Geting a list of all countries in the country column



countries = {} #creating empty countries list to store all the countries from the dataset

for country in pycountry.countries:

    countries[country.name] = country.numeric #using the numeric identifier code for country via ISO 3166 standard



codes = [countries.get(country, 999) for country in input_countries] #If country cannot be mapped to a code, default value of 999 will be inputted



print(codes) #Printing the list of country codes that mapped to country name in the dataset

#Adding the new column with country codes to dataset 

data_frame2 = pd.DataFrame(data_frame2) # Convert the dictionary into DataFrame 

  

data_frame2.insert(1, "country_code", codes, True)  # Using DataFrame.insert() to add a column 

  

data_frame2.head(10) #Viewing if the changes that were made





data_frame2.isnull().sum() # checking the null values in the dataset
#converting country_code to int

data_frame2['country_code'] = data_frame2['country_code'].astype(np.int64)

metadata_country = data_frame2[['country','country_code']]

metadata_country
#Dropping Country Column



data_frame2 = data_frame2.drop(['country'], axis = 1) #Not necessary anymore since we added the country_code column
# Finding out how many countries are not encoded based on ISO 3166 international standard.

Unknown_Countries = data_frame2[data_frame2['country_code'] == 999]  



Unknown_Countries.shape
#dropping countries with unkown values

data_frame2 = data_frame2.drop(data_frame2[data_frame2['country_code'] == 999].index)





data_frame2[data_frame2['country_code'] == 999]

#Removing commas from gdp_for_year and converting it to integer

data_frame2['gdp_for_year'] = data_frame2['gdp_for_year'].replace(',','',regex = True)

data_frame2['gdp_for_year'] = data_frame2['gdp_for_year'].astype(np.int64)







#checking if the label encoders and data transformation were successfully applied

data_frame2.info()

data_frame2.head(10)
data_frame2.columns
#splitting the dataset into dependent and independent variables

x = data_frame2.drop(['suicides_100k_pop'], axis = 1) #Independent Variables

y = data_frame2['suicides_100k_pop'] #Dependent variable
# splitting the dataset into training and testing sets

#Most data is used for training



from sklearn.model_selection import train_test_split #importing train_test_split from sklearn



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 45) #setting the test size and randomly generate the data

reg = linear_model.LinearRegression()

reg.fit(X_train,y_train)
print("Regression Coefficients")

pd.DataFrame(reg.coef_,index=X_train.columns,columns=["Coefficient"])
# Make predictions using the testing set

test_predicted = reg.predict(X_test)
df3 = X_test.copy()

df3['predicted Y']=test_predicted

df3['Actual Y']=y_test

df3.head(20)
sns.residplot(test_predicted, y_test, lowess=True, color="g")
#trying to see if this plot gives better results

# it does, whats the diff



df3.plot.scatter(

    x='predicted Y',

    y='Actual Y',

    figsize=(12,8)

)



plt.suptitle("Predicted Y vs Actual Y",size=12)

plt.ylabel("Actual Y")

plt.xlabel("Predicted Y")
#what does this r2 value imply?

print('R squared score is %.2f' % r2_score(y_test, test_predicted))

reg.intercept_
#rebuilding index

dftree=data_frame2

dftree = dftree.reset_index(drop=True)
dftree
#binning suicide rates into low medium and high



#the rationale behind the binning is the granularity of the data. ie gender and age group

#0 normal 1 high 2 alarming



bins = [0,40,100,224]

labels=[0,1,2]

dftree['suicide_pop_bin'] = pd.cut(dftree['suicides_100k_pop'], bins=bins, labels=labels, include_lowest=True)



dftree



metadata_suicidepop=[(0,'0-40'),(1,'41-100'),(2,'101-224')]

metadata_suicidepop = pd.DataFrame(metadata_suicidepop, columns = ['suicide_pop_bin' , 'range'])



metadata_suicidepop
dftree.groupby(['suicide_pop_bin']).size()
#binning GDP per capita into low medium and high



#the rationale behind the binning is the granularity of the data. ie gender and age group

#0 normal 1 high 2 alarming



bins = [0,20000,80000,140000]

labels=[0,1,2]

dftree['gdp_per_capita_bin'] = pd.cut(dftree['gdp_per_capita'], bins=bins, labels=labels, include_lowest=True)



dftree

dftree.groupby(['gdp_per_capita_bin']).size()
#binning pop into low medium and high



#the rationale behind the binning is the granularity of the data. ie gender and age group

#0 normal 1 high 2 alarming



bins = [0,700000, 1300000, 7000000,12000000, 30000000,45000000]



labels=[0,1,2,3,4, 5]

dftree['pop_bin'] = pd.cut(dftree['population'], bins=bins, labels=labels, include_lowest=True)



dftree
dftree.groupby(['pop_bin']).size()
dftree
#dftree=dftree[['sex','age', 'pop_bin', 'gdp_for_year', 'gdp_per_capita_bin', 'suicide_bin']]



#dftree


dftree['suicide_bin'] = dftree['suicide_bin'].astype(int)

dftree['pop_bin'] = dftree['pop_bin'].astype(int)

dftree['gdp_per_capita_bin'] = dftree['gdp_per_capita_bin'].astype(int)
backup=dftree.copy()



X_data = dftree[['sex','age', 'pop_bin',  'gdp_per_capita_bin']]

Y_data = dftree['suicide_bin']



X_data
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.30)

clf = DecisionTreeClassifier(criterion='entropy',min_samples_split=2)

clf.fit(X_train, y_train)
pd.DataFrame([ "%.2f%%" % perc for perc in (clf.feature_importances_ * 100) ], index = X_data.columns, columns = ['Matrix of how factors affect suicide rates'])
clf = DecisionTreeClassifier(criterion='entropy',min_samples_split=2) 
clf.fit(X_train, y_train)
class_names = np.unique([str(i) for i in y_train])

class_names
dot_data = tree.export_graphviz(clf,out_file=None, 

                                feature_names=X_data.columns, 

                                class_names=class_names,

                                max_depth=7,

                         filled=True, rounded=True,  proportion=True,

                                node_ids=True, #impurity=False,

                         special_characters=True)
graph = graphviz.Source(dot_data) 

graph
dftree
cluster_data = dftree[['suicides_100k_pop','sex','age']]

cluster_data
data_values = cluster_data.iloc[ :, :].values

data_values
wcss = []

for i in range( 1, 15 ):

    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=300) 

    kmeans.fit_predict( data_values )

    wcss.append( kmeans.inertia_ )

    

plt.plot( wcss, 'ro-', label="WCSS")

plt.title("Computing WCSS for KMeans++")

plt.xlabel("Number of clusters")

plt.ylabel("WCSS")

plt.show()
kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300) 

cluster_data["cluster"] = kmeans.fit_predict( data_values )

cluster_data
cluster_data['cluster'].value_counts()
cluster_data['cluster'].value_counts().plot(kind='bar',title='Suicide Rates')
sns.pairplot( cluster_data, hue="cluster")