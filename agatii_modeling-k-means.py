# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#  import

import numpy as np

import pandas as pd



# Plotting

from matplotlib import cm

import matplotlib as mpl

import matplotlib.pyplot as plt



# Statistical graphics

import seaborn as sns 



## To Show graphs in same window

get_ipython().run_line_magic('matplotlib', 'inline')



#KMeans

import pylab as pl

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
comp_df=pd.read_csv("../input/free-7-million-company-dataset/companies_sorted.csv")
#Check of dataset

comp_df.head()

comp_df.tail()



comp_df.shape



comp_df.info()

comp_df.describe()



#change name of columns and make it with fist upper letter.

comp_df.rename(columns={'year founded':'year_founded','size range':'size_range','linkedin url':'linkedin_url','current employee estimate':'current_employee_estimate','total employee estimate':'total_employee_estimate'}, inplace=True)

comp_df.columns=comp_df.columns.str.capitalize()

print(comp_df.columns)

comp_df['Country']=comp_df.Country.str.title()

comp_df.head()
#add new variables-Continent

#import dataset

continent_df=pd.read_csv('../input/continent/country_continent.csv',delimiter=';',encoding = "ISO-8859-1") 

continent_df.head()

continent_df.tail()



#leave valuable columns

continent_1=continent_df[['Continent','Country']]

continent_1.head()



#check if there are duplicates

print(any(continent_1['Country'].duplicated()))

continent_1.head()
#new dataframe without missing values on 'Country'

comp_nonan_df = comp_df.dropna(axis=0, subset=['Country'])

comp_nonan_df.shape

comp_nonan_df.info()

comp_nonan_df.head()



#connect dataframes: comp_nonan_df i continent_1f

marge_df=pd.merge(comp_nonan_df, continent_1,on='Country',how='left')

marge_df
#check if there are missing observations after concat

contin_null=marge_df[['Continent','Country']]

contin_null

df_null=contin_null.loc[contin_null['Continent'].isnull()]

df_null

#check unique names of countries that have missing values

df_null['Country'].unique()



#dict for NaN countries and they continent.

dict_null=[{'United States':'North America', 'United Kingdom':'Europe', 'Czechia':'Europe', 'South Korea':'Asia',

       'Taiwan':'Asia', 'Venezuela':'South America', 'Hong Kong':'Asia', 'Russia':'Europe', 'Iran':'Asia', 'Vietnam':'Asia',

       'Palestine':'Asia', 'Trinidad And Tobago':'North America', 'Macau':'Asia', 'Syria':'Asia', 'Tanzania':'Africa',

       'Isle Of Man':'Europe', 'Brunei':'Asia', 'Micronesia':'Oceania', 'Côte D’Ivoire':'Africa',

       'Macedonia':'Europe', 'Bolivia':'South America', 'Moldova':'Europe', 'Bosnia And Herzegovina':'Europe',

       'Democratic Republic Of The Congo':'Africa', 'Netherlands Antilles':'Europe', 'Laos':'Asia',

       'Saint Vincent And The Grenadines':'North America', 'Faroe Islands':'Europe',

       'Saint Kitts And Nevis':'North America', 'Kosovo':'Europe', 'Cape Verde':'Africa',

       'Svalbard And Jan Mayen':'Europe', 'Turks And Caicos Islands':'North America',

       'São Tomé And Príncipe':'Africa', 'Caribbean Netherlands':'North America', 'Sint Maarten':'North America',

       'North Korea':'Asia', 'Antigua And Barbuda':'North America', 'Republic Of The Congo':'Africa',

       'Saint Martin':'North America', 'U.S. Virgin Islands':'North America', 'Saint Pierre And Miquelon':'North America',

       'Saint Barthélemy':'North America'}]



##add dict to dataframe

df_5= pd.DataFrame(dict_null).transpose()

df_5=df_5.reset_index()

df_5.columns=['Country','Continent']

df_5.info()



#concat df_5 and continent_1 

continent_2=pd.concat([continent_1,df_5],sort=True)

continent_2.info()

continent_2.head()



#concat comp_nonan_df, continent_2

fi_df=pd.merge(comp_nonan_df, continent_2,on='Country',how='left')

fi_df.info()

fi_df.head()



#check missing values

fi_df['Continent'].isnull().any()



#drop not needed columns

fi_df.drop(['Unnamed: 0','Domain','Locality','Linkedin_url'], axis=1, inplace=True)

fi_df.info()

fi_df.head()
#dataframe only EU

Europa_df=fi_df[(fi_df['Continent'] == 'Europe')]

Europa_df.head(2)

Europa_df.info()
#IMPORT

pkb_df=pd.read_csv("../input/gdpset/GDP.csv",delimiter=';')

pkb_df.info()

pkb_df.head()

pkb_df=pkb_df.loc[:,['Country','GDP_2016','GDP_2017','GDP_PC_2018']]

pkb_sorted=pkb_df.sort_values(['GDP_2017'], ascending=[True])

pkb_sorted.set_index('Country')

pkb_sorted.head()
# # MODELING



#PREPARATION OF DATA

#marge data Europa_df i pkb_df

pkb_eu=pd.merge(Europa_df, pkb_df,on='Country',how='left')

pkb_eu.info()



#DEL NULL, CHANGE TYPE

pkb_eu_cl = pkb_eu.dropna(axis=0, subset=['GDP_2017'])

pkb_eu_cl['GDP_2017'] = pkb_eu_cl['GDP_2017'].astype('int64')

pkb_eu_cl['GDP_2016'] = pkb_eu_cl['GDP_2016'].astype('int64')

pkb_eu_cl['GDP_PC_2018'] = pkb_eu_cl['GDP_PC_2018'].astype('int64')



#leave importante variables

pkb_eu_cl=pkb_eu_cl.loc[:,['Country','Total_employee_estimate','Current_employee_estimate', 'GDP_2017','GDP_2016','GDP_PC_2018']]



pkb_eu_cl.info()

pkb_eu_cl.head()
#selection of variables for modeling

X = pkb_eu_cl.loc[:,['Total_employee_estimate', 'GDP_2017']]
#elbow curve chart to know  optimum amount of clusters (k)

Nc = range(1, 20)



kmeans = [KMeans(n_clusters=i) for i in Nc]



kmeans



score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))] 



score



pl.plot(Nc,score)



pl.xlabel('Number of Clusters')



pl.ylabel('Score')



pl.title('Elbow Curve')



pl.show()
#algorytm k-means for 5 clasters

k = 5

kmeans = KMeans(n_clusters=k)

x_kmeans=kmeans.fit(X)

labels = kmeans.labels_

labels[::20]

centroids = kmeans.cluster_centers_

centroids



#plot 

for i in range(k):

    # select only data observations with cluster label == i

    ds = X[labels == i]

    # plot the data observations

    plt.plot(ds.iloc[:,0],ds.iloc[:,1],'o')

    # plot the centroids

    lines = plt.plot(centroids[i,0],centroids[i,1],'ro')

    # make the centroid x's bigger

    plt.setp(lines,ms=15.0)

    plt.setp(lines,mew=2.0)

plt.xlabel('Amount of employees')

plt.ylabel('GDP_2017')

plt.title('5 Cluster K-Means')

plt.show()
#ad column with cluster no-use predict.

predict=kmeans.predict(X)

predict_1=predict

predict_1=predict +1

pkb_eu_cl['cluster'] =pd.Series(predict_1, index=pkb_eu_cl.index)

pkb_eu_cl.head()



#check how algorytm asign countries in clusters

for i in range (1,6):

    unique_countries=pkb_eu_cl[pkb_eu_cl.cluster== i].Country.unique()

    print('Cluster', i)

    print(unique_countries)

    print()