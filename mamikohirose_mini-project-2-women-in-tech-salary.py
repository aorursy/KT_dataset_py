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



# Any results you write to the current directory are saved as output.
#import all necessary libraries

from sklearn.cluster import KMeans

from sklearn.preprocessing import MinMaxScaler

from matplotlib import pyplot as plt

import seaborn as sns

import plotly as py

import plotly.graph_objs as go

import warnings

from sklearn.linear_model import LinearRegression



from io import StringIO

%matplotlib inline

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import TfidfVectorizer
#import original data and inspect

df_orig = pd.read_csv("../input/femaletechsalary/Female.Salaries.csv") #original googlesheet dataset

industry_class = pd.read_csv("../input/job-classification/processed_batch.csv") #monkey learn industry lookup

job_class = pd.read_csv("../input/jobclass/processed_batch (1).csv") #monkey learn job title lookup
#create a copy to preserve the original dataset

df_draft = df_orig.copy()
#note that there are 1855 rows to start before we start excluding data.

df_draft.shape
#see how the data is structured

df_draft.head()
#crate new column "Years" to remove non-numeric values from "Optional How many year experience do you have?"

df_draft['Years'] = df_draft['(OPTIONAL) How many year experience do you have?'].str.replace(r"[a-zA-Z]",'')
#convert "Years" column to numeric

df_draft['Years'] = df_draft['Years'].apply(pd.to_numeric, errors='coerce')
#confirm datatype

df_draft.dtypes
#remove deuplicates from lookuptables.

#these lookup tables will be used to append clean industry and job titles to the original dataset

industry_dedupe = industry_class.drop_duplicates()

job_dedupe = job_class.drop_duplicates()
#inspect the lookup tables

industry_dedupe.shape

industry_dedupe.head()
job_dedupe.shape
#Add industry to the original dataset

merge1= job_dedupe.merge(industry_dedupe, on="What is your title?")
#inspect

merge1.head()
#add job title to the original dataset

merge = df_draft.merge(merge1, on='What is your title?')
#inspect

merge.shape
#rename working dataframe to df for easy reference

df=merge
#inspect for good measure

df.shape
df.head()
#rename columns so it's easier to read/reference for later

df.columns = ['Timestamp', 'Title','Orig_Salary','Location','Benefits','Years of Experience','Years','Title Grouping','Group.Score','Industry','Industry.Score']


df.shape
#remove non numeric characters from Annual Salary and convert it to float

df['Annual Salary'] = df['Orig_Salary'].map(lambda x: ''.join([i for i in x if i.isdigit()]))

df['Annual Salary'] = df['Annual Salary'].astype('float64') 
#create salary band column. These thresholds were created after I inspected the CSV output. 

def salary_band(x):

    if x <=15500:

        return 'under exclude'

    elif x > 15500 and x <=400000:

        return 'include'

    elif x > 400000:

        return "exclude"

    else: return 'other'
df["Salary Band"] = df["Annual Salary"].apply(salary_band)

df.shape
#see how the salary bands are distributed by count

df['Salary Band'].value_counts()
#filter to only salary bands that should be included

df = df[(df['Salary Band'] == 'include')]

#view unique locations

df['Location'].unique()
#create a list of non-us locations

non_us = ['Singapore','Montr?al','Vancouver, BC','Vancouver, BC','Australia','Montreal, Canada','Lisbon, Portugal',

'Berlin','Vancouver Canada','Sydney','Mexico','Remote (Jakarta, but for US company)','London, UK','Vancouver, Canada',

'Toronto, Canada','The Hague, Netherlands','Toronto, CA','London','Toronto ','Tokyo, Japan','Malta, Europe','Amsterdam, The Netherlands',

'Cologne, Germany','Ireland','Berlin, Germany','Vancouver BC','Sydney, Australia', 'Brisbane Australia','Melbourne, Australia', 'Sydney/remote',

'Wellington, New Zealand ', 'Perth, Australia', 'Sydney, AU','UK','Brisbane, QLD, Australia','Bath, UK','Belfast, UK','Newcastle, NSW',

'Dublin, Ireland','Melbourne Australia ','Remote (European company)','Halifax, Canada','Newcastle, NSW, Australia','Remote - UK',

'Stockholm, Sweden','United Kingdom','Paris - France ','Canberra, Australia ','Hong Kong','Germany', 'Ukraine, Kyiv', 'Uppsala, Sweden',

'Bangalore, India','MALAGA, SPAIN','Sweden','Toronto, Ontario, Canada ','Oslo, Norway','India','Stockholm, Sweden & Remote','Lahore, Pakistan', 'Stockholm',

'Remote (Company in Spain, working from Nepal)','London,UK','Hampshire, UK','Dublin','Australia ','Ontario Canada','Dublin, Ireland ', 'Copenhagen, Denmark ',

'Gurgaon, India', 'Cambridge, UK', 'North West, England, U.K','Finland','Munich, Germany','Paris, France', 'Nottingham, UK',

'Edinburgh, UK','Sweden, Stockholm','sydney','Barcelona, Spain', 'Cape Town, South Africa','Madrid, Spain','Magdeburg, Germany','Madrid',

'Taipei, Taiwan', 'Taiwan', 'Amsterdam, Netherlands','Zurich','Paris', 'Munich Germany','Remote (Spain)', 'Copenhagen, Denmark', 'Remote / London ','NSW, Australia','Melbourne Australia']
#exclude non us locations from the dataframe

df = df[~df['Location'].isin(non_us)]
#remove unnecessary columns

del df['Timestamp']

del df['Location']

del df['Benefits']

del df['Years of Experience']

del df['Salary Band']

del df['Group.Score']

del df['Industry.Score']
df
#remove null values from "Years" column

filtered_df = df[df['Years'].notnull()]

filtered_df = filtered_df[filtered_df['Years'] > 0]

#rename working dataframe

df = filtered_df
df.shape
#change annual salary to integer

df["Annual Salary"].astype(int)
#export dataframe to csv

df.to_csv('mycsvfile.csv',index=False)
#Create bar chart by count of industry labels

barchart = df['Industry'].value_counts().plot(kind='bar',

                                    figsize=(14,8),

                                    title="Type of Jobs")

barchart.set_xlabel("Type of Jobs")

barchart.set_ylabel("Frequency")
df.Industry.unique()
#Create additional dataframes for the different job types

df1 = df[(df['Industry'] == 'Software Development / IT')]

df2 = df[(df['Industry'] == 'Product Management / Project Management')]

df3 = df[(df['Industry'] == 'Art/Design / Entertainment')]

df4 = df[(df['Industry'] == 'Marketing / Advertising / PR')]
#Understand the distribution of women in tech experience

df['Years'].describe()
X = df['Years'].values.reshape(-1,1) 

Y = df['Annual Salary'].values.reshape(-1,1) 
model = LinearRegression()

model.fit(X,Y)

Y_pred = model.predict(X)



plt.scatter(X,Y, s= 10)

plt.plot(X,Y_pred, color = 'red')

plt.title('Experience to Salary')

plt.xlabel('Years of Experience')

plt.ylabel('Annual Salary')
#create dendogram

dend = df.iloc[:, [2, 5]].values

print(dend)
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(dend, method = 'ward'))

plt.title('Dendrogram')

plt.xlabel('Years of Experience')

plt.ylabel('Euclidean distances')

plt.show()
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

y_hc = hc.fit_predict(X)
plt.scatter(dend[y_hc == 0, 0], dend[y_hc == 0, 1], s = 10, c = 'red', label = 'Cluster 1')

plt.scatter(dend[y_hc == 1, 0], dend[y_hc == 1, 1], s = 10, c = 'blue', label = 'Cluster 2')

plt.scatter(dend[y_hc == 2, 0], dend[y_hc == 2, 1], s = 10, c = 'green', label = 'Cluster 3')

plt.scatter(dend[y_hc == 3, 0], dend[y_hc == 3, 1], s = 10, c = 'cyan', label = 'Cluster 4')

plt.scatter(dend[y_hc == 4, 0], dend[y_hc == 4, 1], s = 10, c = 'magenta', label = 'Cluster 5')

plt.title('Clusters of OVERALL US Women in Tech')

plt.xlabel('Years of Experience')

plt.ylabel('Annual Salary')

plt.legend()

plt.show()
#Understand the distribution of women in tech experience

df1['Years'].describe()
X1 = df1['Years'].values.reshape(-1,1) 

Y1 = df1['Annual Salary'].values.reshape(-1,1) 
model = LinearRegression()

model.fit(X1,Y1)

Y1_pred = model.predict(X1)



plt.scatter(X1,Y1, s= 10)

plt.plot(X1,Y1_pred, color = 'red')

plt.title('Experience to Salary - Software Development')

plt.xlabel('Years of Experience')

plt.ylabel('Annual Salary')
#create dendogram

dend1 = df1.iloc[:, [2, 5]].values

print(dend1)
dendrogram1 = sch.dendrogram(sch.linkage(dend1, method = 'ward'))

plt.title('Dendrogram')

plt.xlabel('Years of Experience')

plt.ylabel('Euclidean distances')

plt.show()
hc1 = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')

y_hc1 = hc1.fit_predict(X1)
plt.scatter(dend1[y_hc1 == 0, 0], dend1[y_hc1 == 0, 1], s = 10, c = 'red', label = 'Cluster 1')

plt.scatter(dend1[y_hc1 == 1, 0], dend1[y_hc1 == 1, 1], s = 10, c = 'blue', label = 'Cluster 2')

plt.scatter(dend1[y_hc1 == 2, 0], dend1[y_hc1 == 2, 1], s = 10, c = 'green', label = 'Cluster 3')

plt.scatter(dend1[y_hc1 == 3, 0], dend1[y_hc1 == 3, 1], s = 10, c = 'cyan', label = 'Cluster 4')

plt.title('Clusters of Software Developers US Women in Tech')

plt.xlabel('Years of Experience')

plt.ylabel('Annual Salary')

plt.legend()

plt.show()
#Understand the distribution of women in tech experience

df2['Years'].describe()
X2 = df2['Years'].values.reshape(-1,1) 

Y2 = df2['Annual Salary'].values.reshape(-1,1) 
model = LinearRegression()

model.fit(X2,Y2)

Y2_pred = model.predict(X2)



plt.scatter(X1,Y1, s= 10)

plt.plot(X1,Y1_pred, color = 'red')

plt.title('Experience to Salry - PM')

plt.xlabel('Years of Experience')

plt.ylabel('Annual Salary')
#create dendogram

dend2 = df2.iloc[:, [2, 5]].values
dendrogram2 = sch.dendrogram(sch.linkage(dend2, method = 'ward'))

plt.title('Dendrogram')

plt.xlabel('Years of Experience')

plt.ylabel('Euclidean distances')

plt.show()
hc2 = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')

y_hc2 = hc1.fit_predict(X2)
plt.scatter(dend2[y_hc2 == 0, 0], dend2[y_hc2 == 0, 1], s = 10, c = 'red', label = 'Cluster 1')

plt.scatter(dend2[y_hc2 == 1, 0], dend2[y_hc2 == 1, 1], s = 10, c = 'blue', label = 'Cluster 2')

plt.scatter(dend2[y_hc2 == 2, 0], dend2[y_hc2 == 2, 1], s = 10, c = 'green', label = 'Cluster 3')

plt.scatter(dend2[y_hc2 == 3, 0], dend2[y_hc2 == 3, 1], s = 10, c = 'cyan', label = 'Cluster 4')

plt.title('Clusters of PM')

plt.xlabel('Years of Experience')

plt.ylabel('Annual Salary')

plt.legend()

plt.show()