#Analysis and Visualization of the Largest Dog Breed Data Set; Liam Larsen's Kaggle submission.  

#The Readme contains original background and discussion of the data analysis in this Jupyter Notebook. 



#Import all libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings  #Chose to ignore the warnings that are printed by the Seaborn library

warnings.filterwarnings("ignore")

sns.set(style="white", color_codes=True)
#Read the first .csv file into data frame "df1".

df1=pd.read_csv("../input/2016.csv")
#Determine number of rows and columns.

df1.shape
#View column names and the first five rows.  Verified all other .csv files contain the same column headings so additional

#.csv files can be simply merged.

df1.head()
#View the last five rows and determine the ID for the last row (27017).

df1.tail()
#Read the next nine .csv files into data frame "dfn" where n=2 through 9

df2=pd.read_csv("../input/2015.csv")
df3=pd.read_csv("../input/2014.csv")
df4=pd.read_csv("../input/2013.csv")
df5=pd.read_csv("../input/2012.csv")
df6=pd.read_csv("../input/2011.csv")
df7=pd.read_csv("../input/2010.csv")
df8=pd.read_csv("../input/2009.csv")
df9=pd.read_csv("../input/2008.csv")
df10=pd.read_csv("../input/2007.csv")
#Merge the 10 .csv files into one data frame, df11.

df11=pd.concat([df1,df2,df3,df4,df5,df6,df7,df8,df9,df10])
df11.tail()
#Verify all rows are present.

df11.shape
#Verify all columns are present.

df11.columns
#Flag cells that are blank.

df11.isnull()
#Summarize the location and number of cells that are blank.  See Readme for discussion of cells that are not blank but do not

#contain valid data.

df11.isnull().sum()
#Summarize breed count in descending order.

df11["Breed"].value_counts()
#Use a bar plot to visualize breed count in descending order.  Limit results to top 14 only. 

dog_breed = pd.DataFrame(df11.groupby('Breed').size().sort_values(ascending=False).rename('Amount').reset_index())

f, ax = plt.subplots(figsize=(6, 15))



sns.barplot(x='Amount', y='Breed', data=dog_breed.head(14))

plt.savefig('dogbar.png',bbox_inches='tight')

plt.show()

#Create data frame containing only Breed and ExpYear.  Data frame will be used to create a time series showing change in breed 

#count vs. year for the top five breeds.

df12 = df11.drop(df11.columns[[0,2,3,4,6]],axis=1)
#Create data frame containing count per year for first most popular Breed = Mixed.

df12.head()

df12["ExpYear"].value_counts()

df12["Breed"].value_counts()

mixed = df12[df12["Breed"] == "MIXED"]
#View count per year for Mixed.

mixed["ExpYear"].value_counts()
#Create data frame containing count per year for second most popular Breed = Labrador Retriever.

df12.head()

df12["ExpYear"].value_counts()

df12["Breed"].value_counts()

lab = df12[df12["Breed"] == "LABRADOR RETRIEVER"]
#View count per year for Labrador Retriever.

lab["ExpYear"].value_counts()
#Create data frame containing count per year for third most popular Breed = Lab Mix.

df12.head()

df12["ExpYear"].value_counts()

df12["Breed"].value_counts()

labmix = df12[df12["Breed"] == "LAB MIX"]
#View count per year for Lab Mix.

labmix["ExpYear"].value_counts()
#Create data frame containing count per year for fourth most popular Breed = Golden Retriever.

df12.head()

df12["ExpYear"].value_counts()

df12["Breed"].value_counts()

golden = df12[df12["Breed"] == "GOLDEN RETRIEVER"]
#View count per year for Golden Retriever.

golden["ExpYear"].value_counts()
#Create data frame containing count per year for fifth most popular Breed = Am Pit Bull Terrier.

df12.head

df12["ExpYear"].value_counts()

df12["Breed"].value_counts()

pitbull = df12[df12["Breed"] == "AM PIT BULL TERRIER"]
#View count per year for Am Pit Bull Terrier.

pitbull["ExpYear"].value_counts()
#Use a time series to show changes over time in count of top five breed types.

objects = ('2007','2008','2009','2010','2011','2012','2013','2014','2015','2016')



y_pos = np.arange(len(objects))



data = mixed["ExpYear"].value_counts() 

data1 = lab["ExpYear"].value_counts()

data2 = labmix["ExpYear"].value_counts()

data3 = golden["ExpYear"].value_counts()

data4 = pitbull["ExpYear"].value_counts()



plt.plot(y_pos, data, label = 'Mixed')

plt.plot(y_pos, data1, label = 'Labrador Retriever')

plt.plot(y_pos, data2, label = 'Lab Mix')

plt.plot(y_pos, data3, label = 'Golden Retriever')

plt.plot(y_pos, data4, label = 'Am Pit Bull Terrier')

plt.xticks(y_pos, objects,rotation=90)

plt.legend()

plt.savefig('dogline.png',bbox_inches='tight')

plt.show()
#Summarize color count in descending order.

df11["Color"].value_counts()
#Summarize dog name count in descending order.

df11["DogName"].value_counts()
#Summarize zip code count in descending order.

df11["OwnerZip"].value_counts()