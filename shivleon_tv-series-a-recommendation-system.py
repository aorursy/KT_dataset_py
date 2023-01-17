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

sns.set(palette="deep")
dataSeries=pd.read_csv(os.path.join(dirname, filename))
dataSeries
#Making deep copies

dataS1=dataSeries.copy(deep=True)
#Naming the 1st column as Sno and making it as index column

dataS1=dataS1.rename(columns={"Unnamed: 0":"Sno"})
dataS1=dataS1.set_index("Sno")
dataS1
dataS1.info()
dataS1.shape
dataS1.isnull().sum()
dataS1.Year.unique()
dataS1.Age.unique()
plt.figure(figsize=(25,7))

sns.countplot(dataS1['Year'])

plt.xticks(rotation=45)

plt.show()
import string
# creating a function that will remove all the punctuations

def remove_punctuations(txt):

    text_nopunct="".join([i for i in txt if i not in string.punctuation])

    return text_nopunct
# we will create a new column with shows name with no punctuations

dataS1['Title_nopunt']=dataS1['Title'].apply(lambda x: remove_punctuations(x))

dataS1['Title_nopunt']=dataS1['Title_nopunt'].str.lower()
dataS1.tail()
plt.figure(figsize=(20,6))

sns.barplot(x='Age', y='IMDb', hue='Netflix', data=dataS1, palette='Reds')

plt.xticks(fontweight='bold')
plt.figure(figsize=(20,6))

sns.barplot(x='Age', y='IMDb', hue='Prime Video', data=dataS1, palette='Blues')

plt.xticks(fontweight='bold')
plt.figure(figsize=(20,6))

sns.barplot(x='Age', y='IMDb', hue='Hulu', data=dataS1, palette='Greens')

plt.xticks(fontweight='bold')
plt.figure(figsize=(20,6))

sns.barplot(x='Age', y='IMDb', hue='Disney+', data=dataS1, palette='ocean')

plt.xticks(fontweight='bold')
plt.figure(figsize=(20,7))

plt.hist(dataS1['IMDb'],edgecolor='#DC143C', label="IMDb Ratings")

plt.legend()

plt.show()
dataS1['Rotten Tomatoes']=dataS1['Rotten Tomatoes'].str.replace("%","")
dataS1['Rotten Tomatoes']=dataS1['Rotten Tomatoes'].astype(float)
dataS1['Rotten Tomatoes']=dataS1['Rotten Tomatoes']/10
dataS1
plt.figure(figsize=(20,7))

plt.hist(dataS1['Rotten Tomatoes'],edgecolor='#DC143C', label="Rotten Tomatoes Ratings")

plt.legend()

plt.show()
def recommend_more(df,namesoftheshows):

    #print(namesoftheshows)

    datasub=df.loc[df['Title_nopunt'].isin(namesoftheshows)] #the one with the namesoftheshows

    #print(datasub)

    datanew=df.loc[~df['Title_nopunt'].isin(namesoftheshows)] # the one without the namesoftheshows, and from where the recommendation will come

    datasub=datasub.drop(['Title'],axis=1)

    datanew=datanew.drop(['Title'],axis=1)

    # now we will make a new dataframe, with Age as base we got from previous df

    listage=list(datasub['Age'])

    #print(listage)

    datanew=datanew.loc[datanew['Age'].isin(listage)] #This one contains only those shows who's age matches with the age of namesoftheshows

    listIMDb=np.array(datasub['IMDb']) #this for multiplication purpose

    

    

    """making dummies"""

    datadummysub=pd.get_dummies(datasub['Age'])

    #print(datadummysub)

    datasub=pd.concat([datasub,datadummysub], axis=1)

    datadummynew=pd.get_dummies(datanew['Age'])

    datanew=pd.concat([datanew, datadummynew], axis=1)

    #datadummysubnetflix=pd.get_dummies(datasub['Netflix'])

    #print(datadummysubnetflix)

    

    """From this point on we are trying to make a normalized user weighted matrix given from his choice of shows"""

    #making weighted matrix for datasub which will be multiplied by listIMDb

    datasub1=datasub.drop(['Year','Age','IMDb','Rotten Tomatoes','type','Title_nopunt'], axis=1)

    #print(datasub1)

    listIMDb=listIMDb.reshape(len(listage),1) #reshaping the matrix so that it could be multiplied

    #print(listIMDb)

    datanum=np.array(datasub1) #changing our datasub into numpy array so that we can multiply listIMDb

    #print(datanum)

    weighted_array=np.multiply(listIMDb,datanum) #making our weighted array

    #print(weighted_array)

    #now making a user weighted matrix

    user_weighted_matrix=np.sum(weighted_array, axis=0) #using np.sum() so as to get column wise sum

    #print(user_weighted_matrix)

    #now making a normalized user weighted matrix

    norm_user_weighted_matrix=user_weighted_matrix/sum(user_weighted_matrix) 

    #print(norm_user_weighted_matrix)

    

    """The previous step is done"""

    

    """Now by using the norm_user_weighted_matrix, we will try to recommend the user a list to shows"""

    datanew1=datanew.drop(['Year','Age','IMDb','Rotten Tomatoes','type','Title_nopunt'], axis=1)

    #print(datanew1)

    datanum1=np.array(datanew1) #this is our candidate matrix

    #print(datanum1)

    weighted_candidate_matrix=np.multiply(norm_user_weighted_matrix,datanum1) #now making weighted candidate matrix

    #print(weighted_candidate_matrix)

    recommendation_candidate_matrix=np.sum(weighted_candidate_matrix, axis=1)

    #print(aggregated_weighted_candidate_matrix)

    #recommendation_candidate_matrix=recommendation_candidate_matrix.reshape(-1,1) **** no need to reshape, that's why commented out

    #print(recommendation_candidate_matrix)

    

    """Now since we got the recommendation matrix, now we will merge the matrix as a column in the datanew matrix"""

    datanew['recommendation_rating']=pd.Series(recommendation_candidate_matrix)

    datanew=datanew.sort_values('recommendation_rating',ascending=False)

    #print(datanew)

    print(datanew[['Title_nopunt','recommendation_rating']].head(10))
recommend_more(dataS1,['breaking bad','stranger things', 'the flash', 'one punch man'])
#Trying another example

recommend_more(dataS1,['stranger things', 'the flash', 'one punch man'])