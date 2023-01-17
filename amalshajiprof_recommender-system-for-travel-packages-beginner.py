import warnings

warnings.filterwarnings("ignore")

#importing the packages needed

import pandas as pd

import numpy as np

import re

import nltk
#reading the 12 dataset scraped from different websites

df1=pd.read_csv("../input/travel/d1.csv")

df2=pd.read_csv("../input/travel/d2.csv")

df3=pd.read_csv("../input/travel/d3.csv")

df4=pd.read_csv("../input/travel/d4.csv")

df5=pd.read_csv("../input/travel/d5.csv")

df6=pd.read_csv("../input/travel/d6.csv")

df7=pd.read_csv("../input/travel/d7.csv")

df8=pd.read_csv("../input/travel/d8.csv")

df9=pd.read_csv("../input/travel/d9.csv")

df10=pd.read_csv("../input/travel/d10.csv")

df11=pd.read_csv("../input/travel/d11.csv")

df12=pd.read_csv("../input/travel/d12.csv")
#concatenating all the dataframes to a single one

frames = [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12]

df = pd.concat(frames)

#dropping all the row where the about section is empty

#This is imp as we are analysing the cosine similarity of this section

data=df.dropna(axis = 0, how ='any')
#Joining the columns to make a unique text column which defines a package --- The cosine similarity of this column is used to predict recommendations 

features = ['place','package_name','about_trip']

def combine_features(row):

 return row['place']+" "+row['package_name']+" "+row['about_trip']

for feature in features:

    data[feature] = data[feature].fillna('') #filling all NaNs with blank string

data["combined_features"] = data.apply(combine_features,axis=1) #applying combined_features() method over each rows of dataframe and storing the combined string in “combined_features” column

#data.head(50) ---seeing the data
#downloading stopwords to remove the common words

nltk.download('stopwords')

from nltk.corpus import stopwords

#Function to do basic text cleaning

def clean(text):

           

    # Urls

    text = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", text)

        

    # Words with punctuations and special characters

    punctuations =['@','#','!','?','+','&','*','[',']','-','%','.',':','/','(',')',';','$','=','>','<','|','{','}','^']

    for p in punctuations:

        text = text.replace(p, f' {p} ')



    return text
data.iloc[6] = data.iloc[6].apply(lambda s : clean(s)) #cleaning the text using above function
#removing stopwords 

from nltk.corpus import stopwords

data["combined_features"] = data["combined_features"].str.lower().str.split()

stop = stopwords.words('english')

data['combined_features']=data['combined_features'].apply(lambda x: [item for item in x if item not in stop])

data["combined_features"]= data["combined_features"].str.join(" ") #rejoining the words to text
# defining a new column with no's and setting it as index

ind=[]

for i in range(1609):

     ind.append(i)



data.insert (0,"index",ind)

data.set_index(['index'])
data.describe()
data.head()
data.tail()
data.info()
#import modules

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity
#data.replace('', np.nan, inplace=True)

cv = CountVectorizer() #creating new CountVectorizer() object

count_matrix = cv.fit_transform(data['combined_features'].values.astype('U'))  #fitting cv object to combine features column

cosine_sim = cosine_similarity(count_matrix) #calculating cosine similarity
#creating new column to merge place and package_name which serves as user input identifier

data.head()

features2 = ['place','package_name']

def combine_features2(row):

 return row['place']+" "+row['package_name']

for feature in features2:

    data[feature] = data[feature].fillna('') #filling all NaNs with blank string

data["place_names"] = data.apply(combine_features,axis=1)

data["place_names"] = data["place_names"].str.lower()
#function to search user input on the newly created column and return the index

import re

def get_index_from_title(place):

    for ind in data.index:

        mond=data.iloc[ind]['place_names']

        if re.search(place,mond):

            return(ind)

#finding similar places using cosine similarity

place_selec = "kochi"

place_index = get_index_from_title(place_selec)

#print(place_index)



similar_places = list(enumerate(cosine_sim[place_index])) #accessing the row corresponding to given place name to find all the similarity scores for that place name and then enumerating over it
#sorting those packages in descending order 

sorted_similar_places = sorted(similar_places,key=lambda x:x[1],reverse=True)[1:]

#print(sorted_similar_places)
#printing the top 5 recommendations

i=0

print("Top 5 similar travel packages like "+place_selec+" are:\n")

for element in sorted_similar_places:

    print("Package name: {}".format(data.loc[data['index'] == element[0], 'package_name'].values[0]))

    print("Place: {}".format(data.loc[data['index'] == element[0], 'place'].values[0]))

    dur=data.loc[data['index'] == element[0], 'time'].values[0]

    print("Duration:{}".format(" ".join(dur.split()))) #removing extra places in emi column.

    print("Amount: Rs {} .".format(data.loc[data['index'] == element[0], 'price'].values[0]))

    print('Emi amount: Rs {}'.format(data.loc[data['index'] == element[0], 'emi'].values[0]))

    i=i+1

    if i>5:

        break