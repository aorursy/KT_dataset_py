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
## IMPORTING TOOLS

import os

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

from nltk.tokenize import word_tokenize, sent_tokenize 

from nltk import pos_tag

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer

import re



from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,adjusted_rand_score



from sklearn.naive_bayes import MultinomialNB 

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier, export_graphviz

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer

from xgboost import XGBClassifier

%matplotlib inline
## Reading and exploring on train data¶

train = pd.read_csv("../input/Train-1554810061973.csv")

print(train.shape)

train.head()
### Reading and exploring on test data

test = pd.read_csv("../input/Test-1555730055539.csv")

print(test.shape)

test.head()
### Reading and exploring on Existing hotels Data

exist = pd.read_csv("../input/ExistingHotels_CustomerVisitsdata-1554810038262.csv")

print(exist.shape)

exist.head(3)
### Reading and exploring on New_hotels Data

New = pd.read_csv("../input/NewHotels_CutstomerVisitsdata-1554810098964.csv")

print(New.shape)

New.head(2)
#### Reading and exploring on User_ratings data¶

ratings = pd.read_csv("../input/user_hotel_rating-1555730075105.csv")

print(ratings.shape)

ratings.head(2)
#### VISUALIZATIONS on Train Data

# Let's look at the top 10 reviewed Hotels

top_reviewed_hotels = train.Hotelid.value_counts()

top_reviewed_hotels[:10].plot(kind='barh',figsize=(20,10),title="TOP 10 REVIEWED HOTELS on traindata",legend=True,colormap="PiYG",fontsize=15)

_=plt.xlabel('HotelID',fontsize=20)

_=plt.ylabel('Total No. of reviews',fontsize=20)

### Checking the reviews count with respect to date column

train['Date']=pd.to_datetime(train['Date'])



days = {0:'Mon',1:'Tues',2:'Weds',3:'Thurs',4:'Fri',5:'Sat',6:'Sun'}



train["month"]=train["Date"].dt.month

train["Year"]=train["Date"].dt.year

train["Day"]=train["Date"].dt.day

train["dayOftheweek"] = train["Date"].dt.dayofweek

train['dayOftheweek'] = train['dayOftheweek'].apply(lambda x: days[x])

Review_Day_Count = train['Day'].value_counts()

plt.figure(figsize=(10,4))

sns.barplot(Review_Day_Count.index, Review_Day_Count.values, alpha=0.8)

plt.ylabel("Number Of Review")

plt.xlabel("Reviews By Days")

plt.title('Total reviews count by Day', loc='Center', fontsize=14)

plt.show()



Reviews_Count_Month = train['month'].value_counts()

plt.figure(figsize=(10,4))

sns.barplot(Reviews_Count_Month.index, Reviews_Count_Month.values, alpha=0.8)

plt.ylabel("Number Of Review")

plt.xlabel("Reviews By Months")

plt.title('Total reviews count by month', loc='Center', fontsize=14)

plt.show()



Reviews_Year = train['Year'].value_counts()

plt.figure(figsize=(10,4))

sns.barplot(Reviews_Year.index, Reviews_Year.values, alpha=0.8)

plt.ylabel("Number Of Review")

plt.xlabel("Reviews By Year")

plt.title('Total reviews count by Year', loc='Center', fontsize=14)
Sentiment_count = train['Sentiment'].value_counts()

plt.figure(figsize=(10,4))

sns.barplot(Sentiment_count.index, Sentiment_count.values, alpha=0.8)

plt.ylabel("Count")

plt.xlabel("Feedback Count")

plt.title('Feedback counts across the train data', loc='Center', fontsize=14)

plt.show()
#Top 10 feed back given hotels

top_fb_hotels = train.groupby('Hotelid')['Sentiment'].value_counts().sort_values(ascending=False).head(10)

top_fb_hotels.plot(kind="barh",color="gold",title="Top 10 feed back given hotels ",legend=True,figsize=(10,10))

_=plt.xlabel('count')

_=plt.ylabel('HotelID')

plt.show()
#Least 10 feed back given hotels

least_fb_hotels = train.groupby('Hotelid')['Sentiment'].value_counts().sort_values(ascending=True).head(10)

least_fb_hotels.plot(kind="barh",color="grey",title="Least 10 feed back given hotels ",legend=True,figsize=(10,10))

_=plt.xlabel('count')

_=plt.ylabel('HotelID')

plt.show()
# As we can observe the trend in summer holidays i.e from April to August are more when compared to all the other months in all the years

###   On existing Hotels Data



exist['AverageOverallRatingOfHotel']=exist['AverageOverallRatingOfHotel'].astype("float64")



#Worst Hotels

worst_hotels =exist.groupby('Hotelid')['AverageOverallRatingOfHotel'].mean().sort_values(ascending=True).head(5)

worst_hotels.plot(kind="barh",color="green",title="Worst Hotels ")

_=plt.xlabel('AverageOverallRatingOfHotel')

_=plt.ylabel('HotelID')

plt.show()



#Best Hotels

best_hotels = exist.groupby('Hotelid')['AverageOverallRatingOfHotel'].mean().sort_values(ascending=False).head(5)

best_hotels.plot(kind="barh",color = "pink",title="Best Hotels ")

_=plt.xlabel('AverageOverallRatingOfHotel')

_=plt.ylabel('HotelID')

plt.show()





## Preprocessing On Train data and Test Data 



train["reviewtext"]=train["reviewtext"].apply(lambda x:re.sub("[^A-Za-z]", " ", x.strip()))

test["reviewtext"]=test["reviewtext"].apply(lambda x:re.sub("[^A-Za-z]", " ", x.strip()))



## Lower case

train['reviewtext'] = train['reviewtext'].apply(lambda x: " ".join(x.lower() for x in x.split()))

test['reviewtext'] = test['reviewtext'].apply(lambda x: " ".join(x.lower() for x in x.split()))
### Numbers Removal

train['reviewtext'] = train['reviewtext'].str.replace('[\d]', '')

test['reviewtext'] = test['reviewtext'].str.replace('[\d]', '')
## ### Punctuation Removal

train['reviewtext'] = train['reviewtext'].str.replace('[^\w\s]','')

test['reviewtext'] = test['reviewtext'].str.replace('[^\w\s]','')
#### Stop words Removal



stop = stopwords.words('english')

train['reviewtext'] = train['reviewtext'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

test['reviewtext'] = test['reviewtext'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
### Common word removal

## On trainData

freq = pd.Series(' '.join(train['reviewtext']).split()).value_counts()[:10]

freq

freq = list(freq.index)

train['reviewtext'] = train['reviewtext'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

train['reviewtext'].head()

## On test Data



freq1 = pd.Series(' '.join(test['reviewtext']).split()).value_counts()[:10]

freq1
freq1 = list(freq1.index)

test['reviewtext'] = test['reviewtext'].apply(lambda x: " ".join(x for x in x.split() if x not in freq1))

test['reviewtext'].head()
### Rare words removal

### On train Data

freq = pd.Series(' '.join(train['reviewtext']).split()).value_counts()[-10:]

freq
freq = list(freq.index)

train['reviewtext'] = train['reviewtext'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

train['reviewtext'].head()
## On test Data

freq1 = pd.Series(' '.join(test['reviewtext']).split()).value_counts()[-10:]

freq1
freq1 = list(freq1.index)

test['reviewtext'] = test['reviewtext'].apply(lambda x: " ".join(x for x in x.split() if x not in freq1))

test['reviewtext'].head()
### Stemming

st = PorterStemmer()

train["reviewtext"] = train['reviewtext'].apply(lambda x: " ".join([st.stem(word)

                                                                    for word in x.split()]))

test["reviewtext"] = test['reviewtext'].apply(lambda x: " ".join([st.stem(word)

                                                                  for word in x.split()]))
### Lemmatization

Lem = WordNetLemmatizer()



train["reviewtext"] = train['reviewtext'].apply(lambda x: " ".join([Lem.lemmatize(word)

                                                                    for word in x.split()]))



test["reviewtext"] = test['reviewtext'].apply(lambda x: " ".join([Lem.lemmatize(word)

                                                                  for word in x.split()]))
### Converting the text data into list

Text_data1 = train['reviewtext'].tolist() 

Text_data2 = test['reviewtext'].tolist()
for i in range(len(Text_data1)):     

    Text_data1[i]=re.sub(r'\s+', ' ', Text_data1[i]) #Removing more than one white spaces in the sentence     

    Text_data1[i]=re.sub('[\d]', ' ',Text_data1[i]) 

    Text_data1[i]=re.sub(r'[^\x00-\x7F]+',' ',Text_data1[i])

    

Text_data1
for i in range(len(Text_data2)):     

    Text_data2[i]=re.sub(r'\s+', ' ', Text_data2[i])     

    Text_data2[i]=re.sub('[\d]', ' ',Text_data2[i]) 

    Text_data2[i]=re.sub(r'[^\x00-\x7F]+','',Text_data2[i])



    Text_data2
## Splitting the Data

X_train,X_test,y_train,y_test = train_test_split(Text_data1,train['Sentiment'],test_size=0.3,random_state=124) 
print(y_train.value_counts())

print(y_test.value_counts())
## Creating a Tfidf Matrix

tfidf_transformer = TfidfVectorizer(ngram_range=(1,1),stop_words='english',max_features=350)

X_train_tfidf = tfidf_transformer.fit_transform(X_train)

print(X_train_tfidf.shape)

# Get the tfidf matrix for test documents

X_test_tfidf = tfidf_transformer.transform(X_test) 

print(X_test_tfidf.shape)



test_tfidf=tfidf_transformer.transform(Text_data2)
## MODEL BUILDING



## Navie Bayes model



nb_clf = MultinomialNB().fit(X_train_tfidf,y_train) 



pred_train = nb_clf.predict(X_train_tfidf)  

pred_test = nb_clf.predict(X_test_tfidf) #predict on test data 

print(accuracy_score(y_train,pred_train)) 

print(accuracy_score(y_test,pred_test))
## LOGISTIC REGRESSION

logmod=LogisticRegression()



logmod.fit(X_train_tfidf,y_train)



pred_train_log = logmod.predict(X_train_tfidf)

pred_test_log = logmod.predict(X_test_tfidf)

test_pred = logmod.predict(test_tfidf)



print("Accuracy on train is:",accuracy_score(y_train,pred_train_log))

print("Accuracy on test is:",accuracy_score(y_test,pred_test_log))

## Using Grid search Model



param_grid={

    "C":[10,20],

    "max_iter":[100,150]

}

param_grid_model = GridSearchCV(estimator = logmod, param_grid = param_grid, cv =5,n_jobs=-1)
param_grid_model.fit(X_train_tfidf,y_train)
y_preds_lr_train = param_grid_model.best_estimator_.predict(X_train_tfidf)

y_preds_lr_test = param_grid_model.best_estimator_.predict(X_test_tfidf)

test_pred_gd = param_grid_model.best_estimator_.predict(test_tfidf)
print("Accuracy on train is:",accuracy_score(y_train,y_preds_lr_train))

print("Accuracy on test is:",accuracy_score(y_test,y_preds_lr_test))
### submission file 

test_Id = test.Reviewid.copy()

submission = pd.DataFrame()

submission['Id'] = test_Id
submission['Sentiment'] = test_pred_gd

submission.head()

submission.to_csv('final_subsmission.csv', index=False)
### These are the other models

## DECISION TREES

dtc = DecisionTreeClassifier()

dtc.fit(X_train_tfidf,y_train)



pred_train_dt = dtc.predict(X_train_tfidf)

pred_test_dt = dtc.predict(X_test_tfidf)



print("Accuracy on train is:",accuracy_score(y_train,pred_train_dt))



print("Accuracy on test is:",accuracy_score(y_test,pred_test_dt))



## RANDOM FOREST



rfc=RandomForestClassifier()

rfc.fit(X=X_train_tfidf,y=y_train)



pred_train_rf=rfc.predict(X_train_tfidf)



pred_test_rf=rfc.predict(X_test_tfidf)



print("Accuracy on train is:",accuracy_score(y_train , pred_train_rf))



print("Accuracy on test is:",accuracy_score(y_test , pred_test_rf))
# ADA BOOST

Adaboost_model = AdaBoostClassifier(

    DecisionTreeClassifier(max_depth=3),

    n_estimators = 600,

    learning_rate = 1)

# Train model

%time Adaboost_model.fit(X_train_tfidf, y_train)



# Predictions and Evaluations 

pred_train_Ada = Adaboost_model.predict(X_train_tfidf)



pred_test_Ada = Adaboost_model.predict(X_test_tfidf)



print("Accuracy on train is:",accuracy_score(y_train,pred_train_Ada ))



print("Accuracy on test is:",accuracy_score(y_test,pred_test_Ada ))
### XG BOOST

XGB_model = XGBClassifier()

# training the xgboost classifier

%time XGB_model.fit(X_train_tfidf, y_train)



pred_train_XG = XGB_model.predict(X_train_tfidf)



pred_test_XG = XGB_model.predict(X_test_tfidf)



print("Accuracy on train is:",accuracy_score(y_train,pred_train_XG ))



print("Accuracy on test is:",accuracy_score(y_test,pred_test_XG ))
# TASK 3

### Preprocessing on Existing Hotels Data and New Hotels Data

# Verifying the NULL values

print("__________On existing hotels data _______________")

print(exist.isnull().sum().sort_values(ascending=True))

print("__________On New hotels data _______________")

print(New.isnull().sum().sort_values(ascending=True))
#### Separating the Averagepricing Column

exist=exist.join(exist['AveragePricing'].str.split('$', 1, expand=True).rename(columns={ 1:'AvgPricing'}))

exist = exist.drop(0,axis=1)



New=New.join(New['AveragePricing'].str.split('$', 1, expand=True).rename(columns={ 1:'AvgPricing'}))

New = New.drop(0,axis=1)

#### Creating a LengthOfReview Column

exist['LengthofReview'] = exist['reviewtext'].apply(lambda x: len(str(x).split(" ")))



New['LengthofReview'] = New['reviewtext'].apply(lambda x: len(str(x).split(" ")))
#### Dropping the unnecessary columns

exist = exist.drop(["Date","userid","reviewtext","AveragePricing"],axis=1)

New = New.drop(["Date","userid","reviewtext","AveragePricing"],axis=1)
#### Verifying the datatypes

print("___________Existing hotels Dtypes____________")

print(exist.dtypes)



print("___________New hotels Dtypes____________")

print(New.dtypes)
#### Datatypes conversion

exist["AvgPricing"] = exist["AvgPricing"].astype("float64")

New["AvgPricing"] = New["AvgPricing"].astype("float64")
#### Verifying the unique values count on the data

print("___________Unique values count on existing hotels___________")

unique_counts = pd.DataFrame.from_records([(col, exist[col].nunique()) for col in exist.columns],

                          columns=['Column_Name', 'Num_Unique']).sort_values(by=['Num_Unique'])

print(unique_counts)





print("___________Unique values count on New hotels___________")

unique_counts = pd.DataFrame.from_records([(col, New[col].nunique()) for col in New.columns],

                          columns=['Column_Name', 'Num_Unique']).sort_values(by=['Num_Unique'])

print(unique_counts)

### Feature Engineering

#### On Existing hotels Data

MR = exist.groupby(['Hotelid']).mean().reset_index()



MR.drop(['NoOfReaders',"HelpfulToNoOfreaders"],axis=1, inplace=True)



MR1 = pd.DataFrame(exist.groupby('Hotelid')['NoOfReaders',"HelpfulToNoOfreaders"].sum().reset_index())
Existing_Hotels = pd.merge(MR1,MR,on=['Hotelid','Hotelid'])

Existing_Hotels.dtypes
#### On New Hotels Data

MR2 = New.groupby(['Hotelid']).mean().reset_index()



MR2.drop(['NoOfReaders',"HelpfulToNoOfreaders"],axis=1, inplace=True)



MR3 = pd.DataFrame(New.groupby('Hotelid')['NoOfReaders',"HelpfulToNoOfreaders"].sum().reset_index())
New_Hotels = pd.merge(MR2,MR3,on=['Hotelid','Hotelid'])

New_Hotels.dtypes
#### Drop the Hotelid Column

Existing_Hotels = Existing_Hotels.drop("Hotelid",axis=1)

New_Hotels = New_Hotels.drop("Hotelid",axis=1)
# CLUSTERING

#### Standardizing the data

std = StandardScaler()

std.fit(Existing_Hotels)

X_train_std= std.transform(Existing_Hotels)

X_test_std= std.transform(New_Hotels)
#### KMEANS Model

kmeans = KMeans(n_clusters=2,random_state=99999)



kmeans = kmeans.fit(X_train_std)



labels_train = kmeans.predict(X_train_std)

labels_test = kmeans.predict(X_test_std)



# Centroid values

centroids = kmeans.cluster_centers_
## Checking the values

centroids
labels_train
labels_test
print(kmeans.cluster_centers_)

print(kmeans.inertia_)
####  Checking For Various Values of K

wss= {}

for k in range(1, 10):

    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(X_train_std)

    clusters = kmeans.labels_

    wss[k] = kmeans.inertia_
plt.figure()

plt.plot(list(wss.keys()), list(wss.values()))

plt.xlabel("Number of cluster")

plt.ylabel("WSS")

plt.show()
wss
#### KMEANS Model with "K" value

kmeans = KMeans(n_clusters=5,random_state=9999)



kmeans.fit(Existing_Hotels)



Existing_Hotels["cluster"] = kmeans.predict(Existing_Hotels)

New_Hotels["cluster"] = kmeans.predict(New_Hotels)



# Centroid values

centroids = kmeans.cluster_centers_
#### Checking for cluster Stability

indices=Existing_Hotels.sample(frac=0.8,random_state=123).index

print(indices)
#### Subsetting 80% of train data

Alpha = Existing_Hotels.loc[indices,:]

Alpha.shape
kmeans = KMeans(n_clusters=5,random_state=45)

kmeans2=kmeans.fit(Alpha)

print(len(kmeans2.labels_))

Alpha['cluster']=kmeans2.labels_
g1=Existing_Hotels.loc[indices,'cluster']

g2=Alpha.cluster
#### Cluster Stability

adjusted_rand_score(g1,g2)
print("__________On user ratings data _______________")

print(ratings.isnull().sum().sort_values(ascending=True))
#### Verifying the unique counts on the data

print("___________Unique values count on existing hotels___________")

unique_counts = pd.DataFrame.from_records([(col, ratings[col].nunique()) for col in ratings.columns],

                          columns=['Column_Name', 'Num_Unique']).sort_values(by=['Num_Unique'])

print(unique_counts)
### Feature Engineering

# calculating the mean ratings for each hotel

  

rating = pd.DataFrame(ratings.groupby('Hotelid')['OverallRating'].count())  

rating["ratings_count"] = rating["OverallRating"]

rating=rating.drop("OverallRating",axis=1)
# sorting based on count of ratings that each hotelId got  

  

rating.sort_values('ratings_count', ascending=False).head()
## Creating a Pivot table

# Preparing data table for analysis  

  

ratings_pivot = ratings.pivot_table(values='OverallRating', index='userid', columns='Hotelid')  

  

ratings_pivot.head()
# we can calculate the correlation of the Hotelid column with all others and for this we can use the corrwith function

X = ratings_pivot["hotel_510"]  
### Checking correlation for a Hotelid

Corr = pd.DataFrame(ratings_pivot.corrwith(X)) 

Corr.rename(columns={0: 'corr'}, inplace=True)

Corr.head()
#### Joining the two required columns

Final_summary = Corr.join(rating)
# These are the most similar Hotels  

  

Final_summary.sort_values('corr', ascending=False).head(10)
corr_matrix = ratings_pivot.corr(method="pearson")

corr_matrix.head()
ratings_pivot.iloc[3].dropna().head()
## We create now the list of all Hotels with all correlations multiplied by ratings (integers from 1 to 5). 

user_corr = pd.Series()



userid=3



for Hotelid in ratings_pivot.iloc[userid].dropna().index:

    corr_list = corr_matrix[Hotelid].dropna()*ratings_pivot.iloc[userid][Hotelid]

    user_corr = user_corr.append(corr_list)

    


## We make the groupby in order to not have duplicate Hotels and we also sum their rating: *

user_corr = user_corr.groupby(user_corr.index).sum()

user_corr.head()
## We now create a list of Hotels Visited to drop (if contained in our Series) *

Hotels_list = []

for i in range(len(ratings_pivot.iloc[userid].dropna().index)):

    if ratings_pivot.iloc[userid].dropna().index[i] in user_corr:

        Hotels_list.append( ratings_pivot.iloc[userid].dropna().index[i])

    else:

        pass



user_corr = user_corr.drop(Hotels_list)
print("\n These are the hotels which you have visited \n")

for i in ratings_pivot.iloc[userid].dropna().index:

    print(i)

print("\n We would suggest you to try these 5 Hotels: \n")

for i in user_corr.sort_values(ascending=False).index[:5]:

    print(i)