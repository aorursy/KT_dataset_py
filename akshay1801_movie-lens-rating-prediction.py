#Importing required packages 

import pandas as pd

import numpy as np

import warnings

import matplotlib.pyplot as plt

from scipy import stats

from matplotlib import style

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import Normalizer

from preprocess import create_cat, split_, convert
#Loading the data as pandas DataFrame

movies_df = pd.read_csv("/kaggle/input/movie-lens-dataset/movies.dat",delimiter="::",names=["MovieID","Title","Genre"], engine='python')

rating_df = pd.read_csv("/kaggle/input/movie-lens-dataset/ratings.dat",delimiter="::",names=["ID","MovieID","Rating","Timestamp"],engine='python')

user_df = pd.read_csv("/kaggle/input/movie-lens-dataset/users.dat",delimiter="::",names=["ID","Gender","Age","Occupation","Zip-code"],engine='python')
#Checking for missing entries in the data

print(f"""Missing entries for the following:  

    Movies Data:\n{movies_df.isna().sum()},

    Rating data:\n{rating_df.isna().sum()},

    User Data\n{user_df.isna().sum()}""")
# Combining all the Data into 1 Data Frame

new_df = pd.merge(rating_df,user_df,on="ID")

new_df.isna().sum() # Checking if files were merged properly
new_df.head() # Checking the columns in the new df
# creating a final df with all the data by adding the data from the final file 

df = pd.merge(new_df,movies_df,on="MovieID")
#checking the final df

print(df.isna().sum())

df.head()
#Further Exploring the distribution of Age in Users of MovieLens

style.use("fivethirtyeight")

%matplotlib inline

Age = df.Age

bins_ = np.arange(0,65,2).tolist()

plt.figure(figsize=(10,10))

plt.hist(Age,bins=bins_)

plt.xticks(bins_)

plt.xlabel("Age Range")

plt.ylabel("Frequency")
pd.unique(df.Age)

age_range = ["Under 18","18-24","25-34","35-44","45-49","50-55","56+"]

values = [1,18,25,35,45,50,56]

dict_ = dict(zip(values,age_range))
hist_series = []

for i in df.Age:

    hist_series.append(dict_[i])

hist_series  = pd.Series(hist_series)   

# creating a series with actual values instead of dummy variables for better visualiztion
style.use("fivethirtyeight")

%matplotlib inline

bins_ = np.arange(15,65,2).tolist()

plt.figure(figsize=(10,10))

plt.bar(age_range,hist_series.value_counts().sort_index(),color="c")

#plt.xticks(age_range)

plt.xlabel("Age Range")

plt.ylabel("Frequency")
#User Ratings of the Toy Story Movies

toy_story_r = df[df['Title'].str.contains("Toy Story")]

toy_story_r[["ID","Title","Rating"]]
group =toy_story_r.groupby("Title")

round(group.Rating.mean(),2) # average Rating for both the Toy story movies
group = df.groupby("Title")

round(group.Rating.mean().sort_values(ascending=False)[:25],2) # Top 25 Movies by Avearge Rating
#All the movies reviewed by ID 2696

df[["ID","Title","Rating"]][df.ID == 2696]
# Checking all the unique values in the genres

pd.unique(df.Genre)
# finding out the actual unique genres 



Genre = df.Genre.apply(split_) #spliting the genres 

from itertools import chain

list_ = list(chain(*Genre)) # combining all the list and strings in the series to one big list to find out the set(unqiue values )

set(list_)
# Code for above mentioned process 

columns = list(set(list_))

genre_df = pd.DataFrame(0, index=np.arange(len(df)), columns=columns)  # creating a df of all the categories filled with 0



for i in genre_df.columns:

    genre_df[i]= df.Genre.apply(convert,args = ([i]))



df1 = pd.merge_asof(df,genre_df,left_index=True,right_index=True) # adding the new one - hot columns to the original dataframe
df1.head()
# Exploratory analysis to find out other useful features 

style.use("seaborn")

groupby = df.groupby("Age")

age_r =groupby.Rating.mean() # mean rating by age

#age_ = list(pd.unique(Age_df.Age))

Average_Rating = {"Male":np.mean(df.Rating[df.Gender=="M"]),"Female":np.mean(df.Rating[df.Gender=="F"])} # a dict with average score for gender

groupby = df.groupby("MovieID")

MovieID = groupby.Rating.mean()[:500]

mid = list(pd.unique(df.MovieID.sort_values()))[:500]



fig, axs = plt.subplots(2,2,figsize = (20,10))

axs[0,0].set_title("Number Males and Females in the Data")

axs[0,0].bar(x = ["Male","Female"],height=df.Gender.value_counts()) # plotting the number of Male and Female customers

axs[0,1].set_title("Average Rating by Gender")

axs[0,1].bar(Average_Rating.keys(),Average_Rating.values())

axs[1,0].set_title("Average Rating by Age")

axs[1,0].bar(x =age_range,height=age_r,color="c")

axs[1,1].set_title("Average Rating to Movie ID")

axs[1,1].scatter(x=mid,y = MovieID)
df.corr()
groupby.Rating.mean()
# getting dummy variables for gender columns 

df1 = pd.merge(df1,pd.get_dummies(df.Gender),left_index=True,right_index=True)
new_df = df1.drop(["Title","Gender","Genre"],axis = 1)

new_df["Zip-code"] = create_cat(new_df["Zip-code"]) # converting Zip codes to categories

new_df.head()
ID_list = list(set(new_df.ID.values.tolist())) # Creating a list of IDs

df_list = []

for i in ID_list: 

    exp = new_df[new_df.ID == i] # dividing the the dataframe by ID 

    exp = exp.sort_values(["Timestamp"])  # sorting the values by time

    one = exp.drop(["Occupation","Zip-code","F","M","Timestamp","Age"],axis=1).shift(1) # last watched movie genre

    two = exp.drop(["Occupation","Zip-code","F","M","Timestamp","Age"],axis=1).shift(2) #last to last watched movie genre

    exp = pd.concat([exp,one,two],axis=1)

    df_list.append(exp)
final_df = pd.concat(df_list) # putting everything back together in a single dataframe
final_df.columns # all the columns include
final_df.head()
final_df["label"] = final_df.Rating.iloc[:,0] # the rating of the current movie

final_df["Rating_1"] = final_df.Rating.iloc[:,1] # }

final_df["Rating_2"] = final_df.Rating.iloc[:,2] # } The ratings given to the past two movies
final_df.drop(["Timestamp","ID","MovieID"],axis=1,inplace=True) # these are not required anymore
final_df.dropna(inplace=True) # dropping the nans created during the shifting of the dataframe
len(final_df)
final_df = final_df.sample(frac=1) # random shuffling the data
final_df.label
final_df.drop(["Rating"],axis=1,inplace=True) # dropping the previous rating columns which have been replaced with 

                                              # columns "label","Ratings_1","Rating_abs2"
X = final_df.drop(["label"],axis=1).values # features

std = Normalizer()

X = std.fit_transform(X) # Normalising the data



y = final_df["label"].values # label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=11) #standard split
X_train[1]
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=20,n_jobs=-1) 

forest.fit(X, y)
print(f"Training Accuracy: {forest.score(X_train,y_train)}")

print(f"Testing Accuracy: {forest.score(X_test,y_test)}")