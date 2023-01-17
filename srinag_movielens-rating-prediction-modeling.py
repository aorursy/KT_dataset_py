import pandas as pd



#Input movies dataset

movies = pd.read_csv("../input/movies.dat", sep="::", names=['MovieID', 'Title', 'Genres'] )



#Read the sample movies dataset

movies.head()
#Input ratings dataset

ratings = pd.read_csv("../input/ratings.dat", sep="::", names=['UserID', 'MovieID', 'Rating', 'Timestamp'] )



#Read the sample ratings dataset

ratings.head()
#Input users dataset

users = pd.read_csv("../input/users.dat", sep="::", names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'] )



#Read the sample users dataset

users.head()
#Merge the ratings and users with movieID and UserID

ratings_user = pd.merge(ratings,users, on=['UserID'])

ratings_movie = pd.merge(ratings,movies, on=['MovieID'])



master_data = pd.merge(ratings_user,ratings_movie,

                       on=['UserID', 'MovieID', 'Rating'])[['MovieID', 'Title', 'UserID', 'Age', 'Gender', 'Occupation', "Rating"]]



master_data.head()
#User age distribution

import matplotlib.pyplot as plt



users['Age'].hist(bins=50)

plt.xlabel('Age')

plt.ylabel('Population')

plt.show
#User rating of the movie “Toy Story”



res = master_data[master_data.Title == "Toy Story (1995)"]



plt.plot(res.groupby("Age")["MovieID"].count(),'--bo')

res.groupby("Age")["MovieID"].count()
#Top 25 movies by viewership rating



res = master_data.groupby("Title").size().sort_values(ascending=False)[:25]

plt.ylabel("Title")

plt.xlabel("Viewership Count")

res.plot(kind="barh")

#res

#Find the ratings for all the movies reviewed by for a particular user of user id = 2696



res = master_data[master_data.UserID == 2696]



plt.scatter(y=res.Title, x=res.Rating)



res

#Feature Engineering



val = movies.Genres.str.split("|")



res_col = []

for v in val:

    for i in v:

        if i not in res_col:

            res_col.append(i)



res_col.append("Gender")

res_col.append("Age")

res_col.append("Rating")



df = pd.DataFrame(columns=res_col)



res = master_data.merge(movies, on = ['MovieID'], how="left")[["Genres","Rating","Gender", "Age"]]



for index, row in res.head(20000).iterrows():

    tmp = row.Genres.split("|") 

    

    for i in tmp:

       # print(i)

        df.loc[index,i] = 1

        df.loc[index,"Gender"] = res.loc[index,"Gender"]

        df.loc[index,"Age"] = res.loc[index,"Age"]

        df.loc[index,"Rating"] = res.loc[index,"Rating"]

         

#         var = res.loc[index, "Rating"]

#         if var == 1:

#             df.loc[index,"Rating"] = "one" 

#         elif var == 2:

#             df.loc[index,"Rating"] = "two"

#         elif var == 3:

#             df.loc[index,"Rating"] = "three"

#         elif var == 4:

#             df.loc[index,"Rating"] = "four"

#         else:

#             df.loc[index,"Rating"] = "five"

     

    df.loc[index,df.columns[~df.columns.isin(tmp+["Gender","Rating","Age"])]] = 0



df.head()

    



#df.loc[i,"Animation"] = 1



#df
from sklearn import datasets 

from sklearn.metrics import confusion_matrix 

from sklearn.model_selection import train_test_split 

from sklearn.preprocessing import LabelEncoder



X = df[df.columns[~df.columns.isin(["Rating"])]]

y = df.Rating



# dividing X, y into train and test data 

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0) 



number = LabelEncoder()

X_train.Gender = number.fit_transform(X_train["Gender"].astype("str"))

X_test.Gender = number.fit_transform(X_test["Gender"].astype("str"))

y_train = number.fit_transform(y_train.astype("int"))

y_test = number.fit_transform(y_test.astype("int"))

#SVM



from sklearn.svm import SVC 

svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 

svm_predictions = svm_model_linear.predict(X_test) 

  

# model accuracy for X_test   

accuracy = svm_model_linear.score(X_test, y_test) 

  

# creating a confusion matrix 

cm = confusion_matrix(y_test, svm_predictions) 

accuracy

#cm
#KNN



from sklearn.neighbors import KNeighborsClassifier 

knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, y_train) 

  

# accuracy on X_test 

accuracy = knn.score(X_test, y_test) 

  

# creating a confusion matrix 

knn_predictions = knn.predict(X_test)  

cm = confusion_matrix(y_test, knn_predictions) 



accuracy

#Naive Bayes classifier 



from sklearn.naive_bayes import GaussianNB 

gnb = GaussianNB().fit(X_train, y_train) 

gnb_predictions = gnb.predict(X_test) 

  

# accuracy on X_test 

accuracy = gnb.score(X_test, y_test)  

  

# creating a confusion matrix 

cm = confusion_matrix(y_test, gnb_predictions) 



accuracy


