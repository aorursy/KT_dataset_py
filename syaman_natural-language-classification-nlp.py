
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#if we don't use encoding="latin1", program will give an error 
data=pd.read_csv("../input/twitter-gender/gender-classifier.csv",encoding="latin1")

#we will prepare our data
#with pd.concat, we can combine our datas which can be series or dataframe 
data=pd.concat([data.gender,data.description],axis=1)

#we will clean Nan values from our data with dropna 
data.dropna(axis=0,inplace=True)

#we will change gender type male=0,female=1
data["gender"]=[1 if i=="female" else 0 for i in data["gender"]]
data.head()

#data cleaning
#regular expression
import re

#first we will clean one data from our data to make an example to clean data

example_description=data.description[4] #4th row data

#with re.sub , find character :),%,/,# (except from a to z and from A to Z in alphabet) and change with space
clean=re.sub("[^a-zA-Z]"," ",example_description)
clean=clean.lower() # our LETTER words will be changed by lower()

#now we have just english lower words
clean
#stopwords (irrelevent words)
import nltk # natural language tool kit library
nltk.download("stopwords") # for this download we need internet. if we dont have an internet connection 
# we can have an error

from nltk.corpus import stopwords

#clean=clean.split()# with split every word will be value and stored in a list, we can use word_tokenize too
clean=nltk.word_tokenize(clean) #with tokenize we can split like this words shouldn't to should and n't

clean=[word for word in clean if not word in set(stopwords.word("english"))]

import nltk as nlp
description_list=[]

for description in data.description:
    description=re.sub("[^a-zA-Z]"," ",description)
    description=description.lower() #change from capital to low words
    #description=[word for word in description if not word in set(stopwords.words("english"))]
    description=nltk.word_tokenize(description)
    lemma=nlp.WordNetLemmatizer()
    description=[ lemma.lemmatize(word) for word in description]
    description=" ".join(description)
    description_list.append(description)
description_list
#to create bag of words
from sklearn.feature_extraction.text import CountVectorizer

max_features=5000
count_vectorizer=CountVectorizer(max_features=max_features,stop_words="english")

sparce_matrix=count_vectorizer.fit_transform(description_list).toarray()
print("the most using  {} words: {}".format(max_features,count_vectorizer.get_feature_names()))

from sklearn.model_selection import train_test_split
x=sparce_matrix
y=data["gender"] # male or female

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=21)

#Naive bayes
from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()
nb.fit(xtrain,ytrain)

#prediction
y_prediction=nb.predict(xtest).reshape(-1,1)

print("accuracy : {}".format(nb.score(y_prediction,ytest)))

#we will create our data from sklearn library
from sklearn.datasets import load_iris

iris=load_iris()
data=iris.data
feature_names=iris.feature_names
y=iris.target

df=pd.DataFrame(data,columns=feature_names)
df["class"]=y
x=data


#PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=2,whiten=True) #whitten = normalize, we will have 2-size

pca.fit(x)
x_pca=pca.transform(x) # it will trasnform dimension from to 2

print("variance ratio: {}".format(pca.explained_variance_ratio_))

#we changed our data's dimensiton but we still have the same features( we will see the varience=0.9776..)
print("sum: {}".format(sum(pca.explained_variance_ratio_)))

#PCA visualization
df["p1"]=x_pca[:,0] #principal component
df["p2"]=x_pca[:,1] # second component

color=["red","green","blue"]
for each in range(3):
    plt.scatter(df.p1[df["class"]==each],df.p2[df["class"]==each],color=color[each],label=iris.target_names[each])

plt.legend()
plt.xlabel("p1")
plt.ylabel("p2")
plt.show()
    
    
    
# we will use iris data set
iris=load_iris()
x=iris.data
y=iris.target

#we will make normalization
x=(x-np.min(x))/(np.max(x)-np.min(x))

#we will make train and test split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)

#we will use knn algoritm
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)

#we will use cross validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=knn,X=xtrain,y=ytrain,cv=10)

print("average accuracy: {}".format(np.mean(accuracies)))
print("Standart deviation (std): {}".format(np.std(accuracies)))


knn.fit(xtrain,ytrain)

print("test acuuracy: {}".format(knn.score(xtest,ytest)))

# we will find best k number
#grid search cross validation
from sklearn.model_selection import GridSearchCV

grid={"n_neighbors":np.arange(1,50)}
knn =KNeighborsClassifier()
knn_cros_validation=GridSearchCV(knn,grid,cv=10)# grid search cross validation
knn_cros_validation.fit(x,y)

print("tuned hyperparameter K: {}".format(knn_cros_validation.best_params_))
print("Best Score: {}".format(knn_cros_validation.best_score_))
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

iris=load_iris()
x=iris.data
y=iris.target

x=x[:100,:]
y=y[:100]

#we will make normalization
x=(x-np.min(x))/(np.max(x)-np.min(x))

#we will make train and test split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25)


grid = {"C":np.logspace(-3,3,7),"penalty":["l1","l2"]}

logreg=LogisticRegression()
logreg_cv=GridSearchCV(logreg,grid,cv=10)
logreg_cv.fit(xtrain,ytrain)

print("best parameters(hyperparameters: )",logreg_cv.best_params_)
print("accuracy: {}".format(logreg_cv.best_score_))
logreg2=LogisticRegression(C=1,penalty="l2")
logreg2.fit(xtrain,ytrain)
print("score",logreg2.score(xtest,ytest))
movie=pd.read_csv("../input/movielens-20m-dataset/movie.csv")
movie.columns
movie=movie.loc[:,["movieId","title"]]
movie.head()
rating=pd.read_csv("../input/movielens-20m-dataset/rating.csv")
rating.head()
rating=rating.loc[:,["userId","movieId","rating"]]
rating.head()
#we can compile (merge) movie and rating data
data=pd.merge(movie,rating)

data.head()
#to make rows user and columns movies, we need to use pivot table
#we will chose first 1000000 data
data=data.loc[:1000000,:]
pivot_table=data.pivot_table(index=["userId"],columns=["title"],values="rating")
pivot_table.head()
movie_watched=pivot_table["Babe (1995)"]
similarity_with_other_movies=pivot_table.corrwith(movie_watched) # correlation
similarity_with_other_movies=similarity_with_other_movies.sort_values(ascending=False)
similarity_with_other_movies.head()