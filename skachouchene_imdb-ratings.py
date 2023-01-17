import scipy.stats as stats

import numpy as np    

import pandas as pd   

from matplotlib import pyplot as plt 

import seaborn as sns  

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_absolute_error



from sklearn.datasets import make_regression

from sklearn.model_selection import train_test_split



%matplotlib inline
imdb = pd.read_csv("../input/imdb.csv", encoding="ISO-8859-1")
print(imdb.shape)

imdb.columns = imdb.columns.str.strip().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

#Replaced spaces with underscores

imdb.head()
#This is a small correlation heatmap to exclude some numerical variables

numeric_variables_names = [imdb.columns[i] for i,x in enumerate(imdb.dtypes) if str(x) in ['float64','int64']]



res = imdb[numeric_variables_names].corr()



C = np.matrix(res)

print(type(C), C.shape)

print(C.min(),C.max())
sns.heatmap(C, center=0, annot=False, xticklabels=numeric_variables_names, yticklabels=numeric_variables_names)
target=imdb["Your_Rating"]

features=imdb[["Year","IMDb_Rating"]]
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
X_test.shape

model = LinearRegression()

model.fit(X_train, y_train)
ynew = model.predict(X_test)
val_mae = mean_absolute_error(ynew, y_test)

print("The accuracy of this first model is")

print(abs((1-(val_mae/10))*100),"%")

imdb.Genres.value_counts()

#Genres are: Drama, Comedy, Romance, Mystery, Thriller, Horror, Music, Fantasy, Sci_Fi, Documentary, Action, Crime, War, 

#Biography, Western, History

L=imdb.Genres.tolist()
m=imdb.shape[0]

Genre_list=[[0 for j in range(m)] for i in range(6)]

#Genres are: Drama, Comedy, Romance, Mystery, Thriller, Horror, Music, Fantasy, Sci_Fi, Documentary, Action, Crime, War, 

#Biography, Western, History, Adventure

d={'Drama':10, 'Comedy':20, 'Romance':30, 'Mystery':40, 'Thriller':50, 'Horror':60, 'Music':70, 'Musical':70, 'Fantasy':80, 'Sci-Fi':90, 'Documentary':100, 'Action':110, 'Crime':120, 'War':130, 'Biography':140, 'Western':150, 'History':160,'Adventure':180,'Animation':190,'Family':200,'Talk-Show':0,'News':0}

for i in range(len(L)):

    ch=L[i]

    ch.replace(" ","")

    chl=ch.split(", ")

    for j in range(len(chl)):

        val=chl[j]

        Genre_list[j][i]=d[val]

#Creating new columns for genre classification and numerisation, each column has a number that corresponds to a genre

#If there are less than 6 genres, the empty columns take 0 as a value

Genre1=np.array(Genre_list[0])

Genre2=np.array(Genre_list[1])

Genre3=np.array(Genre_list[2])

Genre4=np.array(Genre_list[3])

Genre5=np.array(Genre_list[4])

Genre6=np.array(Genre_list[5])

imdb['Genre1']=Genre1

imdb['Genre2']=Genre2

imdb['Genre3']=Genre3

imdb['Genre4']=Genre4

imdb['Genre5']=Genre5

imdb['Genre6']=Genre6
target=imdb["Your_Rating"]

features=imdb[["Year","IMDb_Rating","Genre1","Genre2","Genre3","Genre4","Genre5","Genre6",]]
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

model = LinearRegression()

model.fit(X_train, y_train)

ynew = model.predict(X_test)
val_mae = mean_absolute_error(ynew, y_test)

print("The accuracy of this second model is")

print(abs((1-(val_mae/10))*100),"%")



Genre_list=[[0 for j in range(206)] for i in range(6)]

#Genres are: Drama, Comedy, Romance, Mystery, Thriller, Horror, Music, Fantasy, Sci_Fi, Documentary, Action, Crime, War, 

#Biography, Western, History, Adventure

d={'Drama':1, 'Comedy':3, 'Romance':1, 'Mystery':2, 'Thriller':2, 'Horror':2, 'Music':8, 'Musical':8, 'Fantasy':4, 'Sci-Fi':4, 'Documentary':5, 'Action':6, 'Crime':6, 'War':6, 'Biography':5, 'Western':7, 'History':7,'Adventure':2,'Animation':8,'Family':8,'Talk-Show':0,'News':0}

for i in range(len(L)):

    ch=L[i]

    ch.replace(" ","")

    chl=ch.split(", ")

    for j in range(len(chl)):

        val=chl[j]

        Genre_list[j][i]=d[val]

Genre1=np.array(Genre_list[0])

Genre2=np.array(Genre_list[1])

Genre3=np.array(Genre_list[2])

Genre4=np.array(Genre_list[3])

Genre5=np.array(Genre_list[4])

Genre6=np.array(Genre_list[5])

imdb['Genre1']=Genre1

imdb['Genre2']=Genre2

imdb['Genre3']=Genre3

imdb['Genre4']=Genre4

imdb['Genre5']=Genre5

imdb['Genre6']=Genre6
target=imdb["Your_Rating"]

features=imdb[["Year","IMDb_Rating","Genre1","Genre2","Genre3","Genre4","Genre5","Genre6",]]
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

model = LinearRegression()

model.fit(X_train, y_train)
val_mae = mean_absolute_error(ynew, y_test)

print("The accuracy of this third model is")

print(abs((1-(val_mae/10))*100),"% which is was less")

print("However, In a previous version where there were different and more clusters it was less")

print("I think this significant drop is because of the way I classified the Genres")
imdb.drop(imdb.index[[0,7,37,38,48,56,57,63,70,96,105,125,146,162,173,192,197,201,203]], inplace=True)

imdb[imdb.Directors.isna()]
director_list=imdb.Directors.tolist()

director_list
m=imdb.shape[0]

Dir_list=[0 for j in range(m)]

Rated_directors={'Stanley Kubrick':10,'Quentin Tarantino':10,'Yorgos Lanthimos':9,'Gaspar Noé':8,'Lars Von Trier':8,'Martin Scorsese':8,'Denis Villeneuve':8,'David Fincher':7,'Edgar Wright':9}

for i in range(len(director_list)):

    ch=director_list[i]

    if ch in Rated_directors:

        Dir_list[i]=d[val]

Dir_list

Dir_rating=np.array(Dir_list)

imdb['Dir_rating']=Dir_rating

target=imdb["Your_Rating"]

features=imdb[["Year","IMDb_Rating","Genre1","Genre2","Genre3","Genre4","Genre5","Genre6","Dir_rating"]]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

model = LinearRegression()

model.fit(X_train, y_train)

ynew = model.predict(X_test)

ynew
val_mae = mean_absolute_error(ynew, y_test)

print("The accuracy of this model is")

print(abs((1-(val_mae/10))*100),"%")

print("Adding my favourite directors increased the accuracy")
target=imdb["Your_Rating"]

features=imdb[["IMDb_Rating","Genre1","Genre2","Genre3","Genre4","Genre5","Genre6","Dir_rating"]]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

model = LinearRegression()

model.fit(X_train, y_train)

ynew = model.predict(X_test)



val_mae = mean_absolute_error(ynew, y_test)

print("The accuracy of this model is")

print(abs((1-(val_mae/10))*100),"%")
target=imdb["Your_Rating"]

features=imdb[["IMDb_Rating","Genre1","Genre2","Genre3","Genre4","Genre5","Genre6","Dir_rating"]]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

model = LogisticRegression(solver='lbfgs',multi_class='multinomial')

model.fit(X_train, y_train)

ynew = model.predict(X_test)

ynew



val_mae = mean_absolute_error(ynew, y_test)

print("The accuracy of this Logistic Regression model is")

print(abs((1-(val_mae/10))*100),"%")
from sklearn import svm

clf = svm.SVC(gamma='scale')

clf.fit(X_train, y_train)

ynew=clf.predict(X_test)

val_mae = mean_absolute_error(ynew, y_test)

print("The accuracy of this SVM model is")

print(abs((1-(val_mae/10))*100), "%")
