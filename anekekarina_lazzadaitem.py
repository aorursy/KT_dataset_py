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
df_reviews = pd.read_csv("/kaggle/input/lazada-indonesian-reviews/20191002-reviews.csv")

df_items = pd.read_csv("/kaggle/input/lazada-indonesian-reviews/20191002-items.csv")



df_reviews.head(5)
df_items.head(5)
# summary df_reviews

print('\nDataset of Reviews product \n')

print('Shape dataset:', df_reviews.shape)

print('\nInfo dataset:')

print(df_reviews.info())

print('\nDescriptive Statistic:\n', df_reviews.describe())
##rearrange df_review

df_reviews = df_reviews.drop(columns=['itemId','name','originalRating','reviewContent','reviewTitle','reviewTitle','boughtDate','retrievedDate'])



df_reviews = df_reviews.reindex(columns=['category','rating','likeCount','upVotes','downVotes','helpful','clientType','relevanceScore'])



df_reviews.head(5)
##spliting data on df_items

new = df_reviews['category'].str.split("-",n=5,expand=True)

df_reviews["01"] = new[0]



df_reviews["02"] = new[1]



df_reviews["03"] = new[2]



df_reviews= df_reviews.drop(columns=['category','03','helpful'])



df_reviews = df_reviews.rename(columns={"01":"activity","02":"product"})



df_reviews = df_reviews.reindex(columns=['activity','product','rating',	'likeCount','upVotes','downVotes','clientType','relevanceScore'])



df_reviews.head(5)

#df_reviews and items checking missing value



df_reviews.isnull().sum()
df_reviews = pd.get_dummies(df_reviews)



df_reviews.head(10)
#import module

import seaborn as sns

import matplotlib.pyplot as plt



#correlation df_reviews

df_reviews = pd.get_dummies(df_reviews)

df_reviews_corr = df_reviews.corr()



#visualization

plt.figure(figsize=(20,10))

sns.heatmap(df_reviews_corr,annot=True)

plt.title("correlation df_reviews",fontsize =20)

plt.show()





#spliting df_reviews



# removing the target column rating from dataset and assigning to X_reviews

X_reviews = df_reviews.drop(['rating'], axis = 1)

# assigning the target column rating to y_reviews

y_reviews = df_reviews['rating']



# checking the shapes

print("Shape of X_reviews:", X_reviews.shape)

print("Shape of y_reviews:", y_reviews.shape)

##import library and module

from sklearn.model_selection import train_test_split



##training feature and target df_reviews

X_reviews_train, X_reviews_test, y_reviews_train, y_reviews_test = train_test_split(X_reviews,y_reviews,

                                                                            test_size=0.2,random_state=0)





print("Shape feature and target of df_reviews")

print("Shape of X_reviews_train :",X_reviews_train.shape)

print("Shape of y_reviews_train:", y_reviews_train.shape)

print("Shape of X_reviews_test:", X_reviews_test.shape)

print("Shape of y_reviews_test:", y_reviews_test.shape)

##many module



#decision tree

from sklearn.tree import DecisionTreeClassifier

#knn

from sklearn.neighbors import KNeighborsClassifier

#svm

from sklearn.svm import SVC

#naive bayes

from sklearn.naive_bayes import GaussianNB



##Make module



decision = DecisionTreeClassifier()

knn = KNeighborsClassifier()

svm = SVC(kernel='linear')

naive = GaussianNB()

##fitting 



#df_reviews

decision_reviews = decision.fit(X_reviews_train,y_reviews_train)



##prediction training model





#df_reviews

y_reviews_pred = decision_reviews.predict(X_reviews_test)



print("predict y_reviews_test:")

print(y_reviews[800])

print("predict y_reviews_pred:")

print(y_reviews_pred[800])

from sklearn.metrics import confusion_matrix, classification_report

##df_reviews

print("Accuracy Score of model decision tree")

print("Training of df_reviews :", decision_reviews.score(X_reviews_train,y_reviews_train))

print("Testing of df_reviews :", decision_reviews.score(X_reviews_test,y_reviews_test))
#used confussion matrixs

print("Confusion matrixs")

print(confusion_matrix(y_reviews_test,y_reviews_pred))

#classification report

print("Classification of report")

print(classification_report(y_reviews_test,y_reviews_pred))
##fitting 



#df_reviews

knn_reviews = knn.fit(X_reviews_train,y_reviews_train)
##prediction training model





#df_reviews

y_reviews_pred = knn_reviews.predict(X_reviews_test)



print("predict y_reviews_test:")

print(y_reviews[800])

print("predict y_reviews_pred:")

print(y_reviews_pred[800])

from sklearn.metrics import confusion_matrix, classification_report

##df_reviews

print("Accuracy score of model  knn")

print("Training of df_reviews :", knn_reviews.score(X_reviews_train,y_reviews_train))

print("Testing of df_reviews :", knn_reviews.score(X_reviews_test,y_reviews_test))
#used confussion matrixs

print("Confusion matrixs")

print(confusion_matrix(y_reviews_test,y_reviews_pred))
#classification report

print("Classification of report")

print(classification_report(y_reviews_test,y_reviews_pred))
##fitting 



#df_reviews

#svm_reviews = svm.fit(X_reviews_train,y_reviews_train)
##fitting 



#df_reviews

#navie_reviews = naive.fit(X_reviews_train,y_reviews_train)