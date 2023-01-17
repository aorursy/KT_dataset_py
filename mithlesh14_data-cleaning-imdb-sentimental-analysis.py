# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy import stats
%matplotlib inline 


#reading file through pandas and making dataframe
data=pd.read_csv(r"../input/movie_metadata.csv")
#to check information of invidual col:
data.info()
#to check first five row of a datasets
data.head()
#As we have seen that there are 5043 rows and 28 columns.
#through isna() function we can check is there any null value and if it present then by how much 
data.isna().sum()
#describe is a function to check all the possiblities the datasets have,
#it describe all the mathmatical value to make it easy to understand the datasets
data.describe()
#droping the columns which is not necessary
data.drop(["actor_1_facebook_likes","duration","actor_3_facebook_likes",
           "gross","genres","actor_1_name","facenumber_in_poster",
           'plot_keywords','title_year',"movie_imdb_link",'actor_2_facebook_likes','num_user_for_reviews'],axis=1,inplace=True)
data.info() #now we left with 14 columns 
data.head()
#we need to drop some more columns 
data.drop(["language"],axis =1,inplace=True)
data.isna().sum()
#removing NaN value
data['color']=data['color'].fillna("color")
data.replace({"actor_2_name":np.NaN,
              "actor_3_name":np.NaN,
              "country":np.NaN,
             "content_rating":np.NaN},value="None",inplace=True)# As we cannot decide who make what film
data.replace({'director_facebook_likes':np.NaN},value=0.0,inplace=True)
data.replace({'director_name':np.NaN},value="None",inplace=True)
data['num_critic_for_reviews']=data['num_critic_for_reviews'].fillna(value=data['num_critic_for_reviews'].mean())
data["aspect_ratio"]=data['aspect_ratio'].fillna(method='ffill')
data.drop(data.index[4],inplace=True)
data.head(100)
#plotting heat map:
plt.figure(figsize=(18,8),dpi=100,)
plt.subplots(figsize=(18,8))
sns.heatmap(data=data.corr(),square=True,vmax=0.8,annot=True)
#plotting hist curve to and calculating normal distribution 
from scipy.stats import norm
sns.distplot(a=data['movie_facebook_likes'],hist=True,bins=10,fit=norm,color="red")
plt.title("IMDB Movie Review")
plt.ylabel("frequency")
mu,sigma=norm.fit(data['movie_facebook_likes'])
print("\n mu={:.2f} and sigma={:.2f}\n ".format(mu,sigma))
plt.legend(["normal distribution.($\mu=${:.2f} and $\sigma=${:.2f})".format(mu,sigma)])
plt.show()


sns.jointplot(x=data['movie_facebook_likes'],y=data['budget'],kind="reg",dropna=True)
data=data.drop(data[(data['budget']>200000000.0)].index).reset_index(drop=True)


sns.jointplot(x=data['movie_facebook_likes'],y=data['budget'],kind="reg",dropna=True)
data['budget']=data['budget'].fillna(data['budget'].mean())
data.head(10)
sns.jointplot(x=data['imdb_score'],y=data['budget'],kind="reg")
data=data.drop(data[(data['budget']>175000000.0)].index).reset_index(drop=True)
sns.jointplot(x=data['imdb_score'],y=data['budget'],kind="reg" )
plt.figure(figsize=(18,8),dpi=100,)
plt.scatter(x=data['imdb_score'],y=data['movie_facebook_likes'],alpha=0.8,color="red")
plt.ylabel("facebook likes of  movies",color="blue",size=20)
plt.xlabel("imdb score",color="blue",size=20)
plt.title("facebook likes VS imdb score",color="red",size=30);
data=data.drop(data[(data['movie_facebook_likes']>125000)].index).reset_index(drop=True)

plt.figure(figsize=(18,8),dpi=100,)
plt.scatter(x=data['imdb_score'],y=data['movie_facebook_likes'],alpha=0.8,color="red")
plt.ylabel("facebook likes of  movies",color="blue",size=20)
plt.xlabel("imdb score",color="blue",size=20)
plt.title("facebook likes VS imdb score",color="red",size=30);
sns.jointplot(x=data['num_voted_users'],y=data['imdb_score'],kind='reg')
data=data.drop(data[(data['imdb_score']<2)].index).reset_index(drop=True)
sns.jointplot(x=data['num_voted_users'],y=data['imdb_score'],kind='reg')
from scipy.stats import norm
sns.distplot(a=data['movie_facebook_likes'],hist=True,bins=10,fit=norm,color="red")
plt.title("IMDB Movie Review")
plt.ylabel("frequency")
mu,sigma=norm.fit(data['movie_facebook_likes'])
print("\n mu={:.2f} and sigma={:.2f}\n ".format(mu,sigma))
plt.legend(["normal distribution.($\mu=${:.2f} and $\sigma=${:.2f})".format(mu,sigma)])
plt.show()

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encode=LabelEncoder()
data['color'] = encode.fit_transform(data['color'] ) 
data['director_name'] = encode.fit_transform(data['director_name'] ) 
data['actor_2_name'] = encode.fit_transform(data['actor_2_name'] ) 
data['movie_title'] = encode.fit_transform(data['movie_title'] ) 
data['country'] = encode.fit_transform(data['country'] ) 
data['content_rating'] = encode.fit_transform(data['content_rating'] ) 
data['actor_3_name'] = encode.fit_transform(data['actor_3_name'] )
data['num_voted_users'] = encode.fit_transform(data['num_voted_users'] )

#dataset_preprocessed=pd.get_dummies(data) #pd.get_dummies creates a new dataframe which consists of zeros and ones. 
#print(data.shape)
X=data.iloc[:,0:16].values
X=X[:,1:]
X.shape
data.info()
y=data.iloc[:,-1].values
y.shape
from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categories='auto')
onehotencoder.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=100)
X_train
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import accuracy_score
entropy=DecisionTreeClassifier(criterion="entropy",random_state=100,
                               max_depth=12,min_samples_leaf=3,splitter='best',presort=True)
entropy.fit(X_train,y_train)
predict=entropy.predict(X_test)
print("Accuacy",accuracy_score(y_test,predict)*100)
from sklearn.ensemble import RandomForestClassifier
random=RandomForestClassifier()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=100)
random.fit(X_train,y_train)
predict=random.predict(X_test)
print("Accuacy",accuracy_score(y_test,predict)*100)





