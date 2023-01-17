# import all dependencies
import numpy as np # linear algebra
import tensorflow as tf
from keras.utils import normalize
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
# take dataset 
df=pd.read_csv("../input/train.csv")
test_data=pd.read_csv("../input/test.csv")
test_data=test_data.drop(['Name','Ticket','Cabin'],axis=1)
test_data=test_data.dropna()
print(test_data.head())
print("test_data.shape",test_data.shape)
print("Before drop column  shape of dataset : ",df.shape)
# drop useless features/columns
#this process also called ''feature engineering''
df=df.drop(['Name','Ticket','Cabin'],axis=1)
print(df.info())
val=df.isnull().sum()
print("no of null values :",val)
mean1=df['Age'].mean()
df['Age']=df['Age'].fillna(mean1)
#mean2=df["Embarked"].mean()
#we use Q 
df['Embarked']=df['Embarked'].fillna('s')
#df=df.
#df=normalize(df,axis=1)
df=df.dropna()
print("After drop column  shape of dataset : ",df.shape)

# Describe all feature related information e:g- mean,max,std,count etc
print(df.describe())
print(df.shape)

# Feature scalling and manipulate the data
X=np.array(df.iloc[:,2:])
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
x_l=LabelEncoder()
x_o=OneHotEncoder()
print(X.shape)
#print(X[:,-1])
X[:,1]=x_l.fit_transform(X[:,1])
X[:,-1]=x_l.fit_transform(X[:,-1])
#print(X[:,1])
print(X[:,-1])


# Now here play with graphprint
print(df.shape)
print(df.info())
# let's know how many people were survived from which class
print(df.Survived.value_counts())
df[['Pclass','Survived']].groupby('Pclass').count()

#  people survived or not
a=df.Survived.value_counts().plot('bar',color=['y','r'])
a.set_xlabel("survival or not survival",color='r')
a.set_ylabel("no of passenger",color='c')

a=df.Sex.value_counts().plot('bar',color=['b','y'])
a.set_xlabel("Sex",color='r')
a.set_ylabel("No of people",color='c')
a=df.Embarked.value_counts().plot('bar')
a.set_xlabel('Emabrked category',color='r')
a.set_ylabel('No of Person belongs to Emabrked',color='b')
a=df.Parch.value_counts().plot('bar')
a.set_xlabel('Parch',color='r')
a.set_ylabel('No of Person belongs to Parch',color='c')
# lets see on behave of sex what is the probabilities that they survive
a=df[['Sex','Survived']].groupby('Sex').mean().Survived.plot('bar')
a.set_xlabel("Sex")
a.set_ylabel("Probability of survived")
print("female survived more")
# lets see on behave of parch what is the probabilities that they survive
a=df[['Parch','Survived']].groupby('Parch').mean().Survived.plot('bar')
a.set_xlabel("Parch")
a.set_ylabel("Probability of survived")
# rescale data
#X=np.array(df.iloc[:,'Pclass','Sex','Age','SibSp','parch','Embarked'])

y=np.array(df.iloc[:,1])
print(y[:5])
y=y.reshape(-1,1)

#  to check the shape
print("X.shape",X.shape)
print("y.shape",y.shape)

'''
# Before try neural network 
# first try some basics classifiers
# dicidion tree classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
'''
'''
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)
print(accuracy)
'''
'''
# to save file in kaggle apply this method its very simple method
test_data.PassengerId.to_csv('gender_submission.csv',header=True, index_label='Survived')
print('file saved')

'''
'''
# now apply Knearest neighbors classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
'''
'''
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.15)
clf=KNeighborsClassifier()
clf.fit(x_train,y_train)
accuracy=clf.score(x_test,y_test)
print(accuracy)

'''
# Now apply neural network
# import deep learning libraries,layers,models
from keras.models import Sequential
from keras.layers import Dense,Dropout
model=Sequential()

print(y.shape)
# add layers
model.add(Dense(activation='relu',units=128))
model.add(Dense(activation='relu',units=128))
model.add(Dense(activation='relu',units=128))
model.add(Dropout(0.7))
model.add(Dense(activation='relu',units=128))
# to avoid overfitting
model.add(Dropout(0.5))
# output layer
model.add(Dense(activation='sigmoid',units=1))
# compile our model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# train model
print("training....................")
model.fit(X,y,epochs=20,batch_size=10)
print("Training completed !!!")
# Now set test data
x_test=np.array(test_data.iloc[:,1:])
from sklearn.preprocessing import LabelEncoder
x_l=LabelEncoder()
print(x_test.shape)
#print(x_test[:,-1])
x_test[:,1]=x_l.fit_transform(x_test[:,1])
x_test[:,-1]=x_l.fit_transform(x_test[:,-1])
print(x_test[:,-1])

# first 5  Passenger
print(test_data.head())
# those value which is less then 0.5 belongs to 0   
# those value which is greater then 0.5 belongs to 1  
prediction=model.predict_classes(x_test)
print("prediction.shape ",prediction.shape)
print("Survived                         0-Died    1-Survive\n",prediction[:5])
print("These are our first 5  predicted class")
#  people survived or not in test case
a=test_data.Sex.value_counts().plot('bar',color=['y','r'])
a.set_xlabel("survival or not survival in test case",color='r')
a.set_ylabel("no of passenger",color='c')
#  people survived or not in test case
a=test_data.Embarked.value_counts().plot('bar',color=['y','r'])
a.set_xlabel("Embarked in test case",color='r')
a.set_ylabel("no of passenger",color='c')