

import numpy as np 

import pandas as pd 

# Reading data With help of pandas 

Train_data=pd.read_csv('../input/titanic/train.csv')

Test_data=pd.read_csv('../input/titanic/test.csv')
# Lets See What our dataset consist of , .head() will give you top 5 rows of the dataset

Train_data.head()
# Lets see How many rows and columns we have

Train_data.shape

# the training dataset consist of 891 rows and 12 columns
# lets check how many Null values we have and datatype of our columns

Train_data.info()
# Lets Drop some of the columns which aren't going to contribute much for our models

columns_to_drop = ["PassengerId","Name","Ticket","Cabin","Embarked"]

# these are the columns i am going to drop and make a new clean dataframe in which doesn't consist of these columns 
#after dropping columns , make new_train_data is our new dataframe

New_train_data = Train_data.drop(columns_to_drop,axis=1)
# after dropping columns we are left with these columns

New_train_data.head()
# As you can see the column Sex is categorical or object , so we have to map this as male 0 and female 1 

# Because we need everything numerical and we are using LabelEncoder to map this

from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

New_train_data["Sex"] = enc.fit_transform(New_train_data["Sex"])
# Lets have a look again at dataset

New_train_data.head()

# You can see that we successfully convert the Survived columns to male 0 and female 1
# Now lets deal with missing values in our age column and replace it with mean value

New_train_data = New_train_data.fillna(New_train_data["Age"].mean())

# Now take a look again

New_train_data.info()

# We dont have any missing values now 
# Lets create X and Y , X is our features and y is our Output 

input_cols = ['Pclass',"Sex","Age","SibSp","Parch","Fare"]

output_cols = ["Survived"]



X = New_train_data[input_cols]

Y = New_train_data[output_cols]



print(X.shape,Y.shape)
# We are going to convert our dataset in the range 0 to 1 it will help our model to converge faster

from sklearn.preprocessing import StandardScaler

scal=StandardScaler()

scal.fit(X)

std_scal=scal.transform(X)
# Lets divide our dataset 20% for testing and 80% for training

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( std_scal, Y, test_size=0.2, random_state=42)
# now lets train our model , we are using two or three models eg KNN, SVM , Decission tree

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train,y_train)
# Lets predict and test our models accuracy

y_pred=knn.predict(X_test)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_pred,y_test))
from sklearn.svm import SVC

svmm=SVC(kernel="rbf")

svmm.fit(X_train,y_train)

y_pred=svmm.predict(X_test)

print(accuracy_score(y_pred,y_test))
# Now lets create csv for submission from test.csv by appling same method as we done on train.csv

Test_data.head()
passengerid=Test_data["PassengerId"]

Test_data.info()
columns_to_drop = ["PassengerId","Name","Ticket","Cabin","Embarked"]

Test_data_clean = Test_data.drop(columns_to_drop,axis=1)
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



Test_data_clean["Sex"] = le.fit_transform(Test_data_clean["Sex"])

Test_data_clean.head()
Test_data_clean = Test_data_clean.fillna(Test_data_clean["Age"].mean())
Test_data_clean.info()
from sklearn.preprocessing import StandardScaler

scal=StandardScaler()

scal.fit(Test_data_clean)

X_test2=scal.transform(Test_data_clean)
y_pred=svmm.predict(X_test2)
# we have to submit csv with columns passengerid and survived after prediction from our model

submission = pd.DataFrame({

        "PassengerId": passengerid,

        "Survived": y_pred

    })
submission.to_csv('submission1.csv', index=False)