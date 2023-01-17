#This is how you can easily get 100 % accuracy in the poisonous mushroom dataset

#Firstly lets import some helpful libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df=pd.read_csv("../input/mushrooms.csv")



#Firstly we will see some useful information about data

print(df.head(3))
data=df.as_matrix()

print("Shape of data is",data.shape)
#Lets check if there are any missing values in any column

print("No. of missing values in column are:")

names = df.columns.values

for i in range(1,23):

    col_is_null=df.iloc[:,i].isnull().sum()

    print(names[i],":",col_is_null)
def get_data(file_path,label_col="False"):

    df=pd.read_csv(file_path)

    data=df.as_matrix()

    

    if(label_col=="False"): #No labels.Used for test data

        return data

    

    y=data[:,label_col]

    X=np.delete(data,label_col,axis=1)

    return X,y        
X,y=get_data("../input/mushrooms.csv",label_col=0)

print("Shape of data and labels is",X.shape,y.shape)
#Now we will one hot encode the data.This is necessary because our data is categorical

#Firstly lets define few constants for our data

N=X.shape[0] #This is the no. of data points or training examples we have

D=X.shape[1] #This is the dimensionality or no. of features in our dataset



X_encoded=np.empty((N,1))

for i in range(D):

    dum=pd.get_dummies(X[:,i])

    X_encoded=np.hstack((X_encoded,dum))

#Since first col is empty we need to remove it

X_encoded=X_encoded[:,1:]

print("Shape of X_encoded is",X_encoded.shape)

X=X_encoded
#Lets find out the no. of poisonous and edible mushrooms

n_edible= np.sum(y=='p')

n_poisonous= np.sum(y=='e')

total= n_edible + n_poisonous

print("No. of poisonous mushrooms is",n_poisonous)

print("No. of edible mushrooms is",n_edible)

print("% of posionous mushrooms is",(n_poisonous/total)*100)

print("% of edible mushrooms is",(n_edible/total)*100)
#Now after we have processed our data we are good to go.

#Firstly we will split our data into train and test sets

#For this we will be using a sklearn function train_test_split

from sklearn.model_selection import train_test_split

#We will keep 20% of the data for testing

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)



print("Shape of X_train, y_train is ",X_train.shape,y_train.shape)

print("Shape of X_test, y_test is ",X_test.shape,y_test.shape)
#To select a model we will use cross validation for our training set.

#Then we will apply the selected model to our test set and compare accuracies

#Let import cross_val_scores from sklearn to aid us

from sklearn.model_selection import cross_val_score



#Firstly we will define a baseline model

#A baseline model is basically a very simple or trivial approach to our dataset.

#All our models should have a greater performance than the baseline

#Logistic regression will be used as baseline here

from sklearn.linear_model import LogisticRegression

#Logistic Regression has a few parameters which you can try tuning

print(LogisticRegression())
model=LogisticRegression()

cv_baseline=cross_val_score(model,X_train,y_train)

print("Cross val scores have a mean",cv_baseline.mean()," and standard deviation ",cv_baseline.std())

# A low  cross val std indicates that our model is not overfitting.

# A high mean cross val score also shows that our model is not underfitting

model.fit(X,y)

y_test_pred=model.predict(X_test)

print("Score on test set is",np.mean(y_test==y_test_pred))



#We are a getting 100 % accuracy in our baseline model itself.
#We will use some other algorithms and compare accuracies

#Lets first import them

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier

from xgboost import XGBClassifier
models=[LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier(),

        AdaBoostClassifier(),XGBClassifier()]

names=["Logistic Regression","DecisionTree","RandomForest","AdaBoost","XGBoost"]

print("For given model mean cross_val_scores standard deviation\n")

for model,name in zip(models,names):

    model.fit(X,y)

    cv_baseline=cross_val_score(model,X_train,y_train)

    y_test_pred=model.predict(X_test)

    print(name,"Mean",cv_baseline.mean(),"Standard dev:",cv_baseline.std(),"Test Set Score:",

          np.mean(y_test==y_test_pred))

    