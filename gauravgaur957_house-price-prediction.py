import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
boston=load_boston()
boston
boston_df=pd.DataFrame(boston['data'],columns=boston['feature_names'])
boston_df['target']=pd.Series(boston['target'])
boston_df.head()
#Let's try the ridge regression model
from sklearn.linear_model import Ridge

#Setup random seed
np.random.seed(42)

#Create the data
X=boston_df.drop('target',axis=1)
y=boston_df['target']

#Split into training and test data sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#Instantiate the Ridge Model
model=Ridge()
model.fit(X_train,y_train)

#Check the score of ridge model on test data
model.score(X_test,y_test)

#How do we improve the accuracy of our model?
np.random.seed(34)
for i in range(10):
    model=Ridge(alpha=i/10)
    model.fit(X_train,y_train)
    a= model.score(X_test,y_test)
    
    print(f'The accuracy of our model for alpha={i/10} is {a}')
#Let's try the RandomForest regression model
from sklearn.ensemble import RandomForestRegressor

#Setup random seed
np.random.seed(42)

#Create the data
X=boston_df.drop('target',axis=1)
y=boston_df['target']

#Split into training and test data sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

#Instantiate the Ridge Model
model=RandomForestRegressor()
model.fit(X_train,y_train)

#Check the score of ridge model on test data
model.score(X_test,y_test)

#How do we improve the accuracy of our model?
np.random.seed(23)
for i in range(10,100,10):
    model=RandomForestRegressor(n_estimators=i)
    model.fit(X_train,y_train)
    a= model.score(X_test,y_test)
    
    print(f'The accuracy of our model for n_estimators={i} is {a}')
