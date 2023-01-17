import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder  
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
# Importing the dataset
wine = pd.read_csv('../input/winequality-red.csv')

wine.head(10)
for i in range(len(wine)):
    if wine.iloc[i,11]>=7:
        wine.iloc[i,11]='good'
    else:
        wine.iloc[i,11]='bad'
wine.head(10)
labelencdoer=LabelEncoder()
wine['quality']=labelencdoer.fit_transform(wine['quality'])
wine.head(10)
X = wine.iloc[:, :-1].values
y = wine.iloc[:, 11].values

#scaling the Xvalue
sc=StandardScaler()
X=sc.fit_transform(X[:,:])

#Splitting into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#n_estimators represents nos of trees to be used in the model
rfr=RandomForestRegressor(n_estimators=40,random_state=0)
rfr.fit(X_train,y_train)

#Final Prediction
y_pred=np.matrix.round(rfr.predict(X_test))

acc=accuracy_score(y_test, y_pred)
print("accuracy: ",acc*100,'%' )