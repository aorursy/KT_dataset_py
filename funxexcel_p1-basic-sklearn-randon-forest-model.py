import pandas as pd

from sklearn.ensemble import RandomForestClassifier
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
data.head()
data.isnull().sum()
#Get Target data 

y = data['target']



#Load X Variables into a Pandas Dataframe with columns 

X = data.drop(['target'], axis = 1)
print(f'X : {X.shape}')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)
print(f'X_train : {X_train.shape}')

print(f'y_train : {y_train.shape}')

print(f'X_test : {X_test.shape}')

print(f'y_test : {y_test.shape}')
rf_Model = RandomForestClassifier()
rf_Model.fit(X_train,y_train)
print (f'Train Accuracy - : {rf_Model.score(X_train,y_train):.3f}')

print (f'Test Accuracy - : {rf_Model.score(X_test,y_test):.3f}')