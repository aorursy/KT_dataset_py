#check path

from os import path



#read excel file into dataframe

import pandas as pd



#function to produce train test data split

from sklearn.model_selection import train_test_split



#classification model

from sklearn.ensemble import RandomForestClassifier



#scoring model

from sklearn.metrics import accuracy_score

from sklearn.metrics import recall_score

from sklearn.metrics import precision_score

from sklearn.metrics import f1_score
kaggle_file_path = '../input/credit-card-default/default of credit card clients.xls'



if path.exists(kaggle_file_path):

    file_path = kaggle_file_path

else:

    file_path = 'default of credit card clients.xls'
df = pd.read_excel(file_path, header=1)
#df.info() 
#df.describe()
#df.columns
X = df.drop(columns=['default payment next month', 'ID'])
y = df['default payment next month']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
clf = RandomForestClassifier(random_state=20).fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)