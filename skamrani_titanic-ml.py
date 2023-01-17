# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"));

df = pd.read_csv("../input/train.csv");

df.columns



X = pd.DataFrame();

X['Sex'] = df['Sex'];

X['Age'] = df['Age'];

X['Survived'] =df['Survived'];





X = X.dropna(axis=0);

# Survial is the dependent value 

Y = X['Survived'] ;

X = X.drop(['Survived'],axis=1);



#print(X);

#print(Y);



# We need to handle sex as a Categorial vairable . Male = i and female =0

# We can use panads get_dummies to handle categorziaion.

pd.get_dummies(X.Sex);



X['Sex'] = pd.get_dummies(X.Sex)['female'];

print(X);



from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.cross_validation import train_test_split

scaler = StandardScaler();

X = scaler.fit_transform(X);



# Build the test and training set 

X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=0.2,random_state=42);



def base_rate_model(X):

    y = np.zeros(X.shape[0]);

    return y;



#How accurate is our base model

y_base_rate = base_rate_model(X_test);

from sklearn.metrics import accuracy_score;

print (accuracy_score(Y_test,y_base_rate));



# Run Loggistic Regression

from sklearn.linear_model import LogisticRegression

model = LogisticRegression();



model.fit(X_train,Y_train);



print ("Logistic regression accuracy is ");

print (accuracy_score(Y_test,model.predict(X_test)));











# Any results you write to the current directory are saved as output.
