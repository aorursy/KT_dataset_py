# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/eval-lab-1-f464-v2/train.csv")

df.head(100)
df.corr()



test=df["feature9"]



print(np.round(test))
imp=["feature3","feature5","feature6","feature7"]





nimp=["feature1","feature2","feature3","feature4","feature5","feature6","feature7","feature8","feature9","feature10","feature11"];

from sklearn.preprocessing import StandardScaler



sc=StandardScaler();

df[imp]=sc.fit_transform(df[imp])



##df[imp]
df=df.fillna(df.mean())

df.isnull().any()
import matplotlib.pyplot as plt  

import seaborn as seabornInstance 

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics

%matplotlib inline
x_train,x_test,y_train,y_test=train_test_split(df[nimp],df["rating"],test_size=0.1,random_state=42);

#features=list(zip(df[imp]))

#features



'''

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

#create an object of the type GaussianNB

gnb = GaussianNB()

#train the algorithm on training data and predict using the testing data

pred = gnb.fit(x_train,y_train).predict(x_test)

#print(pred.tolist())

#print the accuracy score of the model

print("Naive-Bayes accuracy : ",accuracy_score(y_test, pred, normalize = True))





from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score

#create an object of type LinearSVC

svc_model = LinearSVC(random_state=0)



predic = svc_model.fit(x_train,y_train)



predi=predic.predict(x_test)

#print(pred.tolist())

#print the accuracy score of the model

print("SVC accuracy : ",accuracy_score(y_test, predi, normalize = True))



'''


#from sklearn.ensemble import RandomForestRegressor



#regressor = RandomForestRegressor(n_estimators=500, random_state=0)

#regressor.fit(x_train, y_train)

#y_pred = regressor.predict(x_test)
##from sklearn.neighbors import KNeighborsClassifier



##from sklearn import metrics



#for k in range(2,50):

##model = KNeighborsClassifier(n_neighbors=15)

# Train the model using the training sets

##model.fit(x_train,y_train)

#Predict Output

##predicted= model.predict(x_test) # 0:Overcast, 2:Mild



##print("Accuracy:",metrics.accuracy_score(y_test, predicted),'for k =',k)









from sklearn.ensemble import RandomForestClassifier



regressor = RandomForestClassifier(n_estimators=700);

regressor.fit(df[nimp], df["rating"])

y_pred = regressor.predict(x_test)
from sklearn import metrics



print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
df_new=pd.read_csv('/kaggle/input/eval-lab-1-f464-v2/test.csv')

df_new=df_new.fillna(df_new.mean())

new_test=df_new[nimp]



print(new_test)

##new_test=sc.fit_transform(new_test)



finalans=regressor.predict(new_test);

#finalans=np.round(finalans,0)

#finalans =finalans.astype(int)





finalans[1:40]

# Model Accuracy, how often is the classifier correct?

finalanss=pd.DataFrame({'id':df_new["id"],'rating':finalans})



#export_csv = df.to_csv (r'/home/submission.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

finalanss.to_csv('submission.csv', index=False)

finalanss