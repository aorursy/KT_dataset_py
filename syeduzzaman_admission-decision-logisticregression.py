import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing

from sklearn import utils

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../input/graduate-admissions/Admission_Predict_Ver1.1.csv",sep=',')

#df = pd.read_csv("../input/graduate-admissions/Admission_Predict.csv",sep=',')



print("Row: ",df.shape[0])

print("Column: ",df.shape[1])

df.head(10)
df.describe()
df=df.dropna()
y=df.iloc[:,-1]

#y=df[['Chance of Admit']]

df=df[['GRE Score','TOEFL Score','University Rating','SOP','LOR ','CGPA','Research']]



df.head(2)


xtrain, xtest, ytrain, ytest = train_test_split( 

        df, y, test_size = 0.25, random_state = 0)



scaler = StandardScaler()

xtrain = scaler.fit_transform(xtrain)

xtest = scaler.fit_transform(xtest)



ytrain_binary=(ytrain>0.7).astype(int)

ytest_binary=(ytest>0.7).astype(int)

reg = LogisticRegression() 

   

# train the model using the training sets 

reg.fit(xtrain, ytrain_binary) 

  
ynew = reg.predict(xtest)

Model_accuracy = (reg.score(xtest, ytest_binary))*100

Model_accuracy
from sklearn.metrics import classification_report,confusion_matrix



print(confusion_matrix(ytest_binary,ynew))
print(classification_report(ynew,ytest_binary))
