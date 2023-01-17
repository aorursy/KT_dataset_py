# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix,f1_score,accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
diadf=pd.read_csv('../input/diabetes.csv')

#print(len(diadf))

diadf.head()
diadf.isna().sum()
#Replace zeroes

zero_not_accepted=['Glucose','BloodPressure','SkinThickness','BMI','Insulin']

for column in zero_not_accepted:

    diadf[column]=diadf[column].replace(0,np.NaN)

    mean=int(diadf[column].mean(skipna=True))

    diadf[column]=diadf[column].replace(np.NaN,mean)
X=diadf.iloc[:,0:8]

y=diadf.iloc[:,8]

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)

sc_X=StandardScaler()

X_train=sc_X.fit_transform(X_train)

X_test=sc_X.transform(X_test)
import math

math.sqrt(len(y_test))
#Define the model: Init K-NN

classifier=KNeighborsClassifier(n_neighbors=11,p=2,metric='euclidean')
#fit the model

classifier.fit(X_train,y_train)
#predict the test set results

y_pred=classifier.predict(X_test)

y_pred
#evaluate the model

cm=confusion_matrix(y_test,y_pred)

print(cm)

print(f1_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred))