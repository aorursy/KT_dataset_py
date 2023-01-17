
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')
df.head(5)
#Dropping the last column from the df
df.drop(columns=['Unnamed: 32'],axis=1,inplace=True)
df.info()
#Simple Plot of Area difference between Malignant and Benign Tumors
df.groupby('diagnosis').area_mean.mean().plot(kind='bar',color=['r','b'])
plt.ylabel('Mean Area of Tumor')
plt.xlabel('Diagnosis of Tumor')
#Dividing our data into X(independent Variables) and Y(target)
X=df.iloc[:,2:32]
Y=df.iloc[:,1:2]
#Seperating our data into training and testing data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
#We will be using StandardScaler() it is a general purpose scaling tool to scale the data generally used when there is a disparity in our data elements.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

%%time
#Lets use GridsearchCV as we promised to tune our hyperparameters, and then discuss why we did it after getting our result.
#Defining our classifier
from sklearn.svm import SVC
classifier = SVC(random_state=0)

from sklearn.model_selection import GridSearchCV
x = [1.0,10.0,100.0,500.0,1000.0]
y = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
z = [2,3,4]    
parameters=[{'C': x,'kernel': ['linear']},
            {'C': x,'kernel': ['rbf'],'gamma': y} ,
            {'C': x,'kernel': ['poly'],'gamma': y,'degree': z}
           ]
gridsearch=GridSearchCV(estimator = classifier,
                        param_grid = parameters,
                        scoring='accuracy',
                        cv=10,
                        n_jobs=-1)
gridsearch=gridsearch.fit(X_train,y_train)
#Getting our accuracy score for the model
accuracy=gridsearch.best_score_
accuracy
#seeing our best parameters
gridsearch.best_params_
#Using the best parameters as suggested by our GridsearchCV to finetune our model and validate it.
classifier=SVC(kernel='linear',C=1.0)
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)
acc=acc*100
acc=round(acc,2)
print('Accuracy Score of our model is: ',acc,'%')
