#1.1 Calling Libraries



%reset -f

import numpy as np

import pandas as pd 

from sklearn.decomposition import PCA

from sklearn.decomposition import NMF

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt



from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
#2.0 Import OS directory and import data from CSV file

import os

import pandas as pd



fashion = pd.read_csv("/kaggle/input/fashionmnist/fashion-mnist_train.csv")

fashion.shape

#3.0 Let's get some feel of Data How Fashion Images visualizes.



y = fashion['label']

X= fashion.drop(['label'],axis =1)



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1,random_state =42)



X_train_np = X_train.to_numpy()

y_train_np = y_train.to_numpy()

X_train.shape

X_test.shape

y_train.shape

y_test.shape





#3.1 See the images of various fashion items

class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure(figsize=(10, 10))

for i in range(36):

    plt.subplot(6, 6, i + 1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(X_train_np[i].reshape((28,28)))

    label_index = int(y_train_np[i])

    plt.title(class_names[label_index])

plt.show()
#4.1 Checking Correlation Between Different Components



df=fashion.drop('label', 1)

f , ax = plt.subplots(figsize = (14,12))

plt.title('Correlation of Features- HeatMap',y=1,size=16)

sns.heatmap(df.corr(),square = True,  vmax=0.8)
#4.2 Great we find correlation between different variable



#Dimentionality reduction using Principle Component Analysis convert the image size from 

# 1*784 to 1*144 size



#Drop the Label Column from Fashion dataset dimentionality reduction in Pixel columns only

fashion_label =fashion['label']

fashion_pixel =fashion.drop(['label'],axis=1)

ss = StandardScaler()

pca = PCA(n_components =144)

pipeline =make_pipeline(pca)



pipeline.fit(fashion_pixel)



fashion_new = pipeline.transform(fashion_pixel)

fashion_new.shape

#5.0 Divide the Dataset into Test and Train dataset.

y = fashion_label

X= fashion_new



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1,random_state =42)



#X_train_np = X_train.to_numpy()

#y_train_np = y_train.to_numpy()

X_train.shape

X_test.shape

y_train.shape

y_test.shape

#Doing Classification through Random Forest Classifier

rf= RandomForestClassifier(n_estimators =50,oob_score = True,bootstrap=True)

ss = StandardScaler()

pca = PCA()



pipeline_rf = make_pipeline(ss,pca,rf)



#Fit the fashion data into pipeline

pipeline_rf.fit(X_train,y_train)

y_predict_rf = pipeline_rf.predict(X_test)

score_rf = np.sum(y_predict_rf == y_test)/len(y_test)

score_rf





#5.1 Checking the Score of Logistics Regression Model

logreg = LogisticRegression()

                            

pipeline_logreg =make_pipeline(ss,pca,logreg)



#Fit the fashion data into pipeline



pipeline_logreg.fit(X_train,y_train)



y_predict_logreg = pipeline_logreg.predict(X_test)



score_logreg =np.sum(y_predict_logreg ==y_test)/len(y_test)

score_logreg
#5.2 Building Model using Adaboost Classifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier



#Instantiate Decision Tree Classifier

dt = DecisionTreeClassifier(max_depth =1,random_state =1) 

ada = AdaBoostClassifier(base_estimator =dt,n_estimators =1000,random_state = 1)

                            

pipeline_Adaboost =make_pipeline(ss,pca,ada)





#Fit the fashion data into pipeline



pipeline_Adaboost.fit(X_train,y_train)



y_predict_Adaboost = pipeline_Adaboost.predict(X_test)



score_Adaboost =np.sum(y_predict_Adaboost ==y_test)/len(y_test)

score_Adaboost

#5.3 Confusion Matrix of AdaBoost Algorithm

cf_Adaboost = confusion_matrix(y_test,y_predict_rf)

cf_Adaboost