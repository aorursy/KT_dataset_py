import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import os
train_set = pd.read_csv('../input/train.csv')
train_set.head()
# Number of rows is 1600 and 22 features

train_set.shape
# Looking into the target variable
train_set.price_range.value_counts()
# Basic stats about the variables
train_set.iloc[:,1:].describe()
# No missing values in the dataset
train_set.isnull().sum()
#Lets convert these categorical values into numeric
l = ['touch_screen','bluetooth','dual_sim','wifi','4g','3g']
for i in l:
    train_set[i] = train_set[i].replace({'yes':1,'no':0})
train_set['price_range'] = train_set['price_range'].replace({'very low':0,'low':1,'medium':2,'high':3})
sns.jointplot(x='ram',y='price_range',data=train_set,color='red',kind='kde')
sns.boxplot(x="price_range", y="battery_power", data=train_set)
train_set['Ratio'] = train_set['primary_camera'][train_set.front_camera!=0] / train_set['front_camera'][train_set.front_camera!=0]
train_set = train_set.fillna(0)
train_set['diag'] = np.sqrt(np.square(train_set['resolution_width']) + np.square(train_set['resolution_height']))
train_set['ht_wd'] = train_set['screen_width'] / train_set['screen_height']
train_set = train_set.fillna(0)
train_set.columns
train_set = train_set[['ram',
                       'internal_memory',
                       'mobile_weight',
                       'Ratio',
                       'touch_screen',
                       #'talk_time',
                       #'primary_camera',
                       #'front_camera',
                       'bluetooth',
                       #'clock_speed',
                       '4g',
                       #'3g',
                       'wifi',
                       #'diag',
                       #'Aspect_Ratio',
                       #"mobile_depth",
                       'battery_power',
                       'n_cores',
                       'resolution_width','resolution_height',
                       'screen_width','screen_height',
                       'price_range']]
train_set.head(6)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
features = list(set(train_set.columns) - set(['Id','price_range']))
features
X = train_set[features]
Y = train_set['price_range']
trainX, testX, trainY, testY =  train_test_split(X, Y, test_size = .3)
dt = DecisionTreeClassifier()
model = dt.fit(trainX,trainY)
preds = model.predict(testX)
accuracy = accuracy_score(testY, preds)
precision = precision_score(testY, preds,average='micro')
recall = recall_score(testY, preds,average='micro')
print (accuracy,precision,recall)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors= 8, n_jobs= -1, weights= 'distance', leaf_size= 1, algorithm= 'auto')
model = knn.fit(trainX,trainY)
preds = model.predict(testX)
accuracy = accuracy_score(testY, preds)
precision = precision_score(testY, preds,average='micro')
recall = recall_score(testY, preds,average='micro')
print (accuracy,precision,recall)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
model = lda.fit(trainX,trainY)
preds = model.predict(testX)
accuracy = accuracy_score(testY, preds)
precision = precision_score(testY, preds,average='micro')
recall = recall_score(testY, preds,average='micro')
print (accuracy,precision,recall)
#With Hyper Parameters Tuning
#2-3,SVM
#importing modules
from sklearn.model_selection import GridSearchCV
from sklearn import svm
#making the instance
model=svm.SVC()
#Hyper Parameters Set
params = {'C': [0.01,0.02,0.2,0.25,5,1,2,10,7], 
          'kernel': ['linear','rbf']}
#Making models with hyper parameters sets
model1 = GridSearchCV(model, param_grid=params, n_jobs=-1)
#Learning
model1.fit(trainX,trainY)
#The best hyper parameters set
print("Best Hyper Parameters:\n",model1.best_params_)
#Prediction
prediction=model1.predict(testX)
#importing the metrics module
from sklearn import metrics
#evaluation(Accuracy)
print("Accuracy:",metrics.accuracy_score(prediction,testY))
#evaluation(Confusion Metrix)
print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,testY))
print (classification_report(testY,preds))
from sklearn.svm import SVC
svm = SVC(kernel= 'linear', C= 0.01)
model = svm.fit(trainX,trainY)
preds = model.predict(testX)
accuracy = accuracy_score(testY, preds)
precision = precision_score(testY, preds,average='micro')
recall = recall_score(testY, preds,average='micro')
print (accuracy,precision,recall)
test_set = pd.read_csv('../input/test.csv')
l = ['touch_screen','bluetooth','dual_sim','wifi','4g','3g']
for i in l:
    test_set[i] = test_set[i].replace({'yes':1,'no':0})
test_set['Ratio'] = test_set['primary_camera'][test_set.front_camera!=0] / test_set['front_camera'][test_set.front_camera!=0]
test_set = test_set.fillna(0)
test_set['diag'] = np.sqrt(np.square(test_set['resolution_width']) + np.square(test_set['resolution_height']))
test_set['ht_wd'] = test_set['screen_width'] / test_set['screen_height']
test_set = test_set.fillna(0)
test_set['price_range'] = model.predict(test_set[features])
test_set['price_range'] = test_set['price_range'].replace({0:'very low',1:'low',2:'medium',3:'high'})