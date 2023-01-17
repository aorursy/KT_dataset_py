# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/Dataset_spine.csv')
print(df.shape)
print(df.head(10))
df.head(10)
df.drop(columns='Unnamed: 13',inplace=True)
col_name = ['pelvic_incidnece','pelvic_tilt','lumbar_lordosis_angle','sacral_slope','pelvic_radius','degree_spondylolisthesis',
           'pelvic_slope','Direct_tilt','thoracic_slope','cervical_tilt','sacrum_angle','scoliosis_slope','Class_att']
df.columns = col_name

df.head(10)
df.describe()
dummies = pd.get_dummies(df['Class_att'],drop_first=True)
df = pd.concat([df,dummies],axis=1)
df.drop(columns="Class_att",inplace=True)
df.head(10)
import seaborn as sns
import matplotlib.pyplot as plt

corr = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr,annot=True)
#Split to training & testing dataset

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

x = df.drop(columns='Normal')
y = np.array(df['Normal'])

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=15)

# Logistic regression model
log_model = LogisticRegression()

log_model.fit(x_train,y_train)

prediction = log_model.predict(x_test)
#Check its accuracy
from sklearn.metrics import accuracy_score,confusion_matrix

#Check overall accuracy
log_acc_score = accuracy_score(y_test,prediction)
print('accuracy score of logistic model is: {}'.format(log_acc_score))

#Print confusion matrix
pd.DataFrame(confusion_matrix(y_test,prediction),columns=['Abmormal','Normal'],index=['Abnormal','Normal'])


#Random Forest technique

from sklearn.ensemble import RandomForestClassifier

#One question that we commonly think of
#How many trees should we use?
#Let's tune this hyperparameter

#Techniquely, the more trees we use, the precise outcome we would get
x_train2,x_test2,y_train2,y_test2 = train_test_split(x,y,test_size=0.3,random_state=15)


#Generate forests containing 10(default), 50, 100, 200, 300 trees
n_trees = [10,50,100,200,300]
for i in n_trees:
    ran_for = RandomForestClassifier(n_estimators=i)
    ran_for.fit(x_train2,y_train2)
    pred = ran_for.predict(x_test2)
    
    print('n of trees: {}'.format(i))
    #Each time of prediction,the accuracy is measured
    correct_pred = 0
    for j,k in zip(y_test2,pred):
        if j == k:
            correct_pred += 1
    print('correct predictions: {}'.format(correct_pred/len(y_test2) *100))
    matrix = pd.DataFrame(confusion_matrix(y_test2,pred),columns=['Abmormal','Normal'],index=['Abnormal','Normal'])
    print(matrix)
