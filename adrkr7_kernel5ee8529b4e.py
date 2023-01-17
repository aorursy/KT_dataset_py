import pandas as pd
import numpy as np
import scipy.stats as probplot
import pylab
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df=pd.read_csv(r"/kaggle/input/cardiovascular-disease-dataset/cardio_train.csv",sep=";")
df.head()
df.info()
df.shape
df.head()
for i in df.columns:
    print(i)
    print(df[i].describe())
for i in df.columns:
    print(i)
    sns.distplot(df[i])
    plt.show()
(df["ap_hi"]<0).sum()
'''but no systolic pressure can be negative . Assuming it be a recording error I have imputed these values to its absolute values'''
df["ap_lo"]=abs(df["ap_lo"])
'''Similary it was found that the diastolic presssure column also had some values that was negative and hence the absolute value of the column was found and replaced '''
df["ap_hi"]=abs(df["ap_hi"])
'''To check whether the classs is Balanced or not......'''
print("No of class 0 Records/Observatios------> {o} ".format(o=(df["cardio"]==0).sum()))
print("No of class 1 Records/Observatios------> {o} ".format(o=(df["cardio"]==1).sum()))
'''No of Class 1 Observation is nearly the same as Class 0'''
'''To check Whether the systolic pressure column values are  consistent....Aim is to check the veracity of the data '''
'''After going through the desciption of each column generated earlier it was found that the maximum value for column ap_hi was 11000 ...next to impossible right. ' '''
'''Dropping the records which were found to have an abnormal value for column ap_hi(Taking a threshold value 500 as the permissible limit..... I know its still very high '''
print("No of record with ap_hi value greater than 500-------> {o}".format(o=(df["ap_hi"]>500).sum()))
#storing the indexes of these records
ind=df[(df["ap_hi"]>500)].index
#dropping these record
df.drop(index=ind,inplace=True)
'''The above process was again fowlled  for column ap_lo as is had adnormal values as well'''
#storing the index of the records
ind1=df[df["ap_lo"]>500].index
#dropping these records
df.drop(index=ind1,inplace=True)
df.head()
#finding the new shape of the data
df.shape
#Using the probplot to find the distribution of the data ie.if the data is not skewed ie follows the normal distribution the points(blue) will trace the red line as shown in these figures.....  
for i in df.columns:
    print(i)
    scipy.stats.probplot(df[i],dist="norm",plot=pylab)
    #sns.distplot(df[i])
    plt.show()
'''After alalyzing the distribution of all the columns it was found that column:"age","height","weight","ap_hi","ap_lo" did not follow the normal distribution ....and for the ordered catagorical varibles like "active","alco","smoke","gluc","cholesterol" no normalization needed becaude we need to reatinthe ordered relation '''
l=["age","height","weight","ap_hi","ap_lo"]
for i in l:
    print(i)
    scipy.stats.probplot((df[i]**(1/3)),dist="norm",plot=pylab)
    df[i]=df[i]**(1/3)
    #sns.distplot(df[i])
    plt.show()
plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)
#dropping coulns id:Used to uniquely identify each and every record in the database has got nothing to do with the disease,Height:Being a continious variable still has a very less corelationtion 
x=df.drop(columns=["id","height"])
y=df.iloc[:,-1]
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
oh=make_column_transformer(
    (OneHotEncoder(categories='auto'), [1]), 
    remainder="passthrough")
x=oh.fit_transform(x)
#though the columns more or less follow the normal distribution curve but they may not follow the ideal bell shaped curve ....ie mean=0 and std=1 so inorder to achieve this condition standard scalar is used (Column may follow the bell_shaped curve but its mean may not be =0 ie diffrent columns may have different range of values so inorder to bring thenm on even grounds this is done) )
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
#splitting the data set into train and test with test_size=30%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=.3)
from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()
lg.fit(x_train,y_train)
#Cardiovascular Dataset using logistic Regression on train set
log_train=lg.score(x_train,y_train)*100
#Cardiovascular Dataset using logistic Regression on test set
log_test=lg.score(x_test,y_test)*100
#confusion matrix of prediction obtainned using Logistic
from sklearn.metrics import confusion_matrix
p_test_1=lg.predict(x_test)
confusion_matrix(y_test,p_test_1,labels=[0,1])
y_pred_log=lg.predict(x_test)
from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier(11,algorithm='kd_tree')
kn.fit(x_train,y_train)
kn.score(x_train,y_train)*100
#KNN score on Test
kn.score(x_test,y_test)*100
#KNN confusion matrix
y_pred_KNN=kn.predict(x_test)
confusion_matrix(y_test,y_pred_KNN,labels=[0,1])
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(x_train,y_train)
from sklearn.metrics import confusion_matrix
print(clf.score(x_test,y_test)*100)
y_pred_Decision=clf.predict(x_test)
confusion_matrix(y_test,y_pred_Decision,labels=[0,1])
Model_Decision_eval=precision_recall_fscore_support(y_test,p_test_1)
import xgboost as xgb
xg_model=xgb.XGBClassifier()
xg_model.fit(x_train,y_train)
y_pred_xg=xg_model.predict(x_test)*100
xg_model.score(x_train,y_train)*100
print(xg_model.score(x_test,y_test)*100)
p_test_1=xg_model.predict(x_test)
confusion_matrix(y_test,p_test_1,labels=[0,1])
Model_XG_eval=precision_recall_fscore_support(y_test,p_test_1)
'''Model Evaluation results '''
l_train=[lg.score(x_train,y_train),kn.score(x_train,y_train),clf.score(x_train,y_train),xg_model.score(x_train,y_train)]
l_test=[lg.score(x_test,y_test),kn.score(x_test,y_test),clf.score(x_test,y_test),xg_model.score(x_test,y_test)]
pd.DataFrame({"Train":np.array(l_train)*100,"Test":np.array(l_test)*100},index=["Logistic Regression","KNN","Decision_Tree","XG_Boost"])
'''Logistic Regression'''
print(classification_report(y_test,y_pred_log,target_names=['class 0','class 1']))
'''KNN'''
print(classification_report(y_test,y_pred_KNN))
'''Decision Tree'''
print(classification_report(y_test,y_pred_Decision,target_names=['class 0','class 1']))
'''XG_BOOST'''
print(classification_report(y_test,y_pred_xg))



