import os

print(os.listdir('../input/'))
# Ignore  the warnings

import warnings

warnings.filterwarnings('always')

warnings.filterwarnings('ignore')



# data visualisation and manipulation

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

import missingno as msno



#configure

# sets matplotlib to inline and displays graphs below the corressponding cell.

%matplotlib inline  

style.use('fivethirtyeight')

sns.set(style='whitegrid',color_codes=True)



#import the necessary modelling algos.

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn import datasets

from sklearn.naive_bayes import GaussianNB



#model selection

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score

from sklearn.model_selection import GridSearchCV



#dim red

from sklearn.decomposition import PCA



#preprocess.

from sklearn.preprocessing import MinMaxScaler,StandardScaler,Imputer,LabelEncoder,OneHotEncoder
train=pd.read_csv(r'../input/pima-indians-diabetes-database/diabetes.csv')
train.head(10)
df=train.copy()
df.head(10)
df.shape # this gives the dimensions of the dataset.
df.index   
df.columns # gives a short description of each feature.
# check for null values.

df.isnull().any()   
msno.matrix(df)  # just to visualize. no missing value.
df.describe()
def plot(feature):

    fig,axes=plt.subplots(1,2)

    sns.boxplot(data=df,x=feature,ax=axes[0])

    sns.distplot(a=df[feature],ax=axes[1],color='#ff4125')

    fig.set_size_inches(15,5)
plot('Pregnancies')
plot('Glucose')
plot('BloodPressure')
plot('SkinThickness')
plot('Insulin')
plot('BMI')
plot('DiabetesPedigreeFunction')
plot('Age')
sns.countplot(data=df,x='Outcome')
df['Outcome'].value_counts()
# drawing features against the target variable.



def plot_against_target(feature):

    sns.factorplot(data=df,y=feature,x='Outcome',kind='box')

    fig=plt.gcf()

    fig.set_size_inches(7,7)
plot_against_target('Glucose') # 0 for no diabetes and 1 for presence of it
plot_against_target('BloodPressure')
plot_against_target('SkinThickness')
plot_against_target('Age') 
df.shape
df.head(10)
scaler=MinMaxScaler()

scaled_df=scaler.fit_transform(df.drop('Outcome',axis=1))

X=scaled_df

Y=df['Outcome'].as_matrix()
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=42)
clf_lr=LogisticRegression()

clf_lr.fit(x_train,y_train)

pred=clf_lr.predict(x_test)

print(accuracy_score(pred,y_test))
clf_knn=KNeighborsClassifier()

clf_knn.fit(x_train,y_train)

pred=clf_knn.predict(x_test)

print(accuracy_score(pred,y_test))
clf_svm=SVC()

clf_svm.fit(x_train,y_train)

pred=clf_svm.predict(x_test)

print(accuracy_score(pred,y_test))
clf_dt=DecisionTreeClassifier()

clf_dt.fit(x_train,y_train)

pred=clf_dt.predict(x_test)

print(accuracy_score(pred,y_test))
clf_rf=RandomForestClassifier()

clf_rf.fit(x_train,y_train)

pred=clf_rf.predict(x_test)

print(accuracy_score(pred,y_test))
clf_gb=GradientBoostingClassifier()

clf_gb.fit(x_train,y_train)

pred=clf_gb.predict(x_test)

print(accuracy_score(pred,y_test))
models=[LogisticRegression(),LinearSVC(),SVC(kernel='rbf'),KNeighborsClassifier(),RandomForestClassifier(),

        DecisionTreeClassifier(),GradientBoostingClassifier(),GaussianNB()]

model_names=['LogisticRegression','LinearSVM','rbfSVM','KNearestNeighbors','RandomForestClassifier','DecisionTree',

             'GradientBoostingClassifier','GaussianNB']



acc=[]

d={}



for model in range(len(models)):

    clf=models[model]

    clf.fit(x_train,y_train)

    pred=clf.predict(x_test)

    acc.append(accuracy_score(pred,y_test))

     

d={'Modelling Algo':model_names,'Accuracy':acc}
acc_frame=pd.DataFrame(d)

acc_frame
sns.barplot(y='Modelling Algo',x='Accuracy',data=acc_frame)
clf_svm=SVC()

clf_svm.fit(x_train,y_train)

pred=clf_svm.predict(x_test)

print(accuracy_score(pred,y_test))
ids=[]

for i,obs in enumerate(x_test):

    s='id'+'_'+str(i)

    ids.append(s)
# ids

d={'Ids':ids,'Outcome':pred}

final_pred=pd.DataFrame(d)

final_pred.to_csv('predictions.csv',index=False)