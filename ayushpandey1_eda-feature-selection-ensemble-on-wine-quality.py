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



#classifiaction.

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC,SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB



#model selection

from sklearn.model_selection import train_test_split,cross_validate

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV



#preprocessing

from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder



#evaluation metrics

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error # for regression

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score  # for classification
df = pd.read_csv(r"../input/winequalityred/winequality-red.csv")

df.head()
df.shape
df.columns
df.info()
msno.matrix(df)  # just to visualize. no missing values.
df.describe()
sns.factorplot(data=df,kind='box')
fig,axes = plt.subplots(5,5)

columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',

       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',

       'pH', 'sulphates', 'alcohol', 'quality']

for i in range (5):

    for j in range (5):

        axes[i,j].hist(x=columns[i+j],data=df,edgecolor='#000000',linewidth=2,color='#ff4125')

        axes[i,j].set_title('Variation of '+columns[i+j])

fig=plt.gcf()

fig.set_size_inches(18,18)

fig.tight_layout()

corr_mat=df.corr()

mask=np.array(corr_mat)

fig=plt.gcf()

fig.set_size_inches(30,12)

sns.heatmap(data=corr_mat,annot=True,cbar=True,square=True)
def plot(feature_x,target='quality'):

    sns.factorplot(x=target,y=feature_x,data=df,kind='bar',size=5,aspect=1)

    sns.factorplot(x=target,y=feature_x,data=df,kind='violin',size=5,aspect=1)

    sns.factorplot(x=target,y=feature_x,data=df,kind='swarm',size=5,aspect=1)
# for fixed acidity.

plot('fixed acidity','quality')
# for alcohol.

plot('alcohol','quality')
lb = LabelEncoder()
df['quality']=lb.fit_transform(df['quality'])
x_train,x_test,y_train,y_test = train_test_split(df.drop('quality',axis=1),df['quality'],test_size=0.25,random_state=42)
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

d

acc_frame=pd.DataFrame(d)

acc_frame
sns.barplot(x='Modelling Algo', y='Accuracy',data=acc_frame)
sns.factorplot(x='Modelling Algo',y='Accuracy',data=acc_frame,kind='point',size=4,aspect=3.5)
def func(x_train,x_test,y_train,y_test,name_scaler):

    models=[LogisticRegression(),LinearSVC(),SVC(kernel='rbf'),KNeighborsClassifier(),RandomForestClassifier(),

        DecisionTreeClassifier(),GradientBoostingClassifier(),GaussianNB()]

    acc_sc=[]

    for model in range(len(models)):

        clf=models[model]

        clf.fit(x_train,y_train)

        pred=clf.predict(x_test)

        acc_sc.append(accuracy_score(pred,y_test))

        

    acc_frame[name_scaler]=np.array(acc_sc)
scalers= [MinMaxScaler(),StandardScaler()]

names=['Acc_Min_Max_Scaler','Acc_Standard_Scaler']

for scale in range(len(scalers)):

    scaler=scalers[scale]

    scaler.fit(df)

    scaled_df=scaler.transform(df)

    X=scaled_df[:,0:11]

    Y=df['quality']

    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)

    func(x_train,x_test,y_train,y_test,names[scale])
acc_frame
# just to visualize the accuracies.

sns.barplot(y='Modelling Algo',x='Accuracy',data=acc_frame)
sns.barplot(y='Modelling Algo',x='Acc_Min_Max_Scaler',data=acc_frame)
sns.barplot(y='Modelling Algo',x='Acc_Standard_Scaler',data=acc_frame)
# preparing the features by using a StandardScaler as it gave better resluts.

scaler=StandardScaler()

scaled_df=scaler.fit_transform(df)

X=scaled_df[:,0:11]

Y=df['quality']

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)
params_dict={'C':[0.001,0.01,0.1,1,10,100,1000], 'penalty':['l1','l2']}

model=GridSearchCV(estimator=LogisticRegression(),param_grid=params_dict,scoring='accuracy',cv=5)

model.fit(x_train,y_train)
model.best_params_
model.best_score_
pred=model.predict(x_test)

accuracy_score(pred,y_test)
l=[i+1 for i in range(50)]

params_dict={'n_neighbors':l,'n_jobs':[-1]}

clf_knn=GridSearchCV(estimator=KNeighborsClassifier(),param_grid=params_dict,scoring='accuracy',cv=10)

clf_knn.fit(x_train,y_train)
clf_knn.best_params_
clf_knn.best_score_
pred=clf_knn.predict(x_test)

accuracy_score(pred,y_test)
params_dict={'C':[0.001,0.01,0.1,1,10,100],'gamma':[0.001,0.01,0.1,1,10,100],'kernel':['linear','rbf']}

clf_svc=GridSearchCV(estimator=SVC(),param_grid=params_dict,scoring='accuracy',cv=10)

clf_svc.fit(x_train,y_train)
clf_svc.best_score_
clf_svc.best_params_
# now tuning finally around these values of C and gamma and the kernel for 

#further increasing the accuracy.



params_dict={'C':[0.90,0.92,0.96,0.98,1.0,1.2,1.5],'gamma':[0.90,0.92,0.96,0.98,1.0,1.2,1.5],'kernel':['linear','rbf']}

clf_svm=GridSearchCV(estimator=SVC(),param_grid=params_dict,scoring='accuracy',cv=10)

clf_svm.fit(x_train,y_train)
clf_svm.best_score_
clf_svm.best_params_
pred=clf_svm.predict(x_test)

accuracy_score(pred,y_test)
params_dict={'n_estimators':[500],'max_features':['auto','sqrt','log2']}

clf_rf= GridSearchCV(estimator=RandomForestClassifier(n_jobs=-1),param_grid=params_dict,

                     scoring='accuracy',cv=10)

clf_rf.fit(x_train,y_train)
clf_rf.best_score_
clf_rf.best_params_
pred=clf_rf.predict(x_test)

accuracy_score(pred,y_test)
clf_gb=GridSearchCV(estimator=GradientBoostingClassifier(),cv=10,param_grid=

                    dict({'n_estimators':[500]}))

clf_gb.fit(x_train,y_train)
clf_gb.best_score_
clf_gb.best_params_
pred=clf_rf.predict(x_test)

accuracy_score(pred,y_test)
clf_dt=GridSearchCV(estimator=DecisionTreeClassifier(),

                    scoring='accuracy',cv=10,param_grid=dict({'max_depth':[3]}))

clf_dt.fit(x_train,y_train)
clf_dt.best_score_
clf_dt.best_params_
pred=clf_dt.predict(x_test)

accuracy_score(pred,y_test)
