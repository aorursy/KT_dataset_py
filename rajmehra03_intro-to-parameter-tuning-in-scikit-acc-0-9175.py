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



#classifiaction.

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC,SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

 



#regression

from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV

from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor



#model selection

from sklearn.model_selection import train_test_split,cross_validate

from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV



#preprocessing

from sklearn.preprocessing import MinMaxScaler,StandardScaler,Imputer,LabelEncoder



#evaluation metrics

from sklearn.metrics import mean_squared_log_error,mean_squared_error, r2_score,mean_absolute_error # for regression

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score  # for classification
df=pd.read_csv(r'../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
df.shape
df.columns # the quality is the target variable that we have to predict.
df.info()
df.isnull().sum() # no null or Nan values.
msno.matrix(df)  # just to visualize. no missing values.
df.describe(include='all')
#fixed acidity.

sns.factorplot(data=df,kind='box',size=10,aspect=2.5) # the values are distributed over a very small scale.
# using a histogram.

fig,axes=plt.subplots(5,5)

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

#corelation matrix.

cor_mat= df.corr()

mask = np.array(cor_mat)

mask[np.tril_indices_from(mask)] = False

fig=plt.gcf()

fig.set_size_inches(30,12)

sns.heatmap(data=cor_mat,mask=mask,square=True,annot=True,cbar=True)
# can remove some highly corelated features but for now let us keep them.
def plot(feature_x,target='quality'):

    sns.factorplot(x=target,y=feature_x,data=df,kind='bar',size=5,aspect=1)

    sns.factorplot(x=target,y=feature_x,data=df,kind='violin',size=5,aspect=1)

    sns.factorplot(x=target,y=feature_x,data=df,kind='swarm',size=5,aspect=1)

    
# for fixed acidity.

plot('fixed acidity','quality')
# for alcohol.

plot('alcohol','quality')
# similarly for other variables.
bins = (2, 6.5, 8)

group_names = ['bad', 'good']

df['quality'] = pd.cut(df['quality'], bins = bins, labels = group_names)
label_quality = LabelEncoder()
#Bad becomes 0 and good becomes 1 

df['quality'] = label_quality.fit_transform(df['quality'])
x_train,x_test,y_train,y_test=train_test_split(df.drop('quality',axis=1),df['quality'],test_size=0.25,random_state=42)
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
sns.barplot(y='Modelling Algo',x='Accuracy',data=acc_frame)
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

    
scalers=[MinMaxScaler(),StandardScaler()]

names=['Acc_Min_Max_Scaler','Acc_Standard_Scaler']

for scale in range(len(scalers)):

    scaler=scalers[scale]

    scaler.fit(df)

    scaled_df=scaler.transform(df)

    X=scaled_df[:,0:11]

    Y=df['quality'].as_matrix()

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

Y=df['quality'].as_matrix()

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)
params_dict={'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000],'penalty':['l1','l2']}

clf_lr=GridSearchCV(estimator=LogisticRegression(),param_grid=params_dict,scoring='accuracy',cv=10)

clf_lr.fit(x_train,y_train)
clf_lr.best_params_
clf_lr.best_score_ # the best accuracy obtained by Grid search on the train set.
pred=clf_lr.predict(x_test)

accuracy_score(pred,y_test)
l=[i+1 for i in range(50)]

params_dict={'n_neighbors':l,'n_jobs':[-1]}

clf_knn=GridSearchCV(estimator=KNeighborsClassifier(),param_grid=params_dict,scoring='accuracy',cv=10)

clf_knn.fit(x_train,y_train)

clf_knn.best_score_
clf_knn.best_params_
pred=clf_knn.predict(x_test)

accuracy_score(pred,y_test)   # actual accuarcy on our test set.
params_dict={'C':[0.001,0.01,0.1,1,10,100],'gamma':[0.001,0.01,0.1,1,10,100],'kernel':['linear','rbf']}

clf=GridSearchCV(estimator=SVC(),param_grid=params_dict,scoring='accuracy',cv=10)

clf.fit(x_train,y_train)
clf.best_score_
clf.best_params_
# now tuning finally around these values of C and gamma and the kernel for further increasing the accuracy.

params_dict={'C':[0.90,0.92,0.96,0.98,1.0,1.2,1.5],'gamma':[0.90,0.92,0.96,0.98,1.0,1.2,1.5],'kernel':['linear','rbf']}

clf_svm=GridSearchCV(estimator=SVC(),param_grid=params_dict,scoring='accuracy',cv=10)

clf_svm.fit(x_train,y_train)
clf_svm.best_score_
clf_svm.best_params_
pred=clf_svm.predict(x_test)

accuracy_score(pred,y_test)   # actual accuarcy on our test set.
#### HENCE TILL NOW THE BEST ACCURACY IS GIVEN BY SVM WITH rbf KERNEL WITH  C=1.5 and gamma=0.90 .
params_dict={'n_estimators':[500],'max_features':['auto','sqrt','log2']}

clf_rf=GridSearchCV(estimator=RandomForestClassifier(n_jobs=-1),param_grid=params_dict,scoring='accuracy',cv=10)

clf_rf.fit(x_train,y_train)
clf_rf.best_score_
clf_rf.best_params_
pred=clf_rf.predict(x_test)

accuracy_score(pred,y_test)   # actual accuarcy on our test set.
clf_gb=GridSearchCV(estimator=GradientBoostingClassifier(),cv=10,param_grid=dict({'n_estimators':[500]}))

clf_gb.fit(x_train,y_train)
clf_gb.best_score_
clf_gb.best_params_
pred=clf_gb.predict(x_test)

accuracy_score(pred,y_test)