import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



import plotly.express as px

import plotly.graph_objects as go



from plotly.subplots import make_subplots



#Showing full path of datasets

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# Disable warnings 

import warnings

warnings.filterwarnings('ignore')
breast=pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")
breast.head()
breast.info()
breast.drop(columns=["id","Unnamed: 32"],axis=1,inplace=True)
breast.columns
breast.describe()
breast.isna().sum()
breast.skew(axis=0)
breast['diagnosis'].value_counts()
sns.countplot(x="diagnosis",data=breast)
diagnosis = breast['diagnosis']

breast_mean = breast.iloc[:,0:11]

breast_se = pd.concat([breast.iloc[:,11:21],diagnosis],axis=1)

breast_worst = pd.concat([breast.iloc[:,21:31],diagnosis],axis=1)



display(breast_mean)

display(breast_se)

display(breast_worst)

import plotly.graph_objects as go

fig = make_subplots(rows=5,cols=2,subplot_titles=("Area_mean",'Texture_mean',

                                                 "radius_mean","compactness_mean",

                                                 "perimeter_mean","concavity_mean",

                                                 "concave points_mean","symmetry_mean",

                                                 "fractal_dimension_mean","smoothness_mean"))

fig.add_trace(go.Box(x=breast_mean['diagnosis'],y=breast_mean['area_mean'],name='area_mean'),row=1,col=1)

fig.add_trace(go.Box(x=breast_mean['diagnosis'],y=breast_mean['texture_mean'],name='texture_mean'),row=1,col=2)

fig.add_trace(go.Box(x=breast_mean['diagnosis'],y=breast_mean['radius_mean'],name='radius_mean'),row=2,col=1)

fig.add_trace(go.Box(x=breast_mean['diagnosis'],y=breast_mean['compactness_mean'],name='compactness_mean'),row=2,col=2)

fig.add_trace(go.Box(x=breast_mean['diagnosis'],y=breast_mean['perimeter_mean'],name='perimeter_mean'),row=3,col=1)

fig.add_trace(go.Box(x=breast_mean['diagnosis'],y=breast_mean['concavity_mean'],name='concavity_mean'),row=3,col=2)

fig.add_trace(go.Box(x=breast_mean['diagnosis'],y=breast_mean['concave points_mean'],name='concave points_mean'),row=4,col=1)

fig.add_trace(go.Box(x=breast_mean['diagnosis'],y=breast_mean['symmetry_mean'],name='symmetry_mean'),row=4,col=2)

fig.add_trace(go.Box(x=breast_mean['diagnosis'],y=breast_mean['fractal_dimension_mean'],name='fractal_dimension_mean'),row=5,col=1)

fig.add_trace(go.Box(x=breast_mean['diagnosis'],y=breast_mean['smoothness_mean'],name='smoothness_mean'),row=5,col=2)



# Update title and height

fig.update_layout(title_text="Breast mean Visualizations", height=1500,width=1000)
for col in breast_mean.columns:

    if col != 'diagnosis':

        print(col+' vs diagnosis')

        fig = px.box(breast_mean,x='diagnosis',y=col,color='diagnosis',width=500,height=500)

        fig.show()
for col in breast_se.columns:

    if col != 'diagnosis':

        print(col+' vs diagnosis')

        fig = px.box(breast_se,x='diagnosis',y=col,color='diagnosis',width=500,height=500)

        fig.show()

        
for col in breast_worst.columns:

    if col != 'diagnosis':

        print(col+' vs diagnosis')

        fig = px.box(breast_worst,x='diagnosis',y=col,color='diagnosis',width=500,height=500)

        fig.show()
breast.isna().sum()
plt.figure(figsize=(15,10))

sns.heatmap(breast_mean.corr(),annot=True,vmin=0,vmax=1,cmap="YlGnBu")

plt.figure(figsize=(15,10))

sns.heatmap(breast_se.corr(),annot=True,vmin=0,vmax=1,cmap="YlGnBu")

plt.figure(figsize=(15,10))



sns.heatmap(breast_worst.corr(),annot=True,vmin=0,vmax=1,cmap="YlGnBu")
sns.jointplot(x='radius_mean',y='perimeter_mean',data=breast_mean,kind='reg')
sns.jointplot(x='area_mean',y='radius_mean',data=breast_mean,kind='reg')
sns.jointplot(x='area_mean',y='perimeter_mean',data=breast_mean,kind='reg')
sns.jointplot(x='concavity_mean',y='concave points_mean',data=breast_mean,kind='reg')
sns.jointplot(x='fractal_dimension_mean',y='area_mean',data=breast_mean,kind='reg')
y=breast.diagnosis

x=breast.drop(columns="diagnosis",axis=1)

x.head()
##Standardize data

breast_dia=y

breast_x=x

breast_2=(breast_x-breast_x.mean())/breast_x.std()

breast_x=pd.concat([y,breast_2.iloc[:,0:10]],axis=1)

breast_x=pd.melt(breast_x,id_vars='diagnosis',

                      var_name='features',

                      value_name='value')

plt.figure(figsize=(12,10))

sns.violinplot(x='features',y='value',hue='diagnosis',split=True,data=breast_x)

plt.xticks(rotation=90)



##next 10 features

breast_x=pd.concat([y,breast_2.iloc[:,10:20]],axis=1)

breast_x=pd.melt(breast_x,id_vars='diagnosis',var_name='features',value_name='value')



plt.figure(figsize=(12,10))

sns.violinplot(x='features',y='value',hue='diagnosis',data=breast_x,split=True)

plt.xticks(rotation=90)
##next 10 features

breast_x=pd.concat([y,breast_2.iloc[:,20:31]],axis=1)

breast_x=pd.melt(breast_x,id_vars='diagnosis',var_name='features',value_name='value')

plt.figure(figsize=(12,10))

sns.violinplot(x='features',y='value',hue='diagnosis',data=breast_x,split=True)

plt.xticks(rotation=90)

##pairplot for 5 features at a time
temp=pd.concat([y,x.iloc[:,0:5]],axis=1)

temp.shape

sns.pairplot(data=temp,hue="diagnosis")

plt.figure(figsize=(12,10))
temp=pd.concat([y,x.iloc[:,5:10]],axis=1)

temp.shape

sns.pairplot(data=temp,hue="diagnosis")

plt.figure(figsize=(12,10))
x
y
breast_swarm_dia=y

breast_x=x

breast_s_2=(breast_x-breast_x.mean())/(breast_x.std())

breast_x=pd.concat([y,breast_s_2.iloc[:,0:10]],axis=1)

breast_x=pd.melt(breast_x,id_vars="diagnosis",

    var_name="features",

    value_name="value")

breast_x
plt.figure(figsize=(12,10))

sns.swarmplot(x="features",y="value",hue="diagnosis",data=breast_x)

plt.xticks(rotation=90)

breast=pd.concat([y,breast_s_2.iloc[:,10:21]],axis=1)

breast=pd.melt(breast,id_vars="diagnosis",

    var_name="features",

    value_name="value")

plt.figure(figsize=(12,10))

sns.swarmplot(x="features",y="value",hue="diagnosis",data=breast_x)

plt.xticks(rotation=90)
breast=pd.concat([y,breast_s_2.iloc[:,21:30]],axis=1)

breast=pd.melt(breast,id_vars="diagnosis",

    var_name="features",

    value_name="value")

plt.figure(figsize=(12,10))

sns.swarmplot(x="features",y="value",hue="diagnosis",data=breast_x)

plt.xticks(rotation=90)
##pairplot

plt.figure(figsize=(18,18))

sns.heatmap(x.corr(),annot=True,linewidths=.5,fmt='.1f')
##Feature selection



Features=['smoothness_mean', 'compactness_mean', 'concavity_mean','symmetry_mean', 'fractal_dimension_mean', 'texture_se','smoothness_se',

       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',

       'fractal_dimension_se','texture_worst',

       'perimeter_worst','smoothness_worst',

       'compactness_worst','symmetry_worst', 'fractal_dimension_worst']



selected_features=x.drop(columns=Features,axis=1)

selected_features=selected_features.drop(columns="texture_mean",axis=1)

selected_features
y
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(selected_features,y,test_size=0.3,random_state=10)
X_train.shape
X_test.shape

selected_features.describe()
##As we see in describe values are not proper we need to standardize
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

X_train2=scaler.fit_transform(X_train)

X_test2=scaler.transform(X_test)

X_train2.shape

#but



X_train2
##As features are not independent ,its not proper to use naive bayes

##So we'kll just see to our practice
from sklearn.naive_bayes import GaussianNB

gnb=GaussianNB()

gnb.fit(X_train2,y_train)

y_pred2=gnb.predict(X_test2)
from sklearn.metrics import accuracy_score,confusion_matrix

print("Accuracy using accuracy_score is {}".format(accuracy_score(y_test,y_pred2)))
cm=confusion_matrix(y_test,y_pred2)

print("COnfusion matrix score is \n{}".format(cm))
sns.heatmap(cm,annot=True,

    fmt='d')
#This wont work as y is text classification

#from sklearn.metrics import f1_score

#print("Accuracy f1score is {}".format(f1_score(y_test,y_pred)))
x
y
from sklearn.model_selection import train_test_split

X_train1,X_test1,y_train1,y_test1=train_test_split(x,y,test_size=0.3,random_state=10)

from sklearn.preprocessing import StandardScaler

scaler1=StandardScaler()

X_train1=scaler1.fit_transform(X_train1)

X_test1=scaler1.transform(X_test1)

from sklearn.naive_bayes import GaussianNB

gnb1=GaussianNB()

gnb1.fit(X_train1,y_train1)

y_pred1=gnb1.predict(X_test1)
from sklearn.metrics import accuracy_score,confusion_matrix

print("Accuracy using accuracy_score is {}".format(accuracy_score(y_test,y_pred1)))

from sklearn.feature_selection import SelectKBest,chi2

selectbest10=SelectKBest(chi2,k=10)

X_best_10=selectbest10.fit(X_train,y_train)

X_best_10
X_train
print("Scores :",X_best_10.scores_)

print("Features:",X_train.columns)
from sklearn.feature_selection import SelectKBest,chi2

selectbest5=SelectKBest(chi2,5)

X_best_5=selectbest5.fit(x,y)

print("Scores:",X_best_5.scores_)

print("Features:",x.columns)
#x5=["area_mean","area_se","texture_mean","concavity_worst","compactness_mean"]

featuresfor5=['radius_mean','perimeter_mean',

       'smoothness_mean', 'concavity_mean',

       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',

       'radius_se', 'texture_se', 'perimeter_se', 'smoothness_se',

       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',

       'fractal_dimension_se', 'radius_worst', 'texture_worst',

       'perimeter_worst', 'area_worst', 'smoothness_worst',

       'compactness_worst', 'concave points_worst',

       'symmetry_worst', 'fractal_dimension_worst']

selected_for5=x.drop(columns=featuresfor5,axis=1)

selected_for5
x.columns
y
from sklearn.model_selection import train_test_split

X_train5,X_test5,y_train5,y_test5=train_test_split(selected_for5,y,test_size=0.3,random_state=10)





 

from sklearn.preprocessing import StandardScaler

scaler5=StandardScaler()

X_train5=scaler5.fit_transform(X_train5)

X_test5=scaler5.transform(X_test5)
from sklearn.naive_bayes import GaussianNB

gnb5=GaussianNB()

gnb5.fit(X_train5,y_train5)

y_pred5=gnb5.predict(X_test5)
from sklearn.metrics import accuracy_score

print("Accuracy using 5 selected features is {}".format(accuracy_score(y_test5,y_pred5)))
X_train
y_train
from sklearn.ensemble import RandomForestClassifier

rfclf1=RandomForestClassifier()

rfclf1.fit(X_train,y_train)

y_predrfc=rfclf1.predict(X_test)

from sklearn.metrics import accuracy_score

print("Accuracy using rfclassifier is {}".format(accuracy_score(y_test,y_predrfc)))