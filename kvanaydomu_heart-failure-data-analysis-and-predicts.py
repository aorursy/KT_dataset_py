# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

import plotly.graph_objs as go

from collections import Counter

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBRegressor

from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import LabelEncoder, StandardScaler,OneHotEncoder, OrdinalEncoder

from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_score





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
filePath = "/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv"



heartData = pd.read_csv(filePath)
heartData.head()
heartData.columns
heartData.dtypes
len(heartData.columns)
heartData.describe()
heartData.shape
heartData.columns[heartData.isnull().any()]
numericCol = [col for col in heartData.columns if heartData[col].dtype in ["int64","float64"]]

numericCol
if (len(numericCol) == len(heartData.columns)):

    print("Whole columns dtypes is numeric.")
target = heartData["DEATH_EVENT"]

sns.distplot(target)

plt.title("Distribiton of Death Event")

plt.show()
num_attributes = heartData.drop("DEATH_EVENT",axis=1).copy()



fig = plt.figure(figsize=(12,18))



for i in range(len(num_attributes.columns)):

    fig.add_subplot(4,3,i+1)

    sns.distplot(num_attributes.iloc[:,i].dropna())

    plt.xlabel(num_attributes.columns[i])



plt.show()    
fig = plt.figure(figsize=(12,18))



for i in range(len(num_attributes.columns)):

    fig.add_subplot(4,3,i+1)

    sns.boxplot(num_attributes.iloc[:,i].dropna())

    plt.xlabel(num_attributes.columns[i])



plt.show()    
fig = plt.figure(figsize=(12,18))



for i in range(len(num_attributes.columns)):

    fig.add_subplot(4,3,i+1)

    sns.kdeplot(num_attributes.iloc[:,i].dropna())

    plt.xlabel(num_attributes.columns[i])



plt.show()    
correlation = heartData.corr()

plt.figure(figsize=(12,15))

sns.heatmap(correlation,annot=True,fmt=".2f")



plt.show()
# correlation according to target column that DEATH_EVENT



corrToTarget = heartData.corr()["DEATH_EVENT"].sort_values(ascending=False)

sns.pointplot(x=corrToTarget.index,y=corrToTarget.values)

plt.xticks(rotation=90)

plt.title("Correlation Rates According to Target Column That DEATH_EVENT")

plt.ylabel("Correlation rates")

plt.show()
recover = heartData.age[heartData.DEATH_EVENT==1]

dead = heartData.age[heartData.DEATH_EVENT==0]





trace1 = go.Histogram(

    x=recover,

    opacity=0.75,

    name = "1",

    marker=dict(color='rgba(171, 50, 96, 0.6)'))



trace2 = go.Histogram(

    x=dead,

    opacity=0.75,

    name = "0",

    marker=dict(color='rgba(12, 50, 196, 0.6)'))



data = [trace1, trace2]



layout = go.Layout(barmode='overlay',

                   title='patients age distribiton according to Deaths event',

                   xaxis=dict(title='ages distribition'),

                   yaxis=dict( title='Count'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
sns.boxplot(x=heartData.DEATH_EVENT,y=heartData.age)

plt.xticks(rotation=45)

plt.title("patients age distribiton according to age")

plt.show()
anaemia = heartData.anaemia.value_counts()

anaemia.values
data = {

    "values" : anaemia.values,

    "labels" : anaemia.index,

    "type" : "pie",

    "hoverinfo":"label+percent",

    "hole":.3

}



layout={

    "title":"Rates anaemia bar chart"

}

fig=go.Figure(data=data,layout=layout)

iplot(fig)
belongToRecover = heartData.creatinine_phosphokinase[heartData.DEATH_EVENT == 1]

belongToDead = heartData.creatinine_phosphokinase[heartData.DEATH_EVENT == 0]



trace1 = go.Histogram(

    x=belongToRecover,

    opacity=0.75,

    name = "1",

    marker=dict(color='rgba(171, 50, 96, 0.6)'))



trace2 = go.Histogram(

    x=belongToDead,

    opacity=0.75,

    name = "0",

    marker=dict(color='rgba(12, 50, 196, 0.6)'))



data = [trace1, trace2]



layout = go.Layout(barmode='overlay',

                   title='patients creatinine phosphokinase distribiton according to Deaths event',

                   xaxis=dict(title='ages distribition'),

                   yaxis=dict(title='Count'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
diabetes = heartData.diabetes.value_counts()

diabetes
data = {

    "values" : diabetes.values,

    "labels" : diabetes.index,

    "type" : "pie",

    "hoverinfo":"label+percent",

    "hole":.3

}



layout={

    "title":"Rates diabetes bar chart"

}

fig=go.Figure(data=data,layout=layout)

iplot(fig)
belongToRecover = heartData.ejection_fraction[heartData.DEATH_EVENT == 1]

belongToDead = heartData.ejection_fraction[heartData.DEATH_EVENT == 0]



trace1 = go.Histogram(

    x=belongToRecover,

    opacity=0.75,

    name = "1",

    marker=dict(color='rgba(171, 50, 96, 0.6)'))



trace2 = go.Histogram(

    x=belongToDead,

    opacity=0.75,

    name = "0",

    marker=dict(color='rgba(12, 50, 196, 0.6)'))



data = [trace1, trace2]



layout = go.Layout(barmode='overlay',

                   title='patients ejection fraction distribiton according to Deaths event',

                   xaxis=dict(title='ages distribition'),

                   yaxis=dict(title='Count'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace1 = go.Box(

    x = heartData.DEATH_EVENT,

    y = heartData.ejection_fraction,

        marker = dict(

        color = 'rgb(12, 128, 128)',

    )

)

layout = go.Layout(title="ejection fraction distribiton of deaths event with box plot")

fig = go.Figure(data=data,layout=layout)

iplot(fig)
bloodPressure = heartData.high_blood_pressure.value_counts()
data = {

    "values" : bloodPressure.values,

    "labels" : bloodPressure.index,

    "type" : "pie",

    "hoverinfo":"label+percent",

    "hole":.3

}



layout={

    "title":"Rates high blood pressure bar chart"

}

fig=go.Figure(data=data,layout=layout)

iplot(fig)
belongToRecover = heartData.platelets[heartData.DEATH_EVENT == 1]

belongToDead = heartData.platelets[heartData.DEATH_EVENT == 0]



trace1 = go.Histogram(

    x=belongToRecover,

    opacity=0.75,

    name = "1",

    marker=dict(color='rgba(171, 50, 96, 0.6)'))



trace2 = go.Histogram(

    x=belongToDead,

    opacity=0.75,

    name = "0",

    marker=dict(color='rgba(12, 50, 196, 0.6)'))



data = [trace1, trace2]



layout = go.Layout(barmode='overlay',

                   title='patients platelets distribiton according to Deaths event',

                   xaxis=dict(title='ages distribition'),

                   yaxis=dict(title='Count'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
belongToRecover = heartData.serum_creatinine[heartData.DEATH_EVENT == 1]

belongToDead = heartData.serum_creatinine[heartData.DEATH_EVENT == 0]



trace1 = go.Histogram(

    x=belongToRecover,

    opacity=0.75,

    name = "1",

    marker=dict(color='rgba(171, 50, 96, 0.6)'))



trace2 = go.Histogram(

    x=belongToDead,

    opacity=0.75,

    name = "0",

    marker=dict(color='rgba(12, 50, 196, 0.6)'))



data = [trace1, trace2]



layout = go.Layout(barmode='overlay',

                   title='patients serum creatinine distribiton according to Deaths event',

                   xaxis=dict(title='ages distribition'),

                   yaxis=dict(title='Count'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace1 = go.Box(

    x = heartData.DEATH_EVENT,

    y = heartData.serum_creatinine,

        marker = dict(

        color = 'rgb(12, 128, 128)',

    )

)

layout = go.Layout(title="serum creatinine distribiton of deaths event with box plot")

fig = go.Figure(data=data,layout=layout)

iplot(fig)
heartData.serum_sodium
belongToRecover = heartData.serum_sodium[heartData.DEATH_EVENT == 1]

belongToDead = heartData.serum_sodium[heartData.DEATH_EVENT == 0]



trace1 = go.Histogram(

    x=belongToRecover,

    opacity=0.75,

    name = "1",

    marker=dict(color='rgba(171, 50, 96, 0.6)'))



trace2 = go.Histogram(

    x=belongToDead,

    opacity=0.75,

    name = "0",

    marker=dict(color='rgba(12, 50, 196, 0.6)'))



data = [trace1, trace2]



layout = go.Layout(barmode='overlay',

                   title='patients serum sodium distribiton according to Deaths event',

                   xaxis=dict(title='ages distribition'),

                   yaxis=dict(title='Count'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace1 = go.Box(

    x = heartData.DEATH_EVENT,

    y = heartData.serum_sodium,

        marker = dict(

        color = 'rgb(12, 128, 128)',

    )

)

layout = go.Layout(title="serum sodium distribiton of deaths event with box plot")

fig = go.Figure(data=data,layout=layout)

iplot(fig)
sex=heartData.sex.value_counts()



data = {

    "values" : sex.values,

    "labels" : sex.index,

    "type" : "pie",

    "hoverinfo":"label+percent",

    "hole":.3

}



layout={

    "title":"Rates sex bar chart"

}

fig=go.Figure(data=data,layout=layout)

iplot(fig)
smoke = heartData.smoking.value_counts()



data = {

    "values" : smoke.values,

    "labels" : smoke.index,

    "type" : "pie",

    "hoverinfo":"label+percent",

    "hole":.3

}



layout={

    "title":"Rates smoke bar chart"

}

fig=go.Figure(data=data,layout=layout)

iplot(fig)

belongToRecover = heartData.time[heartData.DEATH_EVENT == 1]

belongToDead = heartData.time[heartData.DEATH_EVENT == 0]



trace1 = go.Histogram(

    x=belongToRecover,

    opacity=0.75,

    name = "1",

    marker=dict(color='rgba(171, 50, 96, 0.6)'))



trace2 = go.Histogram(

    x=belongToDead,

    opacity=0.75,

    name = "0",

    marker=dict(color='rgba(12, 50, 196, 0.6)'))



data = [trace1, trace2]



layout = go.Layout(barmode='overlay',

                   title='patients time distribiton according to Deaths event',

                   xaxis=dict(title='ages distribition'),

                   yaxis=dict(title='Count'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace1 = go.Box(

    x = heartData.DEATH_EVENT,

    y = heartData.time,

        marker = dict(

        color = 'rgb(12, 128, 128)',

    )

)

layout = go.Layout(title="time distribiton of deaths event with box plot")

fig = go.Figure(data=data,layout=layout)

iplot(fig)




death = heartData.DEATH_EVENT.value_counts()



data = {

    "values" : death.values,

    "labels" : death.index,

    "type" : "pie",

    "hoverinfo":"label+percent",

    "hole":.3

}



layout={

    "title":"Rates death bar chart"

}

fig=go.Figure(data=data,layout=layout)

iplot(fig)

def detectOutliers(df,features):

    outlier_indices=[]

    for c in features:

        Q1=np.percentile(df[c],25)

        Q2=np.percentile(df[c],75)

        IQR = Q2-Q1

        outlierStep = IQR*1.5

        outlierListCol = df[(df[c] < Q1-outlierStep) | (df[c]>Q2+outlierStep)].index

        outlier_indices.extend(outlierListCol)

    outlier_indices=Counter(outlier_indices)

    multiple_outliers=list(i for i,v in outlier_indices.items() if v>2)

    return multiple_outliers
heartData.loc[detectOutliers(heartData,heartData.columns)]

# there arent any outliers exist
heartData.dropna(axis=0,subset=["DEATH_EVENT"],inplace=True)

y = heartData.DEATH_EVENT

x = heartData.drop(["DEATH_EVENT"],axis=1)

X_train,X_test,y_train,y_test = train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=0)
X_train.shape
X_test.shape
x.dtypes
constant_num_cols = x.columns

#len(constant_num_cols) 12

myCols = constant_num_cols
X_train = X_train[myCols].copy()

X_test = X_test[myCols].copy()

x = heartData[myCols].copy()
X_train.shape
X_test.shape
numerical_transform_c = Pipeline(steps = [

    ("imputer",SimpleImputer(strategy="mean")),

    ("scaler", StandardScaler())

])



preprocessor = ColumnTransformer(

    transformers=[

        ("num_mean",numerical_transform_c,constant_num_cols)

    ]

)
neighbours = np.arange(1,30)

testAccuracy = []

trainAccuracy = []



for i,k in enumerate(neighbours):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train,y_train)

    trainAccuracy.append(knn.score(X_train,y_train))

    testAccuracy.append(knn.score(X_test,y_test))



plt.figure(figsize=(15,16))

plt.plot(neighbours,testAccuracy,label="Test accuracy")

plt.plot(neighbours,trainAccuracy,label="Train Accuracy")

plt.legend()

plt.title("Detect most suitable value for n_neighbours")

plt.xticks(neighbours)

plt.show()
grid = {"n_neighbors" : np.arange(1,30)}

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn,grid,cv=3)

knn_cv.fit(X_train,y_train)



print("Tuned parameter k : {}".format(knn_cv.best_params_))

print("Best score for Knn = {}".format(knn_cv.best_score_))
model1 = KNeighborsClassifier(n_neighbors=11)



my_pipeline1 = Pipeline(steps = [

    ("preprocessor",preprocessor),

    ("model",model1)

])



preds1 = my_pipeline1.fit(X_train,y_train)



print("KNN algortihms score = {}".format(my_pipeline1.score(X_test,y_test)))
y_pred = my_pipeline1.predict(X_test)

y_true = y_test

cm_1 = confusion_matrix(y_pred,y_true)



f,ax = plt.subplots(figsize=(6,6))

plt.title("KNN algortihms score matrix")

sns.heatmap(cm_1,annot=True,color="red",fmt="0.5f",ax=ax)

plt.show()
k=5

knn_cross_validation = cross_val_score(my_pipeline1,X_train,y_train,cv=k)



print("CV Knn scores = {}".format(knn_cross_validation))

print("CV Knn scores average : {}".format(np.sum(knn_cross_validation)/k))
grid = {"max_iter" : np.arange(50,560,50)}

logreg = LogisticRegression()

log_cv = GridSearchCV(logreg,grid,cv=3)

log_cv.fit(X_train,y_train)



print("Tuned parameter n_estimators : {}".format(log_cv.best_params_))

print("Best score for Logistic regression = {}".format(log_cv.best_score_))
model2 = LogisticRegression(random_state=42,max_iter=150)



my_pipeline2 = Pipeline(steps = [

    ("preprocessor",preprocessor),

    ("model",model2)

])



my_pipeline2.fit(X_train,y_train)



print("Logistic regression score = {}".format(my_pipeline2.score(X_test,y_test)))
y_pred2 = my_pipeline2.predict(X_test)

y_true2 = y_test

cm_2 = confusion_matrix(y_pred2,y_true2)



f,ax1 = plt.subplots(figsize=(6,6))

plt.title("Logistic regression score matrix")

sns.heatmap(cm_2,annot=True,color="red",fmt="0.5f",ax=ax1)
logistic_cross_val = cross_val_score(my_pipeline2,X_train,y_train,cv=k)



print("CV Logistic reg. scores = {}".format(logistic_cross_val))

print("CV Logistic reg. scores average : {}".format(np.sum(logistic_cross_val)/k))
grid = {"n_estimators" : np.arange(50,560,50)}

rfc = RandomForestClassifier()

rfc_cv = GridSearchCV(rfc,grid,cv=3)

rfc_cv.fit(X_train,y_train)



print("Tuned parameter n_estimators : {}".format(rfc_cv.best_params_))

print("Best score for random forest = {}".format(rfc_cv.best_score_))
model3 = RandomForestClassifier(n_estimators=200,random_state=1)

my_pipeline3 = Pipeline(steps = [

    ("preprocessor",preprocessor),

    ("model",model3)

])



my_pipeline3.fit(X_train,y_train)

print("Random Forest Classification Score = {}".format(my_pipeline3.score(X_test,y_test)))
y_pred3 = my_pipeline3.predict(X_test)

y_true3 = y_test

cm_3 = confusion_matrix(y_pred3,y_true3)



f,ax3 = plt.subplots(figsize=(6,6))

sns.heatmap(cm_3,annot=True,fmt="0.5f",ax=ax3)

plt.title("Random Forest Classification Score Matrix")

plt.show()
random_forest_cross_validation = cross_val_score(my_pipeline3,X_train,y_train,cv=k)



print("CV Random Forest class. scores = {}".format(random_forest_cross_validation))

print("CV Random Forest class. scores average : {}".format(np.sum(random_forest_cross_validation)/k))
grid = {"n_estimators" : np.arange(100,1600,100)}

XGbr = XGBRegressor()

XGbr_cv = GridSearchCV(XGbr,grid,cv=3)

XGbr_cv.fit(X_train,y_train)



print("Tuned parameter n_estimators : {}".format(XGbr_cv.best_params_))

print("Best score = {}".format(XGbr_cv.best_score_))
model4 = DecisionTreeClassifier()

my_pipeline4 = Pipeline(steps = [

    ("preprocessor",preprocessor),

    ("model",model4)

])



my_pipeline4.fit(X_train,y_train)

print("Desicion Tree Classification Score = {}".format(my_pipeline4.score(X_test,y_test)))
y_pred4 = my_pipeline4.predict(X_test)

y_true4 = y_test

cm_4 = confusion_matrix(y_pred4,y_true4)



f,ax4 = plt.subplots(figsize=(6,6))

sns.heatmap(cm_4,annot=True,fmt="0.5f")

plt.title("Desicion Tree Classification Score matrix")

plt.show()
decision_tree_cross_validation = cross_val_score(my_pipeline4,X_train,y_train,cv=k)



print("CV Desicion Tree Classification scores = {}".format(decision_tree_cross_validation))

print("CV Desicion Tree Classification. scores average : {}".format(np.sum(decision_tree_cross_validation)/k))
X_train_full,X_test_full,y_train_full,y_test_full = train_test_split(x,y,train_size=0.8,test_size=0.2,random_state=0)
X_train = X_train_full.copy()

X_test = X_test_full.copy()

y=y_train.copy()

X_tr = X_train[myCols].copy()

X_te = X_test[myCols].copy()
myFinalModel = RandomForestClassifier(n_estimators=200,random_state=1)



myFinalPipeline = Pipeline(steps = [

    ("preprocessor",preprocessor),

    ("model",myFinalModel)

])



myFinalPipeline.fit(X_tr,y)



finalPreds = myFinalPipeline.predict(X_te)
output = pd.DataFrame({

    "Id" : X_te.index,

    "DEATH_EVENT" : finalPreds

})
compression_opts = dict(method="zip",archive_name="submission.csv")

output.to_csv("submission.zip",index=False,compression=compression_opts)