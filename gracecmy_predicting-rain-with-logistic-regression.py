import numpy as np

import pandas as pd

pd.set_option("display.max_columns",100)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

params={"figure.facecolor":(0.0,0.0,0.0,0),

        "axes.facecolor":(1.0,1.0,1.0,1),

        "savefig.facecolor":(0.0,0.0,0.0,0)}

plt.rcParams.update(params)

import warnings

warnings.filterwarnings("ignore")
df=pd.read_csv("../input/weather-dataset-rattle-package/weatherAUS.csv")

df.head()
df.drop(["RISK_MM"],axis=1,inplace=True)
df.info()
df["RainTomorrow"].value_counts()
sns.countplot(df["RainTomorrow"],palette=["lightcoral","skyblue"])

plt.ylabel("Count")
df["RainTomorrow"]=df["RainTomorrow"].apply(lambda x:0 if x=="No" else 1)
df.describe().drop(["RainTomorrow"],axis=1).T
for column in df.select_dtypes(exclude="object").drop(["RainTomorrow"],axis=1).columns:

    print(column,":",df[column].isnull().sum(),"missing values.")
fig,axes=plt.subplots(1,2,figsize=(12,5))



df[df.select_dtypes(exclude="object").columns.drop(["Pressure9am","Pressure3pm","RainTomorrow"])].plot(kind="box",color="#AE9CCD",ax=axes[0])

axes[0].set_xticklabels(axes[0].get_xticklabels(),rotation=90)

axes[0].set_ylabel("Measurement")



df[["Pressure9am","Pressure3pm"]].plot(kind="box",color="#AE9CCD",ax=axes[1])
fig,axes=plt.subplots(1,3,figsize=(15,4))



sns.distplot(df["Rainfall"],bins=12,color="lightskyblue",ax=axes[0])

sns.distplot(df["Evaporation"],bins=12,color="lightcoral",ax=axes[1])

sns.distplot(df["WindSpeed9am"],bins=12,color="lightgreen",ax=axes[2])
droppers=df.loc[(df["Rainfall"]>300)|(df["Evaporation"]>100)|(df["WindSpeed9am"]>100)]

df.drop(droppers.index,inplace=True)
print("We have dropped {num1} rows, so now instead of the initial 142193 readings, we have {num2}.".format(num1=142193-df.shape[0],num2=df.shape[0]))
df.select_dtypes(include="object").describe()
print("{num} missing values.".format(num=df["Date"].isnull().sum()))
df["Date"]=pd.to_datetime(df["Date"])
df["Month"]=df["Date"].dt.month
df.drop(["Date"],axis=1,inplace=True)

df.head(2)
print("{num} missing values.".format(num=df["Location"].isnull().sum()))
df["Location"].value_counts()
print("{num} missing values.".format(num=df["WindGustDir"].isnull().sum()))
df["WindGustDir"].value_counts()
print("{num} missing values.".format(num=df["WindDir9am"].isnull().sum()))
df["WindDir9am"].value_counts()
print("{num} missing values.".format(num=df["WindDir3pm"].isnull().sum()))
df["WindDir3pm"].value_counts()
print("{num} missing values.".format(num=df["RainToday"].isnull().sum()))
df["RainToday"].value_counts()
df["RainToday"]=df["RainToday"].apply(lambda x:0 if x=="No" else 1)

df.head(2)
df.head()
from sklearn.model_selection import train_test_split
x=df.drop(["RainTomorrow"],axis=1)

y=df["RainTomorrow"]



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=7)
print("Training set shape:",x_train.shape)

print("Testing set shape:",x_test.shape)
x_train.isnull().sum()
x_test.isnull().sum()
for df in [x_train,x_test]:

    for col in df.select_dtypes(exclude="object").columns:

        col_median=x_train[col].median()

        df[col].fillna(col_median,inplace=True)
x_train.isnull().sum()
x_test.isnull().sum()
for df in [x_train,x_test]:

    for col in df.select_dtypes("object").columns:

        col_mode=x_train[col].mode()[0]

        df[col].fillna(col_mode,inplace=True)
x_train.isnull().sum()
x_test.isnull().sum()
for col in x_train.select_dtypes("object").columns:

    x_train=pd.concat([x_train,pd.get_dummies(x_train[col],drop_first=True)],axis=1)

    x_train.drop([col],axis=1,inplace=True)
x_train.head(2)
for col in x_test.select_dtypes("object").columns:

    x_test=pd.concat([x_test,pd.get_dummies(x_test[col],drop_first=True)],axis=1)

    x_test.drop([col],axis=1,inplace=True)
x_test.head(2)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()



x_train=pd.DataFrame(scaler.fit_transform(x_train),columns=x_train.columns)

x_test=pd.DataFrame(scaler.transform(x_test),columns=x_test.columns)
x_train.head(2)
x_test.head(2)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(random_state=7)

model.get_params()
model.fit(x_train,y_train)



y_predict=model.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,y_predict))
def cm(predictions):

    cm_matrix=pd.DataFrame(data=confusion_matrix(y_test,predictions),columns=["No Rain","Rain"],index=["No Rain","Rain"])

    sns.heatmap(cm_matrix,annot=True,square=True,fmt="d",cmap="Purples",linecolor="w",linewidth=2)

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.yticks(va="center")
cm(y_predict)
print("Training set score: {num:.4f}.".format(num=model.score(x_train,y_train)))

print("Testing set score: {num:.4f}.".format(num=model.score(x_test,y_test)))
from sklearn.model_selection import GridSearchCV
parameters=[{"penalty":["l1","l2","elasticnet"]},{"C":[0.1,1,10,100]},{"class_weight":["balanced",None]},

            {"solver":["newton-cg","lbfgs","liblinear","sag","saga"]},{"multi_class":["auto","ovr","multinomial"]}]



grid=GridSearchCV(estimator=model,param_grid=parameters,refit=True,cv=5,verbose=1)



grid.fit(x_train,y_train)
grid_results=pd.DataFrame(grid.cv_results_)

grid_results
print("Using grid search, the best parameters of the model should be",grid.best_params_)
y_predict2=grid.predict(x_test)
print(classification_report(y_test,y_predict2))
cm(y_predict2)
print("The grid seach only improved the model's accuracy from {num1:.6f} to {num2:.6f}.".format(num1=model.score(x_test,y_test),num2=grid.score(x_test,y_test)))