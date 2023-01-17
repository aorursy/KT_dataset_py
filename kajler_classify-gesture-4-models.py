import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt

import os

import warnings

warnings.filterwarnings("ignore")
rock_data=pd.read_csv("../input/0.csv",header=None) #Rock gesture signals

scis_data=pd.read_csv("../input/1.csv",header=None)  #Siccors gesture signals

paper_data=pd.read_csv("../input/2.csv",header=None)  #Paper gesture signals

ok_data=pd.read_csv("../input/3.csv",header=None)  #Ok gesture signals

rock_data.head()
print("Rock Shape: ",rock_data.shape,

      "\nScissor Shape: ",scis_data.shape,

      "\nPaper Shape: ",paper_data.shape,

      "\nOK Shape: ",ok_data.shape)
def plot_sensor(data,name,color):

    color_list=["navy","darkmagenta","red","black"]

    fig, ax = plt.subplots(2,4, figsize=(20,12))

    sns.set(style="white")

    sns.set(style="whitegrid")

    x=0

    for i in range(2):

        for j in range(4):

            plt.suptitle(name)

            #rock_data.iloc[:,i].plot.hist(bins=10,ax=ax[i][j],grid=True)

            sns.distplot(data.iloc[:,x],kde=False,ax=ax[i][j],color=color_list[color],bins=15);

            x+=1

            if i==1:

                ax[i][j].set_title("Sensor_"+str(j+5))

            else:

                ax[i][j].set_title("Sensor_"+str(j+1))

    plt.show()

    
plot_sensor(rock_data,"Rock_Data",0)
plot_sensor(scis_data,"Scissor_Data",1)
plot_sensor(paper_data,"Paper_Data",2)
plot_sensor(ok_data,"OK_Data",3)
colors=["forestgreen","teal","crimson","chocolate","darkred","lightseagreen","orangered","chartreuse"]

time_rock=rock_data.iloc[:,0:8]

time_rock.index=pd.to_datetime(time_rock.index)

time_rock.iloc[:170,:].plot(subplots=True,figsize=(10,10),colors=colors);
time_scis=scis_data.iloc[:,0:8]

time_scis.index=pd.to_datetime(time_scis.index)

time_scis.iloc[:170,:].plot(subplots=True,figsize=(10,10),colors=colors);
datas=pd.concat([rock_data,scis_data,paper_data,ok_data],ignore_index=True)

df=datas.copy()

#Kolon isimlerini String yaptım.

liste=[str(x) for x in range(65)]

df.columns=liste 

df.head()
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

X=df.drop(["64"],axis=1)

y=df["64"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)



naive=GaussianNB().fit(X_train,y_train)

naive
y_pred=naive.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))
y_test.head()
y_pred[0:5]
from sklearn.neighbors import KNeighborsClassifier

kneigh=KNeighborsClassifier()

k_model=kneigh.fit(X_train,y_train)

k_model
y_pred=k_model.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))
params={"n_neighbors": np.arange(1,10)}

knn=KNeighborsClassifier()

knn_cv=GridSearchCV(knn,params,cv=10)

knn_cv.fit(X_train,y_train)
knn_cv.best_params_
knn_model=KNeighborsClassifier(n_neighbors=9)

knn_tuned=knn_model.fit(X_train,y_train)

knn_tuned
y_pred=knn_tuned.predict(X_test)

print("Accuracy:",accuracy_score(y_test,y_pred))
from sklearn.preprocessing import StandardScaler



scaler=StandardScaler()

scaler.fit(X_train)

X_train_scaled=scaler.transform(X_train)

X_test_scaled=scaler.transform(X_test)



from sklearn.neural_network import MLPClassifier

mlp=MLPClassifier().fit(X_train_scaled,y_train)

mlp
y_pred=mlp.predict(X_test_scaled)

accuracy_score(y_test,y_pred)
from catboost import CatBoostClassifier



cat_model=CatBoostClassifier(silent=True).fit(X_train,y_train)



y_pred=cat_model.predict(X_test)

accuracy_score(y_test,y_pred)