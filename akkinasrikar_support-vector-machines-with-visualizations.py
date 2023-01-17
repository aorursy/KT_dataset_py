import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import plotly.graph_objects as go
import plotly.express as px
from matplotlib import style
style.use("fivethirtyeight")
dataset=pd.read_csv("/kaggle/input/IRIS.csv")
dataset.head()
dataset.describe().transpose()
dataset.corr()
plt.figure(figsize=(10,6))
sns.heatmap(dataset.corr(),annot=True,yticklabels=False)
plt.figure(figsize=(10,6))
sns.scatterplot(x="sepal_length",y="petal_length",data=dataset,hue="species")
plt.figure(figsize=(10,6))
sns.scatterplot(x="sepal_length",y="species",data=dataset,hue="species")
plt.figure(figsize=(10,6))
sns.scatterplot(x="sepal_width",y="sepal_length",data=dataset,hue="species")
ax=plt.figure(figsize=(12,8))
ax.add_subplot(121)
sns.boxplot(x="species",y="sepal_length",data=dataset,hue="species")
ax.add_subplot(122)
sns.boxplot(x="species",y="petal_length",data=dataset,hue="species")
plt.figure(figsize=(10,6))
dataset[["sepal_length","petal_length","petal_width"]].plot()
df=dataset[["sepal_length","petal_length","petal_width"]]
fig=go.Figure(data=[go.Surface(z=df.values)])
fig.update_layout(title="Iris dataset",autosize=False,
           width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))
fig.show()
fig=go.Figure(data=[go.Surface(z=df.values)])
fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))
fig.update_layout(title="Iris dataset",autosize=False,
           width=500, height=500,
           scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64),
            margin=dict(l=65, r=50, b=65, t=90))
fig.show()
from sklearn.preprocessing import LabelEncoder
encoding=LabelEncoder()
dataset["speciesID"]=encoding.fit_transform(dataset["species"])

fig=px.parallel_coordinates(dataset,color="speciesID",labels={"species":"Species",
                                                           "sepal_width":"sepal width",
                                                           "sepal_length":"sepal length",
                                                           "petal_length":"petal length",
                                                           "petal_width":"petal width"},
                           color_continuous_scale=px.colors.diverging.Tealrose,
                           color_continuous_midpoint=2)
fig.show()
fig = px.scatter_3d(df, x="sepal_length", y="petal_length", z="petal_width")
fig.show()
plt.figure(figsize=(10,6))
sns.pairplot(dataset,hue="species")

#lets consider only two species
newdata=dataset[dataset["species"]!="Iris-virginica"]
newdata.head()

x=newdata.drop(["sepal_length","sepal_width","species","speciesID"],axis=1)
y=newdata["speciesID"]
plt.figure(figsize=(10,6))
plt.scatter(x["petal_length"],x["petal_width"],c=y,s=70)
from sklearn.svm import SVC
model=SVC(kernel="linear",C=1)
model.fit(x,y)
sp=model.support_vectors_
sp
plt.figure(figsize=(10,6))
plt.scatter(x["petal_length"],x["petal_width"],c=y,s=70)
plt.scatter(sp[:,0],sp[:,1],s=70)
plt.title("support vector points")
def svm(c):
    model=SVC(kernel="linear",C=c)
    model.fit(x,y)
    ax=plt.gca()
    plt.scatter(x["petal_length"],x["petal_width"],c=y,s=50)
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    xx=np.linspace(xlim[0],xlim[1],30)
    yy=np.linspace(ylim[0],ylim[1],30)
    XX,YY=np.meshgrid(xx,yy)
    xy=np.vstack([XX.ravel(), YY.ravel()]).T
    Z=model.decision_function(xy).reshape(XX.shape)
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
    plt.scatter(sp[:,0],sp[:,1],s=100,facecolors=None,edgecolors="k",linewidth=1)
    plt.show()
svm(10)

x=np.array(dataset[["sepal_length","petal_length"]])
y=np.array(dataset["speciesID"])
plt.figure(figsize=(10,6))
plt.scatter(x[:,0],x[:,1],c=y,s=70)
def svm(kernel,c,gamma,d):
    model1=SVC(kernel=kernel,C=c,gamma=gamma,degree=d)
    model1.fit(x,y)
    sp=model1.support_vectors_
    x_min,x_max=x[:,0].min()-1,x[:,0].max()+1
    y_min,y_max=x[:,1].min()-1.5,x[:,1].max()+1
    height=(x_max/x_min)/100
    xx,yy=np.meshgrid(np.arange(x_min,x_max,height),np.arange(y_min,y_max,height))
    z=model1.predict(np.c_[xx.ravel(),yy.ravel()])
    z=z.reshape(xx.shape)
    plt.figure(figsize=(10,6))
    plt.contourf(xx,yy,z,cmap=plt.cm.Paired,alpha=0.5)
    plt.scatter(x[:,0],x[:,1],cmap=plt.cm.Paired,c=y)
    plt.xlim(xx.min(),xx.max())
    plt.xlabel("sepal length")
    plt.ylabel("petal length")
    plt.title(f"kernel is {kernel} with c={c},gamma={gamma},degree={d} ")
    plt.show()

svm('linear',100,'auto',1)
c=[0.001,0.1,1,10,100,1000,10000]
for i in c:
    svm('rbf',i,'auto',1)
d=[1,2,3,4,5]
for j in d:
    svm('poly',100,'auto',j)
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
x=dataset.drop(["species","speciesID"],axis=1)
y=dataset["speciesID"]
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=6)
param_grid={'C':[1,10,0.1,0.01,100,1000],
            'gamma':[1,10,0.1,0.01,0.001,0.0001],
            'degree':[1,2,3,4,5,6],
            'kernel':["rbf","linear","poly"]
           }
grid_svm=GridSearchCV(SVC(),param_grid,verbose=3)
grid_svm.fit(x_train,y_train)
print(grid_svm.best_estimator_)
print(grid_svm.best_params_)
svm=SVC(kernel="linear",C=100,degree=1,gamma=0.1)
svm.fit(x_train,y_train)
y_pred=svm.predict(x_test)
print(accuracy_score(y_test,y_pred)*100)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
from sklearn.model_selection import cross_val_score
clf=cross_val_score(svm,x,y,cv=10)
print(f"accuracy is {clf.mean()}")
