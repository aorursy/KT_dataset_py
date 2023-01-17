import numpy as np

import pandas as pd

import plotly.express as px #I will only be using plotly express to visualise the data
df=pd.read_csv("../input/80-cereals/cereal.csv")

df.head()
df.rename(columns=lambda x:x.title(),inplace=True)
df.rename(columns={"Potass":"Potassium"},inplace=True)
df.info()
df.describe(include="all").T
df.loc[(df["Carbo"]<0)|(df["Sugars"]<0)|(df["Potassium"]<0)]=df.loc[(df["Carbo"]<0)|(df["Sugars"]<0)|(df["Potassium"]<0)].replace(-1,0)
px.bar(df.sort_values(by="Rating",ascending=True),x="Rating",y="Name",

      labels={"Name":"","Rating":"Rating (%)"},hover_name="Name",hover_data={"Name":False},

      color="Rating",color_continuous_scale="tealgrn",template="plotly")
df["Mfr"].unique()
df["Type"].unique()
px.sunburst(df,path=["Mfr","Type"])
px.box(df,x="Mfr",y="Rating",labels={"Mfr":"Manufacturer","Rating":"Rating (%)"},

       title="Rating Distribution",color="Mfr")
manu=df.drop(["Name","Type","Rating","Shelf","Weight","Cups"],axis=1).groupby(["Mfr"]).mean()



px.bar(manu,x=manu.index,y=["Calories","Protein","Fat","Sodium","Fiber","Carbo","Sugars","Potassium","Vitamins"],

       labels={"Mfr":"Manufacturer"})
px.imshow(df.corr(),color_continuous_scale="tealrose",color_continuous_midpoint=0)
px.scatter(df,x="Calories",y="Rating",trendline="ols",color_discrete_sequence=["gold"],

           labels={"Calories":"Calories per Serve","Rating":"Rating (%)"},

           hover_name="Name",hover_data={"Rating":":.2f"},marginal_x="histogram",marginal_y="box")
px.scatter(df,x="Sugars",y="Fat",trendline="ols",trendline_color_override="mediumseagreen",

           color_continuous_scale="picnic",opacity=0.3,size="Calories",size_max=35,color="Calories",

           labels={"Sugars":"Sugar (g) per Serve","Fat":"Fat (g) per Serve","Calories":"Calories per Serve"},

           hover_name="Name")
px.scatter(df,x=["Fiber","Protein","Potassium"],y="Rating",trendline="ols",

           labels={"Rating":"Rating (%)"},

           hover_name="Name",hover_data={"Rating":":.2f"})
df=pd.concat([df,pd.get_dummies(df["Mfr"])],axis=1)
df=pd.concat([df,pd.get_dummies(df["Type"])["C"]],axis=1)
df.rename(columns={"C":"cold"},inplace=True)
df.drop(["Mfr","Type"],axis=1,inplace=True)
df.drop(["Name","Shelf"],axis=1,inplace=True)
from sklearn.model_selection import train_test_split



x=df.drop(["Rating"],axis=1)

y=df["Rating"]



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=7)
print("x train shape",x_train.shape,"and y train shape",y_train.shape)
print("x test shape",x_test.shape,"and y test shape",y_test.shape)
from sklearn.preprocessing import StandardScaler



scaler=StandardScaler()

x_train=scaler.fit_transform(x_train)

x_test=scaler.transform(x_test)
from sklearn.neighbors import KNeighborsRegressor



model=KNeighborsRegressor(n_neighbors=5)

model.fit(x_train,y_train)

y_predict=model.predict(x_test)
model.score(x_test,y_test)
results=pd.DataFrame({"Actual":y_test,"Predicted":y_predict})



px.histogram(results,x=["Actual","Predicted"],nbins=15,opacity=0.3,marginal="rug",

             title="Distribution of the Acutal and Predicted Values")
neighbours=list(range(1,20))

score=[]



for i in range (1,20):

    model=KNeighborsRegressor(n_neighbors=i)

    model.fit(x_train,y_train)

    y_predict_i=model.predict(x_test)

    score.append(model.score(x_test,y_test))



neighbourlist=pd.DataFrame({"No of Neighbours":neighbours,"Score":score})
fig=px.line(neighbourlist,x="No of Neighbours",y="Score",range_y=[0.2,0.83],color_discrete_sequence=["slateblue"])

fig.add_annotation(x=5,y=0.73,

                   text="Optimal number of neighbours, which <br> was already implemented in the first model!",

                   standoff=0,arrowsize=1,arrowwidth=1.5,arrowhead=2)