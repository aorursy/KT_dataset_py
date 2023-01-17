# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load





""" 

For Data Processing

"""

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



"""

For Visualizations

"""

import seaborn as sns

import matplotlib.pyplot as plt



import plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px





"""

For Machine Learning

"""

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor





import warnings as wrn

wrn.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/iris/Iris.csv")
df.head() # This method shows us the first 5 columns of the dataset

df.tail() #This method shows us the last 5 columns of the dataset
df.info() #This method shows us the general information about the dataset.
describe = df.describe()

describe.drop("Id",axis=1,inplace=True)

describe

data2 = df.copy()
def outlier_dropper(dataFrame,describeFrame):

    drp = 0

    for clm in describeFrame:

        Q3 = describeFrame[clm]["75%"]

        Q1 = describeFrame[clm]["25%"]

        IQR = (Q3-Q1)*1.5

        lower = Q1-IQR

        upper = Q3 + IQR

        ind = dataFrame[(dataFrame[clm]<lower) | (dataFrame[clm]>upper)].index.values

        dataFrame.drop(ind,inplace=True)

        drp += len(ind)

    return dataFrame,drp

        

df,dropcount = outlier_dropper(df,describe)



print("There is/are",dropcount,"outlier values")
corr = df.corr()

corr.drop("Id",axis=1,inplace=True)

corr.drop("Id",axis=0,inplace=True)



fig,ax = plt.subplots(figsize=(10,10))



sns.heatmap(corr,annot=True,fmt="0.2f",linewidth=1,linecolor="Black",ax=ax)

plt.show()
figure = plt.figure(figsize=(22,8))



# Scatter plot

ax = figure.add_subplot(2,2,1)

sns.scatterplot("PetalLengthCm","SepalLengthCm",data=df,ax=ax,s=100,color="green")



# Kde Plot

ax = figure.add_subplot(2,2,2)

sns.kdeplot(df["PetalLengthCm"],df["SepalLengthCm"],ax=ax,color="Green",cmap="BuPu")



#Distplot for PetalLengthCm Feature

ax = figure.add_subplot(2,2,3)

sns.distplot(df["PetalLengthCm"],color="red")



#Distplot for SepalLengthCm Feature

ax = figure.add_subplot(2,2,4)

sns.distplot(df["SepalLengthCm"],color="cyan")



plt.show()
figure = plt.figure(figsize=(22,8))



# Scatter plot

ax = figure.add_subplot(2,2,1)

sns.scatterplot("PetalWidthCm","SepalLengthCm",data=df,ax=ax,s=100,color="orange")



#KDE Plot

ax = figure.add_subplot(2,2,2)

sns.kdeplot(df["PetalWidthCm"],df["SepalLengthCm"],ax=ax,color="Green",cmap="Accent")



# Distplot for PetalWidthCm Feature

ax = figure.add_subplot(2,2,3)

sns.distplot(df["PetalWidthCm"],color="purple")



# Distplot for SepalLengthCm Feature

ax = figure.add_subplot(2,2,4)

sns.distplot(df["SepalLengthCm"],color="brown")



plt.show()
figure = plt.figure(figsize=(22,8))

# Scatter plot

ax = figure.add_subplot(2,2,1)

sns.scatterplot("PetalWidthCm","PetalLengthCm",data=df,ax=ax,s=100,color="#74D1ED")



#KDE Plot

ax = figure.add_subplot(2,2,2)

sns.kdeplot(df["PetalWidthCm"],df["PetalLengthCm"],ax=ax,color="Green",cmap="tab20_r")



# Distplot for PetalWidthCm Feature

ax = figure.add_subplot(2,2,3)

sns.distplot(df["PetalWidthCm"],color="#F8079B",ax=ax)



# Distplot for PetalLengthCm Feature

ax = figure.add_subplot(2,2,4)

sns.distplot(df["PetalLengthCm"],color="#2507F8",ax=ax)



plt.show()
# First I am going to create a linear regression object from sklearn



petal_ln_LR = LinearRegression()



# Then I am going to fit a line 



x = df["PetalWidthCm"].values.reshape(-1,1) 

petal_length_y = df["PetalLengthCm"].values.reshape(-1,1)



petal_ln_LR.fit(x,petal_length_y)



"""

And now we are ready to some predictions and visualizations. Firstly, I want to visualize my model.

"""





petal_length_y_head = petal_ln_LR.predict(x)





fig,ax = plt.subplots(figsize=(11,7))

sns.scatterplot("PetalWidthCm","PetalLengthCm",data=df,s=100)

plt.plot(x,list(petal_length_y_head),color="Red",linewidth=2.5)

plt.xlabel("PetalWidthCm",fontsize=15)

plt.ylabel("PetalLengthCm",fontsize=15)

plt.show()
from sklearn.metrics import r2_score



print(r2_score(petal_length_y,petal_length_y_head))

# First I create an linear regression object

sepalLength_MLR = LinearRegression()



#Then I train it

x = df.loc[:,["PetalWidthCm","PetalLengthCm"]].values

sepal_length_y = df["SepalLengthCm"].values.reshape(-1,1)

sepalLength_MLR.fit(x,sepal_length_y)



# And now I am ready to do some predictions

sepal_length_y_head = sepalLength_MLR.predict(x)

y_head2 = sepal_length_y_head.reshape(-1) # Scalar 





# Visualizations using Plotly

trace = go.Scatter3d(x=df["PetalWidthCm"]

                    ,y=df["PetalLengthCm"]

                    ,z= y_head2

                    ,name="SepalLengthCm"

                    ,marker=dict(color="rgba(255,0,0,0.8)")

                    ,mode="markers"

                    ,text=y_head2)



layout = go.Layout(title="SepalLengthCm Prediction",xaxis=dict(title="PetalWidthCm")

                  ,yaxis=dict(title="PetalLengthCm"))





figure = go.Figure(data=trace)



iplot(figure)

print(r2_score(sepal_length_y,sepal_length_y_head))

x = df["PetalLengthCm"].values.reshape(-1,1)

petal_width_y = df["PetalWidthCm"].values.reshape(-1,1)



petalWidthRegression = LinearRegression()

petalWidthRegression.fit(x,petal_width_y)



petal_width_y_head = petalWidthRegression.predict(x)





fig,ax = plt.subplots(figsize=(11,7))

plt.scatter(x,petal_width_y,color="#D91462")

plt.plot(x,petal_width_y_head,color="#1C81FE")

plt.show()
print(r2_score(petal_width_y,petal_width_y_head))
from sklearn.model_selection import train_test_split # In order to split, I am going to use this function



x = df.drop("Species",axis=1)

y = df.Species



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(x_train,y_train)

print(knn.score(x_test,y_test))
