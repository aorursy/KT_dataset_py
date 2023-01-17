# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go



import seaborn as sns



# word cloud library





# matplotlib

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
heart_df=pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
heart_df.head(10)
heart_df.info()
f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(heart_df.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.show()
g = sns.catplot(x = "exang", y = "age", data = heart_df, kind = "bar")

g.set_ylabels("Age")

g.set_xlabels("Exang(exercise induced angina)")

plt.show()
# Creating trace1

data = go.Scatter(

                    x = heart_df.age,

                    y = heart_df.thalach,

                    mode = "markers",

                    name = "Max Heart Rate",

                    marker = dict(color = 'rgba(16, 112, 2, 0.8)'),

                    text= heart_df.target)



layout = dict(title = 'Max Heart Rate vs Age',

              xaxis= dict(title= 'Age',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
g = sns.FacetGrid(heart_df, col = "fbs", row = "target", height = 2)

g.map(plt.hist, "age", bins = 25)

g.add_legend()

plt.show()
heart_df.sex.replace([1],["male"],inplace = True)

heart_df.sex.replace([0],["female"],inplace = True)

sns.swarmplot(x="sex", y="chol",hue="target", data=heart_df)

plt.show()
g=sns.factorplot(x="cp", y="exang",data=heart_df, kind = "bar", height = 6)

plt.show()
g = sns.jointplot(heart_df.age, heart_df.trestbps, kind="kde", height=7)

plt.savefig('graph.png')

plt.show()
g=sns.factorplot(x="exang", y="target",data=heart_df, kind = "bar", height = 6)

plt.show()
g = sns.FacetGrid(heart_df, col = "cp", row = "exang", size = 2)

g.map(sns.barplot,"target")

plt.show()
g = sns.jointplot(heart_df.oldpeak, heart_df.slope, kind="kde", height=7)

plt.savefig('graph.png')

plt.show()
g = sns.FacetGrid(heart_df, col = "slope", row = "target", size = 2)

g.map(sns.barplot,"oldpeak")

plt.show()
g = sns.catplot(x = "target", y = "oldpeak", data = heart_df, kind = "bar")

g.set_ylabels("oldpeak")

plt.show()
g = sns.FacetGrid(heart_df, col = "ca", row = "target", size = 2)

g.map(sns.barplot,"age")

plt.show()
g = sns.catplot(x = "target", y = "thalach", data = heart_df, kind = "bar")

g.set_ylabels("thalach")

plt.show()
for i in heart_df.target:

    if i == 1:

        heart1_index = list((heart_df[heart_df["target"]==i]).index)

    else:

        heart0_index = list((heart_df[heart_df["target"]==i]).index)

heart1_df = heart_df.iloc[heart1_index,:]

heart0_df = heart_df.iloc[heart0_index,:]



trace1 =go.Scatter(

                    x = heart0_df.thalach,

                    y = heart0_df.chol,

                    mode = "markers",

                    name="Target = 0",

                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'),

                    text= heart_df.age)

trace2 =go.Scatter(

                    x = heart1_df.thalach,

                    y = heart1_df.chol,

                    mode = "markers",

                    name="Target=1",

                    marker = dict(color = 'rgba(255, 1, 10, 255.8)'),

                    text= heart_df.age)



layout = dict(title = 'Chol vs Thalac'

              

             )

data=[trace1,trace2]

fig = dict(data = data, layout = layout)

iplot(fig)
x = heart_df["oldpeak"].values.reshape(-1,1)

y = heart_df["thalach"].values.reshape(-1,1)



from sklearn.linear_model import LinearRegression



lr = LinearRegression()



lr.fit(x,y)



plt.scatter(x,y,color="red")



y_head=lr.predict(x)

plt.plot(x,y_head,color="green")



plt.show()



from sklearn.metrics import r2_score



print("R score =",r2_score(y,y_head))
from sklearn.preprocessing import PolynomialFeatures

pf = PolynomialFeatures(degree=14)



x_poly = pf.fit_transform(x)



lr2 = LinearRegression()



lr2.fit(x_poly,y)



y_head2 = lr2.predict(x_poly)



plt.scatter(x,y,color="red")



plt.plot(x,y_head2,color="green")



plt.show()



from sklearn.metrics import r2_score



print("R score =",r2_score(y,y_head2))
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()



dt.fit(x,y)

x_new = np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head3_ = dt.predict(x_new)



plt.scatter(x,y,color="red")



plt.plot(x_new,y_head3_,color="green")



plt.show()



from sklearn.metrics import r2_score

y_head3 = dt.predict(x)

print("R score =",r2_score(y,y_head3))
from sklearn.ensemble import RandomForestRegressor



rfg = RandomForestRegressor(n_estimators=100,random_state=42)



rfg.fit(x,y)



x_new = np.arange(min(x),max(x),0.01).reshape(-1,1)

y_head4_ = rfg.predict(x_new)



plt.scatter(x,y,color="red")



plt.plot(x_new,y_head4_,color="green")



plt.show()



from sklearn.metrics import r2_score

y_head4 = rfg.predict(x)

print("R score =",r2_score(y,y_head4))
