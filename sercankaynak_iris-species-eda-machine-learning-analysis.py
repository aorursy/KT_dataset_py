# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# matplotlib

import matplotlib.pyplot as plt



# seaborn

import seaborn as sns



#plotly

import plotly.io as pio

from plotly.offline import init_notebook_mode,iplot

init_notebook_mode(connected = True)

import plotly.graph_objs as go



from collections import Counter



import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# reading data

data = pd.read_csv("/kaggle/input/iris/Iris.csv")

data
data.columns
data.head()
data["Species"].unique()
data["Species"].value_counts()
data.describe()
data.info()
# get feature

cat_var = data["Species"]

cat_var



# count number of categoricaal variable (value/sample)

var_value = cat_var.value_counts()

var_value
# visualize



plt.figure(figsize = (8,8))

plt.bar(var_value.index,var_value)

plt.xticks(var_value.index,var_value.index.values)

plt.ylabel("Frequency")

plt.title("Species' Frequency")

plt.show()
# Coding a simple function to make a plot for each numerical variable

def plot_hist(variable):

    plt.figure(figsize = (6,6))

    plt.hist(data[variable],bins = 40)

    plt.xlabel(variable)

    plt.ylabel("Frequency")

    plt.title("{} distribution".format(variable))

    plt.show()
# visualizing for each numerical variable

num_var = ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]

for i in num_var:

    plot_hist(i)
# Sepal Length - iris-setosa / iris-versicolor / iris-virginica



data[["SepalLengthCm","Species"]].groupby(["Species"],as_index = False).mean().sort_values(by = "Species",ascending = False)
# Sepal Width - iris-setosa / iris-versicolor / iris-virginica



data[["SepalWidthCm","Species"]].groupby(["Species"],as_index = False).mean().sort_values(by = "Species",ascending = False)
# Petal Length - iris-setosa / iris-versicolor / iris-virginica



data[["PetalLengthCm","Species"]].groupby(["Species"],as_index = False).mean().sort_values(by = "Species",ascending = False)
# Petal Width - iris-setosa / iris-versicolor / iris-virginica



data[["PetalWidthCm","Species"]].groupby(["Species"],as_index = False).mean().sort_values(by = "Species",ascending = False)
data.describe()
# coding a simple function for detecting outliers



def detect_outliers(df,features):

    outlier_indices = []

    

    for c in features:

        # 1st quartile

        Q1 = np.percentile(df[c],25)

        # 3rd quartile

        Q3 = np.percentile(df[c],75)

        # IQR

        IQR = Q3 - Q1

        # Outlier step

        outlier_step = IQR * 1.5

        # detect outlier and their indexes

        outlier_list_col = df[(df[c] < Q1 - outlier_step) | (df[c] > Q3 + outlier_step)].index

        # store indexes

        outlier_indices.extend(outlier_list_col)

        

    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list(i for i,v in  outlier_indices.items() if v > 2)

    

    return multiple_outliers
data.loc[detect_outliers(data,["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"])]
data.columns[data.isnull().any()]
column_list = ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]

f,ax = plt.subplots(figsize = (10,10))

sns.heatmap(data[column_list].corr(),annot = True,fmt = ".2f")

plt.show()
# drop Id columns for better dataframe

data.drop(["Id"],axis = 1,inplace = True)
# prepare data frames

data_setosa_s = data[data.Species == "Iris-setosa"].iloc[:,:]

data_versicolor_s = data[data.Species == "Iris-versicolor"].iloc[:,:]

data_virginica_s = data[data.Species == "Iris-virginica"].iloc[:,:]



# Creating trace 1

trace1 = go.Scatter(

                    x = data_setosa_s.SepalLengthCm,

                    y = data_setosa_s.SepalWidthCm,

                    mode = "markers",

                    name = "iris-setosa",

                    marker_color = "red",

                    )

# Creating trace 2

trace2 = go.Scatter(

                    x = data_versicolor_s.SepalLengthCm,

                    y = data_versicolor_s.SepalWidthCm,

                    mode = "markers",

                    name = "iris-versicolor",

                    marker_color = "blue",

                    )

# Creating trace 3

trace3 = go.Scatter(

                    x = data_virginica_s.SepalLengthCm,

                    y = data_virginica_s.SepalWidthCm,

                    mode = "markers",

                    name = "iris-virginica",

                    marker_color = "yellow",

                    )

data1 = [trace1,trace2,trace3]

layout = dict(

            title = "Sepal Length vs Sepal Width of All Three Iris Species",

            xaxis = dict(title = "Sepal Length",ticklen = 5,zeroline = False),

            yaxis = dict(title = "Sepal Width",ticklen = 5,zeroline = False)

            )

fig = dict(data = data1,layout = layout)

iplot(fig)
# prepare data frames

data_setosa_p = data[data.Species == "Iris-setosa"].iloc[:,:]

data_versicolor_p = data[data.Species == "Iris-versicolor"].iloc[:,:]

data_virginica_p = data[data.Species == "Iris-virginica"].iloc[:,:]



# Creating trace 1

trace1 = go.Scatter(

                    x = data_setosa_p.PetalLengthCm,

                    y = data_setosa_p.PetalWidthCm,

                    mode = "markers",

                    name = "iris-setosa",

                    marker_color = "purple",

                    )

# Creating trace 2

trace2 = go.Scatter(

                    x = data_versicolor_p.PetalLengthCm,

                    y = data_versicolor_p.PetalWidthCm,

                    mode = "markers",

                    name = "iris-versicolor",

                    marker_color = "cyan",

                    )

# Creating trace 3

trace3 = go.Scatter(

                    x = data_virginica_p.PetalLengthCm,

                    y = data_virginica_p.PetalWidthCm,

                    mode = "markers",

                    name = "iris-virginica",

                    marker_color = "orange",

                    )

data2 = [trace1,trace2,trace3]

layout = dict(

            title = "Petal Length vs Petal Width of All Three Iris Species",

            xaxis = dict(title = "Petal Length",ticklen = 5,zeroline = False),

            yaxis = dict(title = "Petal Width",ticklen = 5,zeroline = False)

            )

fig = dict(data = data2,layout = layout)

iplot(fig)
# Show sepal length distribution with violin plot

pal = sns.color_palette("RdBu", n_colors=3)

f,ax = plt.subplots(figsize = (10,10))

sns.violinplot(x = "Species",y = "SepalLengthCm",data = data,palette = pal,scale = "count",inner = "point")

plt.show()
# Show sepal width distribution with violin plot

pal = sns.color_palette("Set2", n_colors=3)

f,ax = plt.subplots(figsize = (10,10))

sns.violinplot(x = "Species",y = "SepalWidthCm",data = data,palette = pal,scale = "count",inner = "point")

plt.show()
# Show petal length distribution with violin plot

pal = sns.color_palette("rainbow", n_colors=3)

f,ax = plt.subplots(figsize = (10,10))

sns.violinplot(x = "Species",y = "PetalLengthCm",data = data,palette = pal,scale = "count",inner = "point")

plt.show()
# Show petal width distribution with violin plot

pal = sns.color_palette("prism", n_colors=3)

f,ax = plt.subplots(figsize = (10,10))

sns.violinplot(x = "Species",y = "PetalWidthCm",data = data,palette = pal,scale = "count",inner = "point")

plt.show()
column_list = ["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]

f,ax = plt.subplots(figsize = (10,10))

sns.heatmap(data[column_list].corr(),annot = True,fmt = ".2f")

plt.show()
# plotting data

plt.scatter(data.PetalLengthCm,data.PetalWidthCm)

plt.xlabel("Petal Length Cm")

plt.ylabel("Petal Width Cm")

plt.show()
# import sklearn and prepare data

from sklearn.linear_model import LinearRegression

linear_reg = LinearRegression()

x = data.PetalLengthCm.values.reshape(-1,1)

y = data.PetalWidthCm.values.reshape(-1,1)



# line fit and prediciton

linear_reg.fit(x,y)

y_head = linear_reg.predict(x)



# plotting line fit

plt.scatter(data.PetalLengthCm,data.PetalWidthCm)

plt.xlabel("Petal Length Cm")

plt.ylabel("Petal Width Cm")

plt.plot(x,y_head,color="red")

plt.show()
# r^2 score

from sklearn.metrics import r2_score

print("r_square score : ",r2_score(y,y_head))
from sklearn.preprocessing import PolynomialFeatures

polynomial_reg = PolynomialFeatures(degree = 2)

x = data.PetalLengthCm.values.reshape(-1,1)

y = data.PetalWidthCm.values.reshape(-1,1)



x_polynomial = polynomial_reg.fit_transform(x)

linear_reg2 = LinearRegression()

linear_reg2.fit(x_polynomial,y)



y_head2 = linear_reg2.predict(x_polynomial)



# visualize

plt.scatter(x,y)

plt.xlabel("Petal Length Cm")

plt.ylabel("Petal Width Cm")

plt.plot(x,y_head2,color = "orange")

plt.show()
# r^2 score

from sklearn.metrics import r2_score

print("r_square score : ",r2_score(y,y_head2))
# decision tree regression

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor() # random state = 0

x = data.PetalLengthCm.values.reshape(-1,1)

y = data.PetalWidthCm.values.reshape(-1,1)



# fit and predict

tree_reg.fit(x,y)

y_head3 = tree_reg.predict(x)



#visualize

f,ax = plt.subplots(figsize = (10,10))

plt.scatter(x,y_head3,color ="red")

plt.xlabel("Petal Length Cm")

plt.ylabel("Petal Width Cm")

plt.plot(x,y_head3,color = "blue")

plt.show()
# r^2 score

from sklearn.metrics import r2_score

print("r_square score : ",r2_score(y,y_head3))
# random forest regression

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 100,random_state = 42)

x = data.PetalLengthCm.values.reshape(-1,1)

y = data.PetalWidthCm.values.reshape(-1,1)

rf.fit(x,y)

y_head4 = rf.predict(x)



# visualize

f,ax = plt.subplots(figsize = (10,10))

plt.scatter(x,y,color = "purple")

plt.plot(x,y_head4,color = "green")

plt.xlabel("Petal Length Cm")

plt.ylabel("Petal Width Cm")

plt.show()
# r^2 score

from sklearn.metrics import r2_score

print("r_square score : ",r2_score(y,y_head3))