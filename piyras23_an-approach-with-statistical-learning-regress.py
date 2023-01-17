from IPython.core.display import display, HTML, Javascript

import IPython.display
# General Essential Libraries:

import numpy as np 

import pandas as pd 



import seaborn as sns 

sns.set(style = "whitegrid")

import matplotlib.pyplot as plt 



import warnings

warnings.filterwarnings("ignore")



from IPython.core.display import display, HTML, Javascript

import IPython.display
# Libraries for interactive visualisation: 

import plotly.figure_factory as ff 

import  plotly.offline as py

import plotly.graph_objs as go 



from plotly.offline import download_plotlyjs,init_notebook_mode, iplot, plot

from plotly import tools 

py.init_notebook_mode(connected = True)



import cufflinks as cf 

cf.go_offline()
# Libraries for Machine Learning Algroithyms:

from sklearn import preprocessing

from sklearn.preprocessing import OneHotEncoder 



from sklearn import linear_model

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
# Header of Training set 

df_train.head(3)
# Header of Test Set 

df_test.head(3)
# Information of the Training set: 

#df_train.info()
# Information of the Test set: 

#df_test.info()
# General Description of Numberical Columns of the Entire Set (Training and Testing)

df = pd.concat([df_train,df_test], axis = 0)

df.drop("SalePrice", axis =1, inplace=True )

df.describe()
round(df_train["SalePrice"].describe(),2)
df_types = df.dtypes.value_counts()

print(df_types)



plt.figure(figsize = (14,4))

sns.barplot(x = df_types.index, y = df_types.values)

plt.title("Data Type Distribution")
num_col = df.select_dtypes(include=("float64", "int64"))

cat_col = df.select_dtypes(include=("object"))
# Correalation plot in order to identify the relationship between the Numberical Features:

plt.figure(figsize=(20,10))

sns.heatmap(df_train.corr(), linewidths=.1, annot=True, cmap='magma')

df_train.corr()["SalePrice"].sort_values(ascending = False).head(5)
fig, (ax1, ax2) =plt.subplots(nrows=2, ncols=1, figsize = (15,10))



sns.heatmap(cat_col.isnull(), cbar = False, annot = False, cmap ="cividis", yticklabels=False, ax=ax1)

plt.title("Missing Values in Categorical Columns")

sns.heatmap(num_col.isnull(), cbar = False, annot = False, cmap ="cividis", yticklabels=False)

plt.title("Missing Values in Numberical Columns")

plt.tight_layout()

mat = df_train[["SalePrice", "LotFrontage", "TotalBsmtSF","GrLivArea","OverallQual" ]]

mat["index"] = np.arange(len(mat))



fig = ff.create_scatterplotmatrix(mat, diag="box", index="index", colormap_type="seq", colormap="Jet", 

                                 height = 900, width = 1100)

py.iplot(fig)
hist_data = [df_train["SalePrice"]]

label = ["Sales Price"]

color = ["navy"]



fig = ff.create_distplot(hist_data, label, colors = color, show_hist=False)

fig["layout"].update(title ="Sale Price Distribution") 

py.iplot(fig)
for i in cat_col.columns:

    print(cat_col[i].value_counts(), "/n")
def uni(col):

    out = []

    for i in col:

        if i not in out:

            out.append(i)

    return(out)

""""""

colors1 = ["#a9fcca","#d6a5ff", "#639af2", "#fca6da", "#f4d39c", "orange", "#7af9ad","green", "maroon", "#3498db", "#95a5a6", "#e74c3c", "#34495e","#df6a84","#ad2543","#223f08", "#DF3A01", "#045FB4","#088A4B","#FE2E64" ]

''''''

def bar_pie (col, colors = colors1,

            main_title = "Main Title", x_label = "X Label", y = "Y label", do_x = [.6,.9], do_y = [.9,.2]):

    

    col_count = df[col].value_counts()

    

    trace = go.Bar(x = col_count.index, y = col_count.values, marker=dict(color = colors))

    

    trace1 = go.Pie(labels= col_count.index, values=col_count.values, hole= 0.6, textposition="outside", marker=dict(colors = colors),

               domain = dict(x = do_x, y = do_y), name = "title", )

    

    data = [trace, trace1]

    layout = go.Layout(title= main_title)

    fig = go.Figure(data =data, layout = layout)

    iplot(fig)



""""""

def price(col):

    if col in range(0, 150000):

        return("Low")

    elif col in range(15000, 300000):

        return("Medium")

    else:

        return("High")

df_train["Price"] =df_train["SalePrice"].apply(lambda x: price(x))

df_train.head(4)



''''''

neig_hood= uni(df["Neighborhood"])

sale_cond = uni(df["SaleCondition"])

qual = uni(df["OverallQual"])

me = df_train.groupby("Neighborhood").agg({"SalePrice":np.mean}).reset_index()
cat_col.columns.values
num_col.columns.values
bar_pie(col="HouseStyle", main_title="House Style Frequency")
mon_d = {1:"Jan", 2:"Feb",3:"Mar",4:"Aprl",5:"May",6:"Jun",7:"July",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dex"}

df["Month"] = df["MoSold"].map(mon_d)



bar_pie(col="Month", main_title="Number of Houses sold very Month", do_x= [.9,.9] ,do_y= [.9,.4])
data = []

for i in neig_hood:

    data.append(go.Box(y = df_train[df_train["Neighborhood"]==i]["SalePrice"], name = i))



layout = go.Layout(title = 'Sales Price based on Neighborhood', 

                   xaxis = dict(title = 'Neighborhood'), 

                   yaxis = dict(title = 'Sale Price'))

fig = dict(data = data, layout = layout)

py.iplot(fig)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(18,7))

sns.violinplot( x= df_train["Exterior1st"], y=df_train["SalePrice"], ax=ax[0])

plt.title("Sales Price VS Exterior Distibution")

plt.xticks(rotation =90)



sns.boxplot( x= df_train["Exterior2nd"], y=df_train["SalePrice"], ax=ax[1])

plt.xticks(rotation =90)

plt.title("Sales Price VS Exterior Distibution")

plt.tight_layout()
data = []

for i in qual:

    data.append(go.Box(y = df_train[df_train["OverallQual"]==i]["SalePrice"], name = i,  boxpoints='all',

            jitter=0.5,

            whiskerwidth=0.2,

            marker=dict(

                size=2,

            ),

            line=dict(width=1),

        ))

layout = go.Layout(title = 'Sales Price based on Overall Quality', xaxis=dict(title ="Ouality grade"), yaxis=dict(title ="Sales Price"))



fig = dict(data = data, layout = layout)

py.iplot(fig)
yr_built = uni(df["YearBuilt"])

data = []

for i in yr_built:

    data.append(go.Box(y = df_train[df_train["YearBuilt"]==i]["SalePrice"], name = i,  boxpoints='all',

            jitter=0.5,

            whiskerwidth=0.2,

            marker=dict(

                size=2,

            ),

            line=dict(width=1),

        ))

layout = go.Layout(title = 'Sales Price based on Year Built', xaxis=dict(title ="Years"), yaxis=dict(title ="Sales Price"))



fig = dict(data = data, layout = layout)

py.iplot(fig)
yr_rem = uni(df["YearRemodAdd"])

data = []

for i in yr_rem:

    data.append(go.Violin(y = df_train[df_train["YearRemodAdd"]==i]["SalePrice"], name = i))

layout = go.Layout(title = 'Sales Price of Renovated Houses', xaxis=dict(title ="Years"), yaxis=dict(title ="Sales Price"))



fig = dict(data = data, layout = layout)

py.iplot(fig)
num_col.columns
df_train.corr()["SalePrice"].sort_values()
price_uni = uni(df_train["Price"])

qual = uni(df["OverallQual"])

buil_type = uni(df_train["BldgType"])
data1 = []

for item, colors in zip(buil_type, ["lime","deepskyblue","#d6a5ff", "#639af2", "#fca6da", "#f4d39c", "orange", "#7af9ad"]):

    

    tem_df = df_train[df_train["BldgType"]== item]



    data1.append(go.Scatter(x = tem_df["LotArea"], y = tem_df["SalePrice"], name=item, mode= "markers",opacity = 0.75,

                             marker = dict(line = dict(color = 'black', width = 0.5))))

layout = go.Layout(title = 'Sales Price vs Lot Area ', xaxis = dict(title = 'Lot Area'), 

                   yaxis = dict(title = 'Sales Price'))



fig = go.Figure(data = data1, layout = layout)

py.iplot(fig)

df_train.corr()["SalePrice"].sort_values(ascending=False).head(10)
fig, ((ax1, ax2), (ax3, ax4))  = plt.subplots(nrows=2,ncols=2, figsize=(20,8))

sns.scatterplot(x= df_train["LotArea"], y = df_train["SalePrice"], ax= ax1, hue = df_train["Price"])

sns.scatterplot(x= df_train["GrLivArea"], y = df_train["SalePrice"], ax=ax2,hue = df_train["Price"])

sns.scatterplot(x= df_train["GarageArea"], y = df_train["SalePrice"], ax=ax3,hue = df_train["Price"])

sns.scatterplot(x= df_train["TotalBsmtSF"], y = df_train["SalePrice"], ax=ax4, hue = df_train["Price"])

plt.tight_layout()
df_train["Utilities"].value_counts()
plt.figure(figsize = (18,9))

sns.swarmplot(x= df_train["GarageCars"], y = df_train["SalePrice"])
# Total Houses sold in dollars:

df_train.groupby("YrSold")["SalePrice"].sum()
# Boxplot of Sales Price Vs Years

#sns.boxplot(data = df_train, x = "YrSold", y ="SalePrice")



trace = go.Box( y = df_train[df_train["YrSold"]==2006]["SalePrice"], 

              name = "2006")

trace1 = go.Box( y = df_train[df_train["YrSold"]==2007]["SalePrice"],

               name = "2007")

trace2 = go.Box( y = df_train[df_train["YrSold"]==2008]["SalePrice"],

               name = "2008")

trace3 = go.Box( y = df_train[df_train["YrSold"]==2009]["SalePrice"],

               name = "2009")

trace4 = go.Box( y = df_train[df_train["YrSold"]==2010]["SalePrice"],

               name = "2010")



layout = go.Layout(title = "Yearly Sale Prices", 

                   yaxis=dict(title = "Sales Price"), 

                  xaxis=dict(title = "Years"))



data = [trace, trace1, trace2, trace3, trace4]



fig = go.Figure(data= data, layout=layout)

py.iplot(fig)
sns.boxplot(data = df_train, x = "MoSold", y ="SalePrice")
df_train.head(1)
sns.boxplot(data = df_train, x = "HouseStyle", y ="SalePrice")
plt.figure(figsize=(20,15))

sns.boxplot(data = df_train, x = "YearBuilt", y ="SalePrice")

plt.xticks(rotation = "90")
df["YearBuilt"].value_counts()
tot_cel = np.product(df.shape)

tot_cel

miss_cel = df.isnull().sum().sum()

total_missing = (miss_cel/tot_cel)*100

print(f"Total percent of missing values in the data is: {round(total_missing)}%")
print(df.isnull().any().value_counts(),"\nTherefore, the total columns having missing values are 34")
# Data Frame of all the features having missing values with percentage:

total = df_train.isnull().sum().sort_values(ascending= False)



perc = df_train.isnull().sum() / df_train.isnull().count()*100

perc1 = (round(perc, 2).sort_values(ascending = False))



missing_data = pd.concat([total, perc1, df_train.isnull().sum(), df_test.isnull().sum()], axis=1,  keys=["Total Missing Values", "Percantage %", "Missing values in Train", "Missing values in Test"])

missing_data.sort_values(by="Total Missing Values", ascending=False).head(20)
plt.figure(figsize=(20,5))

sns.heatmap(df.isnull(),cbar= False, yticklabels=False, cmap = "cividis")



# Ploting the top features based on their missing values

trace1 = go.Bar(x = missing_data.index, y = missing_data["Total Missing Values"].values,

               marker = dict(color = df["YearRemodAdd"],

                            colorscale = "Picnic"))



layout = go.Layout(title="Total Missing Values Plot", 

                   yaxis= dict(title ="Percatnage (%)"))



data = [trace1]



fig = go.Figure(data= data , layout= layout)

py.iplot(fig)
