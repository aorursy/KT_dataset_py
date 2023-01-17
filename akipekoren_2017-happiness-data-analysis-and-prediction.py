# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import scipy.stats as stats

import matplotlib.pyplot as plt

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/2017.csv")

data.head(10)
data.info()
data.isnull().values.any() # So we don't have any missing value in data

    
data.columns = data.columns.str.replace(".","_")

data
good_bad_list = ["good" if each >=5 else "bad" for each in data.Happiness_Score]

good_bad_list
data["good_or_bad"] = good_bad_list

y = data["good_or_bad"]

x= data.drop(["good_or_bad","Happiness_Rank","Country" ],axis=1)



ax = sns.countplot(data["good_or_bad"], label = "Count")

Good, Bad = data["good_or_bad"].value_counts()

print("Number of good country for living: ", Good)

print("Number of bad country for living: ", Bad)



fig = plt.figure(figsize = (10,6))

sns.barplot(x = "Freedom", y ="Happiness_Score",data = data)
fig = plt.figure(figsize = (10,6))

sns.barplot(x = "Economy__GDP_per_Capita_", y = "Happiness_Score", data = data)
fig =plt.figure(figsize=(10,6))

sns.barplot(x = "Health__Life_Expectancy_", y = "Happiness_Score",data = data)
fig = plt.figure(figsize=(10,6))

sns.barplot(x = "Generosity", y ="Happiness_Score", data = data)
fig = plt.figure(figsize=(10,6))

sns.barplot(x="Trust__Government_Corruption_", y = "Happiness_Score", data = data)
data_normalized = (x - x.min())/(x.max()-x.min()) # normalization



data = pd.concat([y, data_normalized], axis =1)



data = pd.melt(data, id_vars = "good_or_bad",

                      var_name = "features",

                      value_name = "values")



sns.violinplot(x="features", y ="values", hue = "good_or_bad",

               data= data, split ="True")



plt.xticks(rotation=90)

sns.set(style="darkgrid", color_codes = True)

ax = sns.jointplot(x.loc[:,"Economy__GDP_per_Capita_"],

                   x.loc[:,"Health__Life_Expectancy_"],

                  data = x, kind ="reg", height = 8, color="#ce1414")

ax.annotate(stats.pearsonr)

plt.show()
new_data = pd.read_csv("../input/2017.csv")

created_data = pd.concat([new_data["Country"], new_data["Happiness.Rank"],y, data_normalized, ], axis =1)

created_data.rename(columns = {"Happiness.Rank" : "Happiness_Rank"}, inplace = True)



import plotly.graph_objs as go

defined_data = created_data[created_data.Economy__GDP_per_Capita_ > 0.8].iloc[:4,:]

#defined_data

trace1 =go.Bar(

                x = defined_data.Country,

                y = defined_data.Trust__Government_Corruption_,

                name = "Trust to Government Corruption",

                marker = dict(color = 'rgba(255, 174, 255, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

                            )

trace2= go.Bar(

                x = defined_data.Country,

                y = defined_data.Dystopia_Residual,

                name = "Dystopia Residual",

     marker = dict(color = 'rgba(15, 44, 45, 0.5)',

                             line=dict(color='rgb(0,0,0)',width=1.5)),

    

)



data = [trace1, trace2]





layout = {

        "xaxis" : { "title" :" Top 4 Country"},

        "barmode" : "relative",

        "title" : "Dystopia Residual and Trust to Government Corruption of top 4 countries"

}



fig = go.Figure(data = data, layout = layout)

iplot(fig)

new_data = pd.read_csv("../input/2017.csv")

created_data = pd.concat([new_data["Country"], new_data["Happiness.Rank"],y, data_normalized, ], axis =1)

created_data.rename(columns = {"Happiness.Rank" : "Happiness_Rank"}, inplace = True)

created_data



trace1 = go.Scatter( x = created_data.Happiness_Rank , 

                    y = created_data.Economy__GDP_per_Capita_ ,

                    mode = "lines",

                    name = "GDP per Capita",

                    marker = dict(color = "rgba(150, 25, 125, 0.5)"),

                    text = created_data.Country

                    

)



trace2 = go.Scatter ( x = created_data.Happiness_Rank,

                    y = created_data.Family,

                    mode = "lines",

                    name = "Family",

                    marker = dict(color = "rgba(5,150, 125, 0.8)"),

                    text = created_data.Country

                    )





trace3 = go.Scatter ( x = created_data.Happiness_Rank,

                    y = created_data.Health__Life_Expectancy_,

                    mode = "lines",

                    name = "Life Expectancy",

                    marker = dict(color = "rgba(0, 0, 225, 0.4)"),

                    text = created_data.Country

                    )



data = [trace1, trace2, trace3]



layout = dict(title = "Effect of Family, GDP and Life Expectancy of Happiness",

             xaxis =dict( title ="World Rank",ticklen = 5, zeroline = False)

             )



fig = dict(data = data, layout = layout)



iplot(fig)
f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(new_data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
x_norm = (x-x.min())/(x.max()-x.min()) # Normalization

x_norm.drop(["Happiness_Score"], axis =1, inplace = True)





from sklearn.model_selection import train_test_split



x_train,x_test, y_train, y_test = train_test_split(x_norm, y,

                                                   test_size =0.3, random_state = 42)

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 3) # n_neighbor is knn value

knn.fit(x_train, y_train)

prediction = knn.score(x_test, y_test)



print("knn value : {},  score : {}".format(3,prediction))
y_pred = knn.predict(x_test)

y_true = y_pred



from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)

cm
x_norm = (x-x.min())/(x.max()-x.min()) # Normalization



from sklearn.model_selection import train_test_split



x_train,x_test, y_train, y_test = train_test_split(x_norm, y,

                                                   test_size =0.3, random_state = 42)



from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=100, random_state =42)



rf.fit(x_train,y_train)

print("Score : {}".format(rf.score(x_test,y_test)))