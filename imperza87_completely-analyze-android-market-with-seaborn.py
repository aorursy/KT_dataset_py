import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import seaborn as sns
import os 
import plotly
import plotly.plotly as py
import math
# connected=True means it will download the latest version of plotly javascript library.
plotly.offline.init_notebook_mode(connected=True)
import plotly.graph_objs as go

import plotly.figure_factory as ff

df = pd.read_csv("../input/googleplaystore.csv")
df.head()
df.info()
sum(df.duplicated("App"))
df.drop_duplicates(subset='App', inplace=True)
df.isna().sum()
df.dropna(subset = ["Type" , "Content Rating" , "Current Ver" , "Android Ver"]  , inplace = True)
df.isna().sum()
df["Rating"] = df["Rating"].apply(lambda x: np.nan if math.isnan(x) else x)
for i in np.unique(df["Category"]):
    for l in np.unique(df["Type"]):
        df.loc[(df.Category == i) & (df.Type == l) & (df.Rating.isnull()), "Rating"] = np.random.normal(
            np.mean(df.loc[(df.loc[:,"Category"] == i) & (df.loc[:,"Type"] == l)].loc[:,"Rating"]) , np.std(df.loc[(df.loc[:,"Category"] == i) & (df.loc[:,"Type"] == l)].loc[:,"Rating"]) , 
        size = len(df.loc[(df.Category == i) & (df.Type == l) & (df.Rating.isnull()), "Rating"]))                                                                                           
df["Rating"] = df["Rating"].apply(lambda x: 5 if x>5 else x)
df["Rating"] = df["Rating"].apply(lambda x: 1 if x<1 else x)
df["Rating"]
df.dropna(inplace = True)
df.loc[:,'Installs'] = df.loc[:,'Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)
df.loc[:,"Installs"]=df.loc[:,"Installs"].apply(lambda x: x.replace(",","") if "," in str(x) else x)
df.loc[:,"Installs"] = df.loc[:,"Installs"].apply(lambda x: int(x))
df.loc[:,'Price'] = df.loc[:,'Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))
df.loc[:,'Price'] = df.loc[:,'Price'].apply(lambda x: float(x))

df.loc[:,"Reviews"] = df.loc[:,"Reviews"].apply( lambda x : float(x))
df.loc[:,"Size"]
df.loc[:,"Size"] = df.loc[:,"Size"].apply(lambda x : x.replace("M","") if "M" in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)
df.loc[:,"Size"] = df.loc[:,'Size'].apply( lambda x : float(x) if x != "Varies with device" else x)
df.columns
#heat map
bg_color = 'white'
sns.set(rc={"font.style":"normal",
            "axes.facecolor":bg_color,
            "figure.facecolor":bg_color,
            "text.color":"black",
            "xtick.color":"black",
            "ytick.color":"black",
            "axes.labelcolor":"black",
            "axes.grid":False,
            'axes.labelsize':15,
            'figure.figsize':(20.0, 10.0),
            'xtick.labelsize':15,
            'font.size':15,
            'ytick.labelsize':15})
df_cor = df.loc[:,["Rating","Reviews","Installs","Price"]]
df_cor = (df_cor - np.mean(df_cor))/np.std(df_cor)
cor = df_cor.corr()
mask = np.zeros_like(cor, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(cor , cmap = cmap , mask = mask , annot = True)
earning={}
for i in np.unique(df["Category"]):
        earning[i] = np.mean(df.loc[(df.Category == i) & (df.Type == "Paid") , "Price"])*np.median(df.loc[(df.Category == i) & (df.Type == "Paid")  , "Installs"]) 

#how much paid app had earned(median)
a = sns.barplot(y = list(earning.keys()) , x = list(earning.values()) ,
                saturation=1 , 
                palette = "GnBu_d", 
               )
#mean of price of each category 
a = sns.barplot(y = df.loc[(df.Type == "Paid") , "Category"]  ,
                x = df.loc[(df.Type == "Paid") , "Price"] , ci = None ,
                saturation=1 , 
                palette = "GnBu_d", 
               )

#rating of each category
a = sns.barplot(y = df["Category"] , x = df["Rating"] ,ci=None , 
                saturation=1 , 
                data = df , 
                palette = "RdBu",
                hue = df["Type"] ,
                order = df.groupby("Category").mean().sort_values("Rating" ,
                                                                       ascending = False).index[:16])
#rating of each category
a = sns.barplot(y = df["Category"] , x = df["Rating"] ,ci=None , 
                saturation=1 , 
                data = df , 
                palette = "RdBu",
                hue = df["Type"] ,
                order = df.groupby("Category").mean().sort_values("Rating" ,
                                                                       ascending = False).index[16:])
#number of app of each category

p = sns.countplot(data=df,
                  y = 'Category',
                  saturation=1 , 
                  palette = "RdBu",
                  hue = df["Type"] ,
                  order = df['Category'].value_counts().index[:16])



#number of app of each category

p = sns.countplot(data=df,
                  y = 'Category',
                  saturation=1 , 
                  palette = "RdBu",
                  hue = df["Type"] ,
                  order = df['Category'].value_counts().index[16:])


#groups = df_free.groupby('Category').filter(lambda x: len(x) > 258).reset_index()
#array = groups['Rating'].hist(by=groups['Category'], sharex=True, figsize=(20,20))
#groups = df_paid.groupby('Category').filter(lambda x: len(x) > 25).reset_index()
#array = groups['Rating'].hist(by=groups['Category'], sharex=True, figsize=(20,20))
#installs of each category
a = sns.barplot(y = df["Category"] , x = df["Installs"] , ci=None , 
                hue = df["Type"] ,
                saturation=1 , 
                palette = "RdBu",
                data = df , 
                order = df.groupby("Category").mean().sort_values("Installs" ,
                                                                ascending = False).index[:16])
#installs of each category
a = sns.barplot(y = df["Category"] , x = df["Installs"] , ci=None , 
                hue = df["Type"] ,
                saturation=1 , 
                palette = "RdBu",
                data = df , 
                order = df.groupby("Category").mean().sort_values("Installs" ,
                                                                ascending = False).index[16:])
earning={}
for i in np.unique(df["Content Rating"]):
        earning[i] = np.mean(df.loc[(df["Content Rating"] == i) & (df.Type == "Paid") , "Price"])*np.mean(df.loc[(df["Content Rating"] == i) & (df.Type == "Paid")  , "Installs"]) 
earning
#how much paid app had earned(mean)
a = sns.barplot(y = list(earning.keys()) , x = list(earning.values()) ,
                saturation=1 , 
                palette = "GnBu_d", 
               )
earning={}
for i in np.unique(df["Content Rating"]):
        earning[i] = np.mean(df.loc[(df["Content Rating"] == i) & (df.Type == "Paid") , "Price"])*np.median(df.loc[(df["Content Rating"] == i) & (df.Type == "Paid")  , "Installs"]) 
earning
#how much paid app had earned(median)
a = sns.barplot(y = list(earning.keys()) , x = list(earning.values()) ,
                saturation=1 , 
                palette = "GnBu_d", 
               )
#mean of price of each category 
a = sns.barplot(y = df.loc[(df.Type == "Paid") , "Content Rating"]  ,
                x = df.loc[(df.Type == "Paid") , "Price"] , ci = None ,
                saturation=1 , 
                palette = "GnBu_d", 
               )
#numbers of app of each content rating
a = sns.countplot(y = df["Content Rating"] ,
                data = df , 
                hue = df["Type"] ,
                saturation=1 , 
                palette = "RdBu",
                order = df["Content Rating"].value_counts().index)
#rating of each content rating
a = sns.barplot(y = df["Content Rating"], x = df["Rating"] ,
                saturation=1 , 
                hue = df["Type"] ,
                palette = "RdBu",
                data = df)
#installs of each content rating
a = sns.barplot(y = df["Content Rating"], x = df["Installs"] ,
                saturation=1 , 
                hue = df["Type"] ,
                palette = "RdBu",
                data = df, ci=None)
#pie chart of free app
number_of_apps_in_category_free = df.loc[df['Type']=="Free"]["Category"].value_counts().sort_values(ascending=True)

data = [go.Pie(
        labels = number_of_apps_in_category_free.index,
        values = number_of_apps_in_category_free.values,
        hoverinfo = 'label+value'
    
)]

plotly.offline.iplot(data, filename='active_category')
#pie chart of paid app
number_of_apps_in_category_paid = df.loc[df['Type']=="Paid"]["Category"].value_counts().sort_values(ascending=True)

data = [go.Pie(
        labels = number_of_apps_in_category_paid.index,
        values = number_of_apps_in_category_paid.values,
        hoverinfo = 'label+value'
    
)]

plotly.offline.iplot(data, filename='active_category')
#overall rating distribution of paid apps
data = [go.Histogram(
        x = df.loc[df['Type']=="Paid"].Rating,
        xbins = {'start': 1, 'size': 0.1, 'end' :5}
)]

print('Average app rating = ', np.mean(df.loc[df['Type']=="Paid"].Rating))
plotly.offline.iplot(data, filename='overall_rating_distribution')
#overall rating distribution of free apps
data = [go.Histogram(
        x = df.loc[df['Type']=="Free"].Rating,
        xbins = {'start': 1, 'size': 0.1, 'end' :5}
)]

print('Average app rating = ', np.mean(df.loc[df['Type']=="Free"].Rating))
plotly.offline.iplot(data, filename='overall_rating_distribution')
df["Size"] = df["Size"].apply(lambda x : float(x) if x != "Varies with device" else False)
df = df.loc[df["Size"] != False].dropna()

#heat map
df_cor = df.loc[:,["Rating","Reviews","Installs","Price","Size"]]
df_cor = (df_cor - np.mean(df_cor))/np.std(df_cor)
cor = df_cor.corr()
mask = np.zeros_like(cor, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(cor , cmap = cmap , mask = mask , annot = True)
df["size"] = pd.qcut(df["Size"] , 4)
earning={}

for i in np.unique(df["size"]):
        
        earning[i] = np.mean(df.loc[(df["size"] == i) & (df.Type == "Paid") , "Price"])*np.mean(df.loc[(df["size"] == i) & (df.Type == "Paid")  , "Installs"]) 
earning
#how much paid app had earned(mean)
a = sns.barplot(x = list(earning.keys()) , y = list(earning.values()) ,
                saturation=1 , 
                palette = "GnBu_d", 
               )
earning={}
for i in np.unique(df["size"]):
        earning[i] = np.mean(df.loc[(df["size"] == i) & (df.Type == "Paid") , "Price"])*np.median(df.loc[(df["size"] == i) & (df.Type == "Paid")  , "Installs"]) 
earning
#how much paid app had earned(median)
a = sns.barplot(x = list(earning.keys()) , y = list(earning.values()) ,
                saturation=1 , 
                palette = "GnBu_d", 
               )
#mean of price of each category 
a = sns.barplot(x = df.loc[(df.Type == "Paid") , "size"]   ,
                y = df.loc[(df.Type == "Paid") , "Price"] , ci = None ,
                saturation=1 , 
                palette = "GnBu_d", 
                data = df
               )
# rating of apps with different size 
a = sns.barplot(y = df["size"] , x = df["Rating"],
                hue = df["Type"] ,
                palette = "RdBu",
                data = df )
print(np.unique(df["size"]))
plt.yticks([0,1,2,3],["0","1","2","3"])

# installs of apps with different size 
a = sns.barplot(y = df["size"] , x = df["Installs"],ci = None ,
                hue = df["Type"] ,
                palette = "RdBu",
                data = df )
plt.yticks([0,1,2,3],["0","1","2","3"])