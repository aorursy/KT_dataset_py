# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats as stats

df = pd.read_csv("../input/video-game-sales-with-ratings/Video_Games_Sales_as_at_22_Dec_2016.csv")



df.head()
df.isnull().sum()
# Firstly we indicated missing values as a graph by using sns library



sns.set_style("whitegrid")

missing = df.isnull().sum()

missing = (missing[missing > 0]) / 16719

missing.sort_values(inplace=True)

missing.plot.bar()

plt.show()
# As a next phase we give a look to correlation table. 

# According to table global sales have strong correlation with NA and EU sales but on other hand not so much strong correlation with JP sales

# As a interesting finding, critic score has only 0.25 correlation with global sales



plt.subplots(figsize=(8,8))

sns.heatmap(df.corr(), annot=True, fmt=".2f")

plt.show()
#This Graph Indicates Missing Values according to console platforms. PS2 has nearly 1200 missing data 





p5 = df.dropna(subset=['Critic_Score', "Developer" ,"Critic_Count" ,"User_Score","User_Count","Rating"])

olanlar_platform = p5.groupby(['Platform']).size().reset_index()

toplam_platform=df.groupby(['Platform']).size().reset_index()



olanlar_platform.plot.bar(x="Platform", figsize= (15, 7))

plt.show()





# Graph in below is percentage version of previous one

d = {'Platform': [], 'Percent': []}  

nihai = pd.DataFrame(data = d)





for i in range(len(toplam_platform)):

    for j in range(len(olanlar_platform)):

        if toplam_platform["Platform"][i] == olanlar_platform["Platform"][j]:

           b = olanlar_platform[0][j] / toplam_platform[0][i] * 100

           nihai.at[j , "Platform"] = olanlar_platform["Platform"][j]

           nihai.at[j , "Percent"] = b

         

            

nihai.plot.barh(x = "Platform" , figsize=(15,7) , color = "red", title = "% ")

plt.show()
# Published Game Genres



g_df =  df[["Genre"]]

genre_group =  g_df.groupby(['Genre']).size().reset_index()



genre_group.plot(kind='pie', y = 0 , autopct='%1.1f%%',  title = "Published Game Genres",

 startangle=90, shadow=False, labels=genre_group['Genre'], legend = False, fontsize=14, figsize=(10, 10))



plt.show()

# Global Sales of Game Genres

#According to this and previous graphs we will see that action type games are published most but platform type games sold most



gb_df =  df[["Genre","Global_Sales"]]

gb_group_sum =  gb_df.groupby(['Genre']).sum().reset_index()

gb_group_mean =  gb_df.groupby(['Genre']).mean().reset_index()





gb_group_sum.plot(kind='pie', y = "Global_Sales" , autopct='%1.1f%%',  title = "Global Sales of Game Genres",startangle=90,

 shadow=False, labels=gb_group_sum['Genre'], legend = False, fontsize=14, figsize=(10, 10))



gb_group_mean.plot(kind='pie', y = "Global_Sales" , autopct='%1.1f%%',  title = "Global Sales of Game Genres ",startangle=90,

 shadow=False, labels=gb_group_mean['Genre'], legend = False, fontsize=14, figsize=(10, 10))


# We examined distribution of global sales of platform genre games . Apperently it is not so close to normal distribution.

df_genre_global = gb_df.copy()

df_globalsales_platform =  df_genre_global[df_genre_global.Genre == "Platform"]

    

sns.distplot(df_globalsales_platform['Global_Sales']);   

fig = plt.figure()

res = stats.probplot(df_globalsales_platform['Global_Sales'], plot=plt)
#Distribution of global sales of platform genre games which is less than 7 millions





df_genre_global_yedidenkucuk = df_genre_global[df_genre_global["Global_Sales"]<7]



sns.distplot(df_genre_global_yedidenkucuk['Global_Sales']);   

fig = plt.figure()

res = stats.probplot(df_genre_global_yedidenkucuk['Global_Sales'], plot=plt)
# Distribution of global sales of platform genre games which is less than 1 million



df_genre_global_birdenkucuk = df_genre_global[df_genre_global["Global_Sales"]<1]



sns.distplot(df_genre_global_birdenkucuk ['Global_Sales']);   

fig = plt.figure()

res = stats.probplot(df_genre_global_birdenkucuk['Global_Sales'], plot=plt)
# Distribution of global sales of platform genre games which is less than 0.2 million





df_genre_global_sifirikidenkucuk = df_genre_global[df_genre_global["Global_Sales"]<0.2]



sns.distplot(df_genre_global_sifirikidenkucuk['Global_Sales']);   

fig = plt.figure()

res = stats.probplot(df_genre_global_sifirikidenkucuk['Global_Sales'], plot=plt)

# We examined distribution of global sales. Apperently it is not so close to normal distribution.



sns.distplot(df_genre_global['Global_Sales']);   

fig = plt.figure()

res = stats.probplot(df_genre_global['Global_Sales'], plot=plt)

#To able to see values better in graphs, we grouped for global sales as 7 groups 





df_genre_global_gruplanmis = df_genre_global.copy()

df_genre_global_gruplanmis['grup'] = pd.Series(len(df_genre_global_gruplanmis['Global_Sales']), index=df_genre_global_gruplanmis.index)

df_genre_global_gruplanmis['grup'] = 0 



df_genre_global_gruplanmis.loc[ df_genre_global_gruplanmis['Global_Sales'] <= 0.05, 'grup']= 0

df_genre_global_gruplanmis.loc[(df_genre_global_gruplanmis['Global_Sales'] > 0.05) & (df_genre_global_gruplanmis['Global_Sales'] <= 0.1), 'grup'] = 1

df_genre_global_gruplanmis.loc[(df_genre_global_gruplanmis['Global_Sales'] > 0.1) & (df_genre_global_gruplanmis['Global_Sales'] <= 0.2), 'grup']= 2

df_genre_global_gruplanmis.loc[(df_genre_global_gruplanmis['Global_Sales'] > 0.2) & (df_genre_global_gruplanmis['Global_Sales'] <= 0.5), 'grup']= 3

df_genre_global_gruplanmis.loc[(df_genre_global_gruplanmis['Global_Sales'] > 0.5) & (df_genre_global_gruplanmis['Global_Sales'] <= 1), 'grup']= 4

df_genre_global_gruplanmis.loc[(df_genre_global_gruplanmis['Global_Sales'] > 1) & (df_genre_global_gruplanmis['Global_Sales'] <= 5), 'grup']= 5

df_genre_global_gruplanmis.loc[ df_genre_global_gruplanmis['Global_Sales'] > 5, 'grup']= 6



print(df_genre_global_gruplanmis.groupby("grup").count())



sns.distplot(df_genre_global_gruplanmis['grup']);   

fig = plt.figure()

res = stats.probplot(df_genre_global_gruplanmis['grup'], plot=plt)



df_genre_global_gruplanmis['grup'] = df_genre_global_gruplanmis['grup'].astype('str')



df_genre_global_gruplanmis.info()
# These graphs indicate publishing and solding amounts of games according to years respectively.



yearly_sales =  df[["Name","Year_of_Release"]]

yearly_sales_df =yearly_sales.groupby(['Year_of_Release']).count()



yearly_global = df[["Name","Year_of_Release","Global_Sales"]]

yearly_global_df = yearly_global.groupby(["Year_of_Release"])[["Global_Sales"]].sum()



sales_yearly=pd.concat([yearly_global_df,yearly_sales_df ],axis=1).reset_index()





sales_yearly.plot.line(x='Year_of_Release', y='Global_Sales')

plt.show()

sales_yearly.plot.line(x='Year_of_Release', y='Name')

plt.show()
# We wanted to get know which platform is popular in which market?

# Apperantly japanese market has a little different inclination



toplam_platform_NA_EU_JP=df.groupby(['Platform'])[["NA_Sales","EU_Sales","JP_Sales"]].sum().reset_index()

a = toplam_platform_NA_EU_JP[["NA_Sales","EU_Sales","JP_Sales"]] / toplam_platform_NA_EU_JP[["NA_Sales","EU_Sales","JP_Sales"]].sum()







print(toplam_platform_NA_EU_JP)

plat_sum = df[["Platform"]]

plat_sum = plat_sum.groupby(["Platform"]).size().reset_index()



deneme = pd.concat([a,plat_sum["Platform"]],axis=1)



plt.figure(figsize=(30,5))

line = sns.lineplot(data=deneme, x="Platform", y="NA_Sales", marker="o", label= "NA_Sales")

line = sns.lineplot(data=deneme, x="Platform", y="EU_Sales", marker="o", label= "EU_Sales")

line = sns.lineplot(data=deneme, x="Platform", y="JP_Sales", marker="o", label= "JP_Sales")



line.set(xticks=deneme.Platform.values)

line.set(ylabel="Percent")

plt.show()



# We wanted to get know which genre is popular in which market?

# Apperantly japanese market has a little different inclination





toplam_genre_NA_EU_JP=df.groupby(['Genre'])[["NA_Sales","EU_Sales","JP_Sales"]].sum().reset_index()

toplam_genre = toplam_genre_NA_EU_JP[["NA_Sales","EU_Sales","JP_Sales"]] / toplam_genre_NA_EU_JP[["NA_Sales","EU_Sales","JP_Sales"]].sum()







print(toplam_genre_NA_EU_JP)

genre_sum = df[["Genre"]]

genre_sum = genre_sum.groupby(["Genre"]).size().reset_index()



genre_sales = pd.concat([toplam_genre,genre_sum["Genre"]],axis=1)





plt.figure(figsize=(30,5))

line = sns.lineplot(data=genre_sales, x="Genre", y="NA_Sales", marker="o", label= "NA_Sales")

line = sns.lineplot(data=genre_sales, x="Genre", y="EU_Sales", marker="o", label= "EU_Sales")

line = sns.lineplot(data=genre_sales, x="Genre", y="JP_Sales", marker="o", label= "JP_Sales")



line.set(xticks=genre_sales.Genre.values)

line.set(ylabel="Percent")

plt.show()


