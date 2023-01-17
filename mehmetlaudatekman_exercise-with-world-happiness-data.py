# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#These libraries are for data visualization

import seaborn as sbrn 

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/world-happiness/2019.csv")

data.info()
data.head()
#TUR:Şimdi tablo üzerinde,özellikler arasındaki bağıntıyı öğreneceğim

#EN: Now, I'll learn the corralation between the features,on the table

data.corr()
f,ax = plt.subplots(figsize=(12,12))

# I have decided heatmap's figsize

sbrn.heatmap(data.corr(),annot=True,linewidth=5,fmt="0.2f",ax=ax)

plt.show()


plt.scatter(data["Score"],data["GDP per capita"],color="green",label="Score-GDP per capita",alpha=0.5)

plt.xlabel("Score")

plt.legend(loc="upper right")

plt.ylabel("GDP per capita")







plt.scatter(data["Score"],data["Social support"],color="blue",label="Score-Social Support",alpha=0.5)

plt.xlabel("Score")

plt.ylabel("Social Support")

plt.legend(loc="upper right")







plt.scatter(data["Score"],data["Healthy life expectancy"],color="red",label="Score-Healthy Life Expectancy",alpha=0.5)

plt.xlabel("Score")

plt.legend(loc="upper right")

plt.ylabel("Healthy Life Expectancy")

plt.show()



feature_names = [i for i in data.columns]

feature_names.remove("Overall rank")

feature_names.remove("Country or region")



feature_colors = ["green","red","blue","purple","black","yellow","pink"]



features_colors = list(zip(feature_names,feature_colors))





def histogram_creator(l):

    figure = plt.figure(figsize = (5,5))

    for i in l:

        ax = figure.add_subplot(4,2,l.index(i)+1)

        data[i[0]].plot(kind="hist",color=i[1],bins=20,figsize=(7,7),label=i[0])

        plt.legend(loc="upper right")

    plt.tight_layout()

    plt.show()





histogram_creator(features_colors)



data_f10 = data.head(10)



country_names = [i[:3] for i in data_f10["Country or region"]]

feature_names = [i for i in data_f10.columns]

feature_names.remove("Country or region")



feature_colors = ["green","red","blue","purple","black","yellow","pink"]



feature_name_nd_colors = list(zip(feature_names,feature_colors))









def create_barplots(country_names,names_nd_colors):

    for f,c in names_nd_colors:

            fig,ax = plt.subplots(figsize=(4,4))

            ax.bar(country_names,data_f10[f],color = c,label = f)

            plt.legend(loc="upper right")

            plt.show()

            

        

create_barplots(country_names,feature_name_nd_colors)
data.describe()
ftrs = ["Score","GDP per capita","Social support","Healthy life expectancy",

                "Freedom to make life choices","Generosity"]



country_names = [i for i in data["Country or region"]] #Creating feature lists

scores = [i for i in data[ftrs[0]]]

gdp = [i for i in data[ftrs[1]]]

social_support = [i for i in data[ftrs[2]]]

healthy_life = [i for i in data[ftrs[3]]]

freedom = [i for i in data[ftrs[4]]]

generosity = [i for i in data[ftrs[5]]]





describings = data.describe() 

mean_list = [] #Creating empty lists

score_list = []

gdp_list = []

social_support_list = []

healthy_list = []

freedom_list = []

generosity_list = []





for ftr in ftrs:

    mean_list.append(describings[ftr]["mean"])

for i in scores:

    if i>= mean_list[0]:        

        score_list.append("HIGH")

    else:

        score_list.append("LOW")



for i in gdp:

    

    if i>= mean_list[1]:

        

        gdp_list.append("HIGH")

    

    else:

        gdp_list.append("LOW")



for i in social_support:

    

    if i>= mean_list[2]:

        

        social_support_list.append("HIGH")

    else:

        social_support_list.append("LOW")



for i in healthy_life:

    

    if i>=mean_list[3]:

        

        healthy_list.append("HIGH")

    else:

        healthy_list.append("LOW")

for i in freedom:

    

    if i>= mean_list[4]:

        

        freedom_list.append("HIGH")

    else:

        freedom_list.append("LOW")



for i in generosity:

    

    if i>= mean_list[5]:

        

        generosity_list.append("HIGH")

    else:

        generosity_list.append("LOW")



zipped = list(zip(country_names,score_list,gdp_list,healthy_list,freedom_list,generosity_list))



new_df = pd.DataFrame(data=zipped,columns=["Country Names","Score","GDP per capita","Healthy life expectancy","Freedom to make life choices","Generosity"])



dt1 = new_df.head(5)



dt2 = new_df.tail(5)



dt3 = pd.concat([dt1,dt2],axis=0)

dt3

ftr_list = [i for i in new_df.columns]

ftr_list.remove("Country Names")



def counter(ftr_li):

    countlist = []

    for ftr in ftr_li:

        true = new_df[ftr] == "HIGH"

        true = len(new_df[true])

        false = new_df[ftr] == "LOW"

        false = len(new_df[false])

        countlist.append([ftr,true,false])

    return countlist



count_list = counter(ftr_list)





def pie_chart_creator(li):



    

    for pck in li:

        figure = plt.figure(figsize = (6,6))

        lbl,h,l = pck

        plt.pie([h,l],labels=["HIGH","LOW"],startangle=90,autopct = "%1.1f%%", )

        plt.title(lbl)

        plt.show()



pie_chart_creator(count_list)

    



#Median visualization

# I'll visualize median with boxplot



feature_list = [i for i in data.columns]

feature_list.remove("Country or region")



def boxplot_creator(feature_list):

    fig = plt.figure(figsize=(10,10))

    for ftr in feature_list:

        ax = fig.add_subplot(4,2,feature_list.index(ftr)+1)

        data.boxplot(column =ftr,grid=True)

    plt.show()

    fig.tight_layout()



boxplot_creator(feature_list)



    
data_tr = data["Country or region"]=="Turkey"

data[data_tr]
data_filter = (data["Country or region"]=="Turkey") | (data["Country or region"]=="Greece") | (data["Country or region"]=="Armenia") | (data["Country or region"]=="Georgia") 

data[data_filter]


lbls_nd_colors= [["GDP per capita","blue"],["Social support","green"],["Healthy life expectancy","red"],["Freedom to make life choices","purple"]]

for label,color in lbls_nd_colors:    

    plt.plot(data[data_filter]["Country or region"],data[data_filter][label],color=color,label=label)

plt.legend(loc = "upper right")

plt.show()
#Creating bar plot



countries = [i for i in data[data_filter]["Country or region"]]



features = ["GDP per capita","Social support","Healthy life expectancy","Freedom to make life choices","Score","Overall rank"]



colors = ["Blue","Cyan","Red","Green","Brown","Pink"]



features_colors = list(zip(features,colors))



def barplot_creator(country_list,feature_color_list,data):

    for f,c in feature_color_list:

        _,ax = plt.subplots(figsize = (4,4))

        ax.bar(country_list,data[f],color=c,label=f)

        plt.legend(loc = "upper right")

        plt.show()

        



barplot_creator(countries,features_colors,data[data_filter])