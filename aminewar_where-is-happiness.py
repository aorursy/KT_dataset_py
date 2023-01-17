import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.mlab import PCA as mlabPCA

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sqlalchemy import create_engine

import warnings

warnings.filterwarnings('ignore')



import plotly.plotly as py #For World Map

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
happiness=pd.read_csv("../input/2017.csv")

happiness.drop(["Whisker.high","Whisker.low","Dystopia.Residual"],axis=1,inplace=True)
happiness.head()
happiness.info()
print("\nHow much is percantage of missing data in columns ?\n")

print(happiness.isnull().sum()*100 /happiness.shape[0])
for sutun_adi in happiness.columns[2:]:

    print("problematic values for {} column : ".format(sutun_adi),end="")

    count=0

    for deger in happiness[sutun_adi]:

        try:

            float(deger)

        except:

            count +=1

            print(deger)

    if count==0:

        print("No problematic values","\n")

plt.rcParams['figure.dpi'] = 100

plt.rcParams['figure.figsize'] = (15,7)



baslik_font = {'family':'arial','color':'darkred','weight':'bold','size':13}

eksen_font = {'family':'arial','color':'darkblue','weight':'bold','size':10}



secili_kolonlar1 =happiness.columns[3:7]

secili_kolonlar2 =happiness.columns[6:]

for i in range(4):

    plt.subplot(2,4,i+1)

    plt.boxplot(happiness[secili_kolonlar1[i]])

    plt.title(secili_kolonlar1[i],fontdict=baslik_font)



for i in range(4):

    plt.subplot(2,4,i+4)

    plt.boxplot(happiness[secili_kolonlar2[i]])

    plt.title(secili_kolonlar2[i],fontdict=baslik_font)



plt.show()
from scipy.stats.mstats import winsorize

happiness["Winsorize_Family"]=winsorize(happiness["Family"],(0.028,0))

happiness["Winsorize_Generosity"]=winsorize(happiness["Generosity"],(0,0.014))

happiness["Winsorize_Trust_Government_Corruption"]=winsorize(happiness["Trust..Government.Corruption."],(0,0.090))
plt.rcParams['figure.figsize'] = (12,5)

baslik_font = {'family':'arial','color':'darkred','weight':'bold','size':10}



winsorize_kolonlar=["Winsorize_Family","Winsorize_Generosity","Winsorize_Trust_Government_Corruption"]



for i in range(3):

    plt.subplot(1,3,i+1)

    plt.boxplot(happiness[winsorize_kolonlar[i]])

    plt.title(winsorize_kolonlar[i],fontdict=baslik_font)



plt.show() 
plt.figure(figsize=(15,10))



baslik_font = {'family':'arial','color':'darkred','weight':'bold','size':12}





secili_kolonlar=["Family","Generosity","Trust..Government.Corruption."]

winsorize_kolonlar=["Winsorize_Family","Winsorize_Generosity","Winsorize_Trust_Government_Corruption"]



for i in range(3):

    plt.subplot(2,3,i+1)

    plt.hist(happiness[secili_kolonlar[i]])

    plt.title(secili_kolonlar[i],fontdict=baslik_font)

    

for i in range(3):

    plt.subplot(2,3,i+4)

    plt.hist(happiness[winsorize_kolonlar[i]])

    plt.title(winsorize_kolonlar[i],fontdict=baslik_font)

    

plt.show()   


win_happiness=happiness[['Country', 'Region', 'Happiness.Rank', 'Happiness.Score',

               'Economy..GDP.per.Capita.','Health..Life.Expectancy.',

              'Freedom','Winsorize_Family', 'Winsorize_Generosity',

              'Winsorize_Trust_Government_Corruption']]
win_happiness.describe()
plt.figure(figsize=(13,5))



baslik_font = {'family':'arial','color':'darkred','weight':'bold','size':12}



secili_kolonlar1=win_happiness.columns[3:7]





for i in range(4):

    plt.subplot(1,4,i+1)

    plt.hist(win_happiness[secili_kolonlar1[i]])

    plt.title(secili_kolonlar1[i],fontdict=baslik_font)

    

plt.show() 
plt.figure(figsize=(15,15))

baslik_font = {'family': 'arial', 'color': 'purple','weight': 'bold','size': 11 }

eksen_font  = {'family': 'arial', 'color': 'gray','weight': 'bold','size':10 }





degiskenler1=win_happiness.columns[4:7]

degiskenler2=win_happiness.columns[7:]

basliklar=["Happiness Score","Economy","Health","Freedom","Winsorize Family",

          "Winsorize Generosity","Winsorize Trust Government"]





for i in range(3):

    plt.subplot(3,2,i+1)

    plt.scatter(win_happiness["Happiness.Score"],win_happiness[degiskenler1[i]],color="g")

    plt.xlabel(basliklar[0],fontdict=eksen_font)

    plt.ylabel(basliklar[i+1],fontdict=eksen_font)

    plt.title(basliklar[0]+" & "+basliklar[i+1],fontdict=baslik_font)



for i in range(3):

    plt.subplot(3,2,i+4)

    plt.scatter(win_happiness["Happiness.Score"],win_happiness[degiskenler2[i]],color="g")

    plt.xlabel(basliklar[0],fontdict=eksen_font)

    plt.ylabel(basliklar[i+4],fontdict=eksen_font)

    plt.title(basliklar[0]+" & "+basliklar[i+4],fontdict=baslik_font)



plt.show()    
plt.figure(figsize=(15,15))

korelasyon_happiness=win_happiness.corr()

sns.heatmap(korelasyon_happiness,square=True,annot=True,linewidths=.5,vmin=0,vmax=1,cmap='viridis')

plt.title("Happiness Correlation Matrix",color="green",fontsize=20)



plt.show()
avarage_happiness=happiness.groupby(by="Region").mean()["Happiness.Score"].sort_values(ascending=False)



plt.figure(figsize=(10,6))

sns.barplot(x=avarage_happiness.index,y=avarage_happiness.values)

plt.xticks(rotation=90)

plt.xlabel("Regions",color="purple",fontsize=13)

plt.ylabel("Avarage Happiness Score",color="purple",fontsize=13)

plt.title("Happiness Score of Regions",color="purple",fontsize=17)

plt.show()
avarage_happiness=happiness.groupby(by="Region").mean()["Happiness.Score"].sort_values(ascending=False)

regions=list(avarage_happiness.index)

regions_economy=[]

regions_health=[]





for region in regions:

    x=win_happiness[win_happiness["Region"]==region]

    regions_health.append(sum(x["Health..Life.Expectancy."])/len(x))

    regions_economy.append(sum(x["Economy..GDP.per.Capita."])/len(x))

   

    

f,ax = plt.subplots(figsize=(8,5))

sns.set_color_codes("pastel")

sns.barplot(x=regions_health,y=regions,color='purple',label="Health")

sns.barplot(x=regions_economy,y=regions,color='b',alpha=0.5,label="Economy")





ax.legend(loc='lower right',frameon=True)

#ax.set(xlabel='Ratio of Health and Economy',ylabel='Region',title="Economic Factors Affecting Happiness by Regions")

plt.xlabel("Ratio of Health and Economy",color="hotpink",fontsize=12)

plt.ylabel("Region",color="hotpink",fontsize=12)

plt.title("Economic Factors Affecting Happiness by Regions",color="hotpink",fontsize=14)

plt.show()
avarage_happiness=happiness.groupby(by="Region").mean()["Happiness.Score"].sort_values(ascending=False)

regions=list(avarage_happiness.index)

regions_generosity=[]

regions_family=[]

regions_freedom=[]

regions_trust_government=[]





for region in regions:

    x=win_happiness[win_happiness["Region"]==region]

    regions_generosity.append(sum(x["Winsorize_Generosity"])/len(x))

    regions_family.append(sum(x["Winsorize_Family"])/len(x))

    regions_freedom.append(sum(x["Freedom"])/len(x))

    regions_trust_government.append(sum(x["Winsorize_Trust_Government_Corruption"])/len(x))

   

    

    

f,ax = plt.subplots(figsize=(8,5))

sns.set_color_codes("pastel")

sns.barplot(x=regions_generosity,y=regions,color='green',alpha=0.9,label="Generosity")

sns.barplot(x=regions_family,y=regions,color='pink',alpha=0.5,label="Family")

sns.barplot(x=regions_freedom,y=regions,color='y',alpha=0.5,label="Freedom")

sns.barplot(x=regions_trust_government,y=regions,color='blue',alpha=0.5,label="Trust Government")





ax.legend(loc='lower right',frameon=True)

plt.xlabel("Ratio of Factors Affecting Happiness",color="mediumpurple",fontsize=12)

plt.ylabel("Region",color="mediumpurple",fontsize=12)

plt.title("Social Factors Affecting Happiness by Regions",color="mediumpurple",fontsize=14)

plt.show()

plt.figure(figsize = (5,2))

x=["above_avarage"if i>win_happiness["Happiness.Score"].mean() else "below_avarage"

                   for i in win_happiness["Happiness.Score"]]

avarage_happiness=pd.DataFrame({"case":x})

sns.countplot(avarage_happiness.case)

plt.title("Numbers of above or below Avarage Happiness",color="red",fontsize=11)

plt.show()
win_happiness["avarage_case"]=["above_avarage"if i>win_happiness["Happiness.Score"].mean() else "below_avarage"

                   for i in win_happiness["Happiness.Score"]]

data=win_happiness[["Country","Region","Happiness.Score","avarage_case"]]

regions=(data.groupby(by="Region").mean()["Happiness.Score"].sort_values(ascending=False)).index





x=pd.DataFrame()



for region in regions:

    x=pd.concat([x,data[data["Region"]==region]],axis=0)

    

x    
plt.figure(figsize=(25,10))                               

sns.swarmplot(x="Region",y="Happiness.Score", hue="avarage_case",data = x,size=10)

plt.title("Avarage Happiness Case in Regions",color="red",fontsize=30)

plt.xticks(rotation=90,fontsize=18)

plt.yticks(fontsize=18)

plt.legend(fontsize='xx-large')

plt.xlabel("Regions",color="red",fontsize=25)

plt.ylabel("Happiness Score",color="red",fontsize=25)

plt.show()




above_happiness_avarage=win_happiness[win_happiness["avarage_case"]=="above_avarage"]

new_index=(above_happiness_avarage["Happiness.Score"].sort_values(ascending=False)).index.values

sorted_above_happiness_avarage=above_happiness_avarage.reindex(new_index)



plt.figure(figsize=(15,7))

sns.barplot(x=sorted_above_happiness_avarage["Country"],y=sorted_above_happiness_avarage["Happiness.Score"])

plt.xticks(rotation=90)

plt.xlabel("Country",color="green",fontsize=15)

plt.ylabel("Happiness Score",color="green",fontsize=15)

plt.title("Country Having Above Avarage Happiness",color="purple",fontsize=20)

plt.show()

win_happiness[win_happiness["Country"]=="Turkey"]


new_index=win_happiness.sort_values(by=["Happiness.Rank"]).index.values

sort_win_happiness=win_happiness.reindex(new_index )

sort_win_happiness=sort_win_happiness.set_index("Happiness.Rank")

most_happiness=sort_win_happiness.loc[:10,["Country","Region","Happiness.Score","Economy..GDP.per.Capita.","Health..Life.Expectancy.",

                                        "Freedom","Winsorize_Family","Winsorize_Generosity",

                                        "Winsorize_Trust_Government_Corruption",'Winsorize_Dystopia_Residual']]

most_happiness

plt.figure(figsize=(10,6))

sns.barplot(most_happiness["Country"],most_happiness["Economy..GDP.per.Capita."])

sns.barplot(most_happiness["Country"],most_happiness["Health..Life.Expectancy."])



plt.xticks(rotation=90)

plt.xlabel("Country",color="purple",fontsize=13)

plt.ylabel("Economy",color="purple",fontsize=13)

plt.title("Most Happy Countries Economy",color="purple",fontsize=17)

plt.show()
f,ax = plt.subplots(figsize=(13,7))



sns.pointplot(x=most_happiness["Country"],y=most_happiness["Health..Life.Expectancy."],color="purple",alpha=0.9,label="Health",)

sns.pointplot(x=most_happiness["Country"],y=most_happiness["Freedom"],color="blue",alpha=0.5,label="Freedom")

sns.pointplot(x=most_happiness["Country"],y=most_happiness["Winsorize_Family"],color="gold",alpha=0.5,label="Family")

sns.pointplot(x=most_happiness["Country"],y=most_happiness["Winsorize_Trust_Government_Corruption"],alpha=0.5,color="red",label="Trust Government")

sns.pointplot(x=most_happiness["Country"],y=most_happiness["Winsorize_Generosity"],color="grey",alpha=0.5,label="Generosity")







plt.text(7.5,1.30,"Family",color="gold",fontsize=14,style="italic")

plt.text(7.5,1.20,"Health",color="purple",fontsize=14,style="italic")

plt.text(7.5,1.10,"Freedom",color="blue",fontsize=14,style="italic")

plt.text(7.5,1.00,"Generosity",color="grey",fontsize=14,style="italic")

plt.text(7.5,0.90,"Trust Government",color="red",fontsize=14,style="italic")







plt.title("Most Happy Countrys Factors Affecting",color="green",fontsize=17)

plt.xlabel("Country",color="green",fontsize=14)

plt.ylabel("Factors Affecting Happiness",color="green",fontsize=14)



plt.show()



new_index=win_happiness.sort_values(by=["Happiness.Rank"]).index.values

sort_win_happiness=win_happiness.reindex(new_index )

sort_win_happiness=sort_win_happiness.set_index("Happiness.Rank")

sort_win_happiness=sort_win_happiness[["Country","Region","Happiness.Score","Economy..GDP.per.Capita.","Health..Life.Expectancy.",

                                        "Freedom","Winsorize_Family","Winsorize_Generosity",

                                        "Winsorize_Trust_Government_Corruption"]]

most_unhappiness=sort_win_happiness[-10:]



most_unhappiness

plt.figure(figsize=(10,6))

sns.barplot(most_unhappiness["Country"],most_unhappiness["Economy..GDP.per.Capita."])

plt.xticks(rotation=90)

plt.xlabel("Country",color="purple",fontsize=13)

plt.ylabel("Economy",color="purple",fontsize=13)

plt.title("Most Unhappy Countries Economy",color="purple",fontsize=17)

plt.show()
f,ax = plt.subplots(figsize=(13,7))



sns.pointplot(x=most_unhappiness["Country"],y=most_unhappiness["Health..Life.Expectancy."],color="purple",alpha=0.9,label="Health",)

sns.pointplot(x=most_unhappiness["Country"],y=most_unhappiness["Freedom"],color="blue",alpha=0.5,label="Freedom")

sns.pointplot(x=most_unhappiness["Country"],y=most_unhappiness["Winsorize_Family"],color="gold",alpha=0.5,label="Family")

sns.pointplot(x=most_unhappiness["Country"],y=most_unhappiness["Winsorize_Trust_Government_Corruption"],alpha=0.5,color="red",label="Trust Government")

sns.pointplot(x=most_unhappiness["Country"],y=most_unhappiness["Winsorize_Generosity"],color="grey",alpha=0.5,label="Generosity")





plt.text(0,0.70,"Family",color="gold",fontsize=14,style="italic")

plt.text(0,0.65,"Health",color="purple",fontsize=14,style="italic")

plt.text(0,0.60,"Freedom",color="blue",fontsize=14,style="italic")

plt.text(0,0.55,"Generosity",color="grey",fontsize=14,style="italic")

plt.text(0,0.50,"Trust Government",color="red",fontsize=14,style="italic")







plt.title("Most Unhappy Countries Factors Affecting",color="green",fontsize=17)

plt.xlabel("Country",color="green",fontsize=14)

plt.ylabel("Factors Affecting Happiness",color="green",fontsize=14)



plt.show()

happiness_factors=happiness[["Economy..GDP.per.Capita.","Health..Life.Expectancy.","Freedom","Winsorize_Family",

                            "Winsorize_Generosity","Winsorize_Trust_Government_Corruption"]]



X = StandardScaler().fit_transform(happiness_factors)

sklearn_pca = PCA (n_components=3)

Y_sklearn = sklearn_pca.fit_transform(X)

print(sklearn_pca.explained_variance_ratio_)



data = [dict(

        type='choropleth',

        colorscale = 'Portland',

        locations = happiness['Country'],

        z = happiness["Economy..GDP.per.Capita."],

        locationmode = 'country names',

        text = happiness['Country'],

        colorbar = dict(

        title = 'Economy', 

        titlefont=dict(size=15))

)]

layout = dict(

    title = 'How is Economy ?',

    titlefont = dict(size=25),

    geo = dict(

        showframe = True,

        showcoastlines = True,

        projection = dict(type = 'equirectangular')

        )

)

happiness_map1= go.Figure(data = data, layout = layout)

iplot(happiness_map1, validate=False)
data = [dict(

        type='choropleth',

        colorscale = 'Bluered',

        locations = happiness['Country'],

        z = happiness["Health..Life.Expectancy."],

        locationmode = 'country names',

        text = happiness['Country'],

        colorbar = dict(

        title = 'Health', 

        titlefont=dict(size=15))

)]

layout = dict(

    title = 'How is Health ?',

    titlefont = dict(size=25),

    geo = dict(

        showframe = True,

        showcoastlines = True,

        projection = dict(type = 'equirectangular')

        )

)

happiness_map2= go.Figure(data = data, layout = layout)

iplot(happiness_map2, validate=False)
data = [dict(

        type='choropleth',

        colorscale = 'Picnic',

        locations = happiness['Country'],

        z = happiness["Family"],

        locationmode = 'country names',

        text = happiness['Country'],

        colorbar = dict(

        title = 'Family', 

        titlefont=dict(size=15))

)]

layout = dict(

    title = 'How about Family ?',

    titlefont = dict(size=25),

    geo = dict(

        showframe = True,

        showcoastlines = True,

        projection = dict(type = 'equirectangular')

        )

)

happiness_map3 = go.Figure(data = data, layout = layout)

iplot(happiness_map3, validate=False)
data = [dict(

        type='choropleth',

        colorscale = 'Viridis',

        locations = happiness['Country'],

        z = happiness["Freedom"],

        locationmode = 'country names',

        text = happiness['Country'],

        colorbar = dict(

        title = 'Freedom', 

        titlefont=dict(size=15))

)]

layout = dict(

    title = 'How about Freedom?',

    titlefont = dict(size=25),

    geo = dict(

        showframe = True,

        showcoastlines = True,

        projection = dict(type = 'equirectangular')

        )

)

happiness_map5= go.Figure(data = data, layout = layout)

iplot(happiness_map5, validate=False)
data = [dict(

        type='choropleth',

        colorscale = 'Earth',

        locations = happiness['Country'],

        z = happiness["Trust..Government.Corruption."],

        locationmode = 'country names',

        text = happiness['Country'],

        colorbar = dict(

        title = 'Trust Government', 

        titlefont=dict(size=15))

)]

layout = dict(

    title = 'How about Trust Government Corruption?',

    titlefont = dict(size=25),

    geo = dict(

        showframe = True,

        showcoastlines = True,

        projection = dict(type = 'equirectangular')

        )

)

happiness_map4= go.Figure(data = data, layout = layout)

iplot(happiness_map4, validate=False)
data = [dict(

        type='choropleth',

        colorscale = 'jet',

        locations = happiness['Country'],

        z = happiness["Generosity"],

        locationmode = 'country names',

        text = happiness['Country'],

        colorbar = dict(

        title = 'Generosity', 

        titlefont=dict(size=15))

)]

layout = dict(

    title = 'How about Generosity?',

    titlefont = dict(size=25),

    geo = dict(

        showframe = True,

        showcoastlines = True,

        projection = dict(type = 'equirectangular')

        )

)

happiness_map5= go.Figure(data = data, layout = layout)

iplot(happiness_map5, validate=False)


data = dict(type = 'choropleth',

           colorscale = 'YlGnBu', 

           locations = happiness['Country'],

           locationmode = 'country names',

           z = happiness['Happiness.Score'], 

           text = happiness['Country'],

           colorbar = {'title':'Happiness Score'})

layout = dict(title = 'Where is Happiness ?',

              titlefont = dict(size=25),

             geo = dict(showframe = True,

                        showcoastlines = True,

                       projection = {'type': 'natural earth'}))

happiness_map = go.Figure(data = [data], layout=layout)

iplot(happiness_map)