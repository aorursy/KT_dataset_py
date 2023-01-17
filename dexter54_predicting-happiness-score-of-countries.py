import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt 

import plotly.express as px #For animated plots
happiness_2015=pd.read_csv("../input/world-happiness/2015.csv")

happiness_2016=pd.read_csv("../input/world-happiness/2016.csv")

happiness_2017=pd.read_csv("../input/world-happiness/2017.csv")

happiness_2018=pd.read_csv("../input/world-happiness/2018.csv")

happiness_2019=pd.read_csv("../input/world-happiness/2019.csv")
happiness_2015.head() 
happiness_2015.drop(["Standard Error","Dystopia Residual"],axis=1,inplace=True)

#Dropping columns that I mentioned

happiness_2015.head() #Checking updated dataframe
happiness_2015.info() #Happines 2015 have 158 rows and 12 columns
happiness_2015.describe()#We can see averages, means of features
happiness_2015.isnull().any() 
happiness_2016.head()
#Dropping columns

happiness_2016.drop(["Lower Confidence Interval","Upper Confidence Interval","Dystopia Residual"],axis=1,inplace=True)



#Checking Dataframe 

happiness_2016.head()
happiness_2016.info() #We have 157 rows in this dataframe
happiness_2016.describe()
happiness_2016.isnull().any()
happiness_2017.head()
#Changing names of columns

happiness_2017.rename(columns={"Happiness.Rank":"Happiness Rank","Happiness.Score":"Happiness Score","Whisker.high":"Upper Confidence Interval","Whisker.low":"Lower Confidence Interval","Economy..GDP.per.Capita.":"Economy (GDP per Capita)","Health..Life.Expectancy.":"Health (Life Expectancy)","Trust..Government.Corruption.":"Trust (Government Corruption)","Dystopia.Residual":"Dystopia Residual"},inplace=True)



#Dropping uncommon features

happiness_2017.drop(["Upper Confidence Interval","Lower Confidence Interval","Dystopia Residual"],axis=1,inplace=True)



#Checking updates

happiness_2017.head()
happiness_2017.info() 
#Getting countries and regions from happiness 2016

countries_regions=happiness_2016[["Country","Region"]]



#Adding regions to happiness 2017 dataframe

happiness_2017=pd.merge(countries_regions,happiness_2017,on="Country")



happiness_2017.head()
happiness_2017.sort_values(by="Happiness Rank",inplace=True,ignore_index=True)
happiness_2017.head()
happiness_2017.describe()
happiness_2017.isnull().any() #We do not have any nan values
happiness_2018.head()
#Changing Column Names

happiness_2018.rename(columns={"Overall rank":"Happiness Rank","Country or region":"Country","Score":"Happiness Score","Freedom to make life choices":"Freedom","Perceptions of corruption":"Trust (Government Corruption)","GDP per capita":"Economy (GDP per Capita)","Healthy life expectancy":"Health (Life Expectancy)","Social support":"Family"},inplace=True)



#Adding Region Column to Dataframe 

happiness_2018=pd.merge(happiness_2018,countries_regions, on="Country")



#Checking Updates

happiness_2018.head()
happiness_2018.info() #We have 150 rows in this dataframe
happiness_2018.describe()
happiness_2018.isnull().any()
happiness_2018[happiness_2018["Trust (Government Corruption)"].isnull()]
#Finding trust average of Middle East and Northern Africa

middleEast_trust=happiness_2018.groupby("Region")["Trust (Government Corruption)"].mean().loc["Middle East and Northern Africa"]



#Filling nan value with this average

happiness_2018.fillna(middleEast_trust,inplace=True)
happiness_2018.isnull().any() #No nan values left
#Like we did in other dataframes, we will standardise happiness 2019

happiness_2019.head() 
happiness_2019.rename(columns={"Overall rank":"Happiness Rank","Country or region":"Country","Score":"Happiness Score","GDP per capita":"Economy (GDP per Capita)","Social support":"Family","Healthy life expectancy":"Health (Life Expectancy)","Freedom to make life choices":"Freedom","Perceptions of corruption":"Trust (Government Corruption)"},inplace=True)

happiness_2019=pd.merge(happiness_2019,countries_regions,on="Country")
happiness_2019.head()
happiness_2019.describe()
happiness_2019.isnull().any()
#This function will write values of histograms on them

def autolabel(rects,ax):

    """Attach a text label above each bar in *rects*, displaying its height."""

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(round(height,2)),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),  # 3 points vertical offset

                    textcoords="offset points",

                    ha='center', va='bottom')
f,ax=plt.subplots(ncols=2,figsize=(20,15))

rect_1=ax[0].bar(happiness_2015.head(10)["Country"],happiness_2015.head(10)["Happiness Score"],color=["red","yellow","orange","purple","pink","blue","cyan","brown","green","grey"])

rect_2=ax[1].bar(happiness_2015.tail(10)["Country"],happiness_2015.tail(10)["Happiness Score"],color=["red","yellow","orange","purple","pink","blue","cyan","brown","green","grey"])



autolabel(rect_1,ax[0])

autolabel(rect_2,ax[1])

f.tight_layout()

ax[0].set_title("Countries with Highest Happiness Rank in 2015",size=20,color="purple")

ax[1].set_title("Countries with Lowest Happiness Rank in 2015",size=20,color="purple")

ax[0].set_xlabel("Countries",size=15,color="purple")

ax[0].xaxis.set_label_coords(1.03,-0.09)

ax[0].set_ylabel("Happiness Rank",size=15,color="purple")



ax[0].tick_params(axis="x", labelsize=12,rotation=90)

ax[0].tick_params(axis="y", labelsize=12)



ax[1].tick_params(axis="x", labelsize=12,rotation=90)

ax[1].tick_params(axis="y", labelsize=12)



ax[0].set_ylim([0,10])

ax[1].set_ylim([0,10])



plt.show()

plt.figure(figsize=(15,15))

mask_2015= np.triu(np.ones_like(happiness_2015.corr(), dtype=np.bool))

sns.heatmap(happiness_2015.corr(),annot=True,fmt=".2f",mask=mask_2015,linewidth=2,cmap="YlGnBu",vmax=1,vmin=-1)

plt.title("Correlations in Happiness 2015",size=15,color="r")
f,ax=plt.subplots(nrows=2,ncols=2,figsize=(20,20))



sns.scatterplot(data=happiness_2015,x="Economy (GDP per Capita)",y="Happiness Score",hue="Region",s=50,ax=ax[0][0])

sns.scatterplot(data=happiness_2015,x="Health (Life Expectancy)",y="Happiness Score",hue="Region",s=50,ax=ax[0][1])

sns.scatterplot(data=happiness_2015,x="Freedom",y="Happiness Score",hue="Region",s=50,ax=ax[1][0])

sns.scatterplot(data=happiness_2015,x="Trust (Government Corruption)",y="Happiness Score",hue="Region",s=50,ax=ax[1][1])





ax[0][0].grid(True)

ax[0][0].set_title("Happiness Score vs Economy with respect to Region in 2015",size=15, color="orange")

ax[0][0].set_ylabel("Happiness Score",size=12,color="orange")

ax[0][0].set_xlabel("Economy (GDP per Capita)",size=12,color="orange")

ax[0][0].tick_params(axis="x", labelsize=12)

ax[0][0].tick_params(axis="y", labelsize=12)

ax[0][0].legend(prop={"size":8})



ax[0][1].grid(True)

ax[0][1].set_title("Happiness Score vs Health (Life Expectancy) with respect to Region in 2015",size=15, color="orange")

ax[0][1].set_ylabel("Happiness Score",size=12,color="orange")

ax[0][1].set_xlabel("Health (Life Expectancy)",size=12,color="orange")

ax[0][1].tick_params(axis="x", labelsize=12)

ax[0][1].tick_params(axis="y", labelsize=12)

ax[0][1].legend(prop={"size":8})



ax[1][0].grid(True)

ax[1][0].set_title("Happiness Score vs Freedom with respect to Region in 2015",size=15, color="orange")

ax[1][0].set_ylabel("Happiness Score",size=12,color="orange")

ax[1][0].set_xlabel("Freedom",size=12,color="orange")

ax[1][0].tick_params(axis="x", labelsize=12)

ax[1][0].tick_params(axis="y", labelsize=12)

ax[1][0].legend(prop={"size":8})



ax[1][1].grid(True)

ax[1][1].set_title("Happiness Score vs Trust (Government Corruption) with respect to Region in 2015",size=15, color="orange")

ax[1][1].set_ylabel("Happiness Score",size=12,color="orange")

ax[1][1].set_xlabel("Trust (Government Corruption)",size=12,color="orange")

ax[1][1].tick_params(axis="x", labelsize=12)

ax[1][1].tick_params(axis="y", labelsize=12)

ax[1][1].legend(prop={"size":8})



plt.show()
f,ax=plt.subplots(ncols=2,figsize=(20,15))

rect_1=ax[0].bar(happiness_2015.head(10)["Country"],happiness_2016.head(10)["Happiness Score"],color=["red","yellow","orange","purple","pink","blue","cyan","brown","green","grey"])

rect_2=ax[1].bar(happiness_2015.tail(10)["Country"],happiness_2016.tail(10)["Happiness Score"],color=["red","yellow","orange","purple","pink","blue","cyan","brown","green","grey"])

autolabel(rect_1,ax[0])

autolabel(rect_2,ax[1])

f.tight_layout()

ax[0].set_title("Countries with Highest Happiness Rank in 2016",size=20,color="purple")

ax[1].set_title("Countries with Lowest Happiness Rank in 2016",size=20,color="purple")

ax[0].set_xlabel("Countries",size=15,color="purple")

ax[0].xaxis.set_label_coords(1.03,-0.09)

ax[0].set_ylabel("Happiness Rank",size=15,color="purple")



ax[0].tick_params(axis="x", labelsize=12,rotation=90)

ax[0].tick_params(axis="y", labelsize=12)



ax[1].tick_params(axis="x", labelsize=12,rotation=90)

ax[1].tick_params(axis="y", labelsize=12)



ax[0].set_ylim([0,10])

ax[1].set_ylim([0,10])





plt.show()
plt.figure(figsize=(15,15))

mask_2016 = np.triu(np.ones_like(happiness_2016.corr(), dtype=np.bool))

sns.heatmap(happiness_2016.corr(),annot=True,fmt=".2f",mask=mask_2016,linewidth=2,cmap="YlGnBu",vmax=1,vmin=-1)

plt.title("Correlations in Happiness 2016",size=15,color="r")
f,ax=plt.subplots(nrows=2,ncols=2,figsize=(20,20))



sns.scatterplot(data=happiness_2016,x="Economy (GDP per Capita)",y="Happiness Score",hue="Region",s=50,ax=ax[0][0])

sns.scatterplot(data=happiness_2016,x="Health (Life Expectancy)",y="Happiness Score",hue="Region",s=50,ax=ax[0][1])

sns.scatterplot(data=happiness_2016,x="Freedom",y="Happiness Score",hue="Region",s=50,ax=ax[1][0])

sns.scatterplot(data=happiness_2016,x="Trust (Government Corruption)",y="Happiness Score",hue="Region",s=50,ax=ax[1][1])





ax[0][0].grid(True)

ax[0][0].set_title("Happiness Score vs Economy with respect to Region in 2016",size=15, color="orange")

ax[0][0].set_ylabel("Happiness Score",size=12,color="orange")

ax[0][0].set_xlabel("Economy (GDP per Capita)",size=12,color="orange")

ax[0][0].tick_params(axis="x", labelsize=12)

ax[0][0].tick_params(axis="y", labelsize=12)

ax[0][0].legend(prop={"size":8})



ax[0][1].grid(True)

ax[0][1].set_title("Happiness Score vs Health (Life Expectancy) with respect to Region in 2016",size=15,pad=10, color="orange")

ax[0][1].set_ylabel("Happiness Score",size=12,color="orange")

ax[0][1].set_xlabel("Health (Life Expectancy)",size=12,color="orange")

ax[0][1].tick_params(axis="x", labelsize=12)

ax[0][1].tick_params(axis="y", labelsize=12)

ax[0][1].legend(prop={"size":8})



ax[1][0].grid(True)

ax[1][0].set_title("Happiness Score vs Freedom with respect to Region in 2016",size=15, color="orange")

ax[1][0].set_ylabel("Happiness Score",size=12,color="orange")

ax[1][0].set_xlabel("Freedom",size=12,color="orange")

ax[1][0].tick_params(axis="x", labelsize=12)

ax[1][0].tick_params(axis="y", labelsize=12)

ax[1][0].legend(prop={"size":8})



ax[1][1].grid(True)

ax[1][1].set_title("Happiness Score vs Trust (Government Corruption) with respect to Region in 2016",pad=10,size=15, color="orange")

ax[1][1].set_ylabel("Happiness Score",size=12,color="orange")

ax[1][1].set_xlabel("Trust (Government Corruption)",size=12,color="orange")

ax[1][1].tick_params(axis="x", labelsize=12)

ax[1][1].tick_params(axis="y", labelsize=12)

ax[1][1].legend(prop={"size":8})



plt.show()
f,ax=plt.subplots(ncols=2,figsize=(20,15))

rect_1=ax[0].bar(happiness_2017.head(10)["Country"],happiness_2017.head(10)["Happiness Score"],color=["red","yellow","orange","purple","pink","blue","cyan","brown","green","grey"])

rect_2=ax[1].bar(happiness_2017.tail(10)["Country"],happiness_2017.tail(10)["Happiness Score"],color=["red","yellow","orange","purple","pink","blue","cyan","brown","green","grey"])

autolabel(rect_1,ax[0])

autolabel(rect_2,ax[1])

f.tight_layout()

ax[0].set_title("Countries with Highest Happiness Rank in 2017",size=20,color="purple")

ax[1].set_title("Countries with Lowest Happiness Rank in 2017",size=20,color="purple")

ax[0].set_xlabel("Countries",size=15,color="purple")

ax[0].xaxis.set_label_coords(1.03,-0.09)

ax[0].set_ylabel("Happiness Rank",size=15,color="purple")



ax[0].tick_params(axis="x", labelsize=12,rotation=90)

ax[0].tick_params(axis="y", labelsize=12)

ax[1].tick_params(axis="x", labelsize=12,rotation=90)

ax[1].tick_params(axis="y", labelsize=12)



ax[0].set_ylim([0,10])

ax[1].set_ylim([0,10])





plt.show()
plt.figure(figsize=(15,15))

mask_2017= np.triu(np.ones_like(happiness_2017.corr(), dtype=np.bool))

sns.heatmap(happiness_2017.corr(),annot=True,fmt=".2f",mask=mask_2017,linewidth=2,cmap="YlGnBu",vmax=1,vmin=-1)

plt.title("Correlations in Happiness 2017",size=15,color="r")
f,ax=plt.subplots(nrows=2,ncols=2,figsize=(20,20))



sns.scatterplot(data=happiness_2017,x="Economy (GDP per Capita)",y="Happiness Score",hue="Region",s=50,ax=ax[0][0])

sns.scatterplot(data=happiness_2017,x="Health (Life Expectancy)",y="Happiness Score",hue="Region",s=50,ax=ax[0][1])

sns.scatterplot(data=happiness_2017,x="Freedom",y="Happiness Score",hue="Region",s=50,ax=ax[1][0])

sns.scatterplot(data=happiness_2017,x="Trust (Government Corruption)",y="Happiness Score",hue="Region",s=50,ax=ax[1][1])





ax[0][0].grid(True)

ax[0][0].set_title("Happiness Score vs Economy with respect to Region in 2017",size=15, color="orange")

ax[0][0].set_ylabel("Happiness Score",size=12,color="orange")

ax[0][0].set_xlabel("Economy (GDP per Capita)",size=12,color="orange")

ax[0][0].tick_params(axis="x", labelsize=12)

ax[0][0].tick_params(axis="y", labelsize=12)

ax[0][0].legend(prop={"size":8})



ax[0][1].grid(True)

ax[0][1].set_title("Happiness Score vs Health (Life Expectancy) with respect to Region in 2017",size=15,pad=10, color="orange")

ax[0][1].set_ylabel("Happiness Score",size=12,color="orange")

ax[0][1].set_xlabel("Health (Life Expectancy)",size=12,color="orange")

ax[0][1].tick_params(axis="x", labelsize=12)

ax[0][1].tick_params(axis="y", labelsize=12)

ax[0][1].legend(prop={"size":8})



ax[1][0].grid(True)

ax[1][0].set_title("Happiness Score vs Freedom with respect to Region in 2017",size=15, color="orange")

ax[1][0].set_ylabel("Happiness Score",size=12,color="orange")

ax[1][0].set_xlabel("Freedom",size=12,color="orange")

ax[1][0].tick_params(axis="x", labelsize=12)

ax[1][0].tick_params(axis="y", labelsize=12)

ax[1][0].legend(prop={"size":8})



ax[1][1].grid(True)

ax[1][1].set_title("Happiness Score vs Trust (Government Corruption) with respect to Region in 2017",pad=10,size=15, color="orange")

ax[1][1].set_ylabel("Happiness Score",size=12,color="orange")

ax[1][1].set_xlabel("Trust (Government Corruption)",size=12,color="orange")

ax[1][1].tick_params(axis="x", labelsize=12)

ax[1][1].tick_params(axis="y", labelsize=12)

ax[1][1].legend(prop={"size":8})



plt.show()
f,ax=plt.subplots(ncols=2,figsize=(20,15))

rect_1=ax[0].bar(happiness_2018.head(10)["Country"],happiness_2018.head(10)["Happiness Score"],color=["red","yellow","orange","purple","pink","blue","cyan","brown","green","grey"])

rect_2=ax[1].bar(happiness_2018.tail(10)["Country"],happiness_2018.tail(10)["Happiness Score"],color=["red","yellow","orange","purple","pink","blue","cyan","brown","green","grey"])

autolabel(rect_1,ax[0])

autolabel(rect_2,ax[1])

f.tight_layout()

ax[0].set_title("Countries with Highest Happiness Rank in 2018",size=20,color="purple")

ax[1].set_title("Countries with Lowest Happiness Rank in 2018",size=20,color="purple")

ax[0].set_xlabel("Countries",size=15,color="purple")

ax[0].xaxis.set_label_coords(1.03,-0.09)

ax[0].set_ylabel("Happiness Rank",size=15,color="purple")



ax[0].tick_params(axis="x", labelsize=12,rotation=90)

ax[0].tick_params(axis="y", labelsize=12)



ax[1].tick_params(axis="x", labelsize=12,rotation=90)

ax[1].tick_params(axis="y", labelsize=12)



ax[0].set_ylim([0,10])

ax[1].set_ylim([0,10])



plt.show()

plt.figure(figsize=(15,15))

mask_2018 = np.triu(np.ones_like(happiness_2018.corr(), dtype=np.bool))

sns.heatmap(happiness_2018.corr(),annot=True,fmt=".2f",mask=mask_2018,linewidth=2,cmap="YlGnBu",vmax=1,vmin=-1)

plt.title("Correlations in Happiness 2018",size=15,color="r")
f,ax=plt.subplots(nrows=2,ncols=2,figsize=(20,20))



sns.scatterplot(data=happiness_2018,x="Economy (GDP per Capita)",y="Happiness Score",hue="Region",s=50,ax=ax[0][0])

sns.scatterplot(data=happiness_2018,x="Health (Life Expectancy)",y="Happiness Score",hue="Region",s=50,ax=ax[0][1])

sns.scatterplot(data=happiness_2018,x="Freedom",y="Happiness Score",hue="Region",s=50,ax=ax[1][0])

sns.scatterplot(data=happiness_2018,x="Trust (Government Corruption)",y="Happiness Score",hue="Region",s=50,ax=ax[1][1])





ax[0][0].grid(True)

ax[0][0].set_title("Happiness Score vs Economy with respect to Region in 2018",size=15, color="orange")

ax[0][0].set_ylabel("Happiness Score",size=12,color="orange")

ax[0][0].set_xlabel("Economy (GDP per Capita)",size=12,color="orange")

ax[0][0].tick_params(axis="x", labelsize=12)

ax[0][0].tick_params(axis="y", labelsize=12)

ax[0][0].legend(prop={"size":8})



ax[0][1].grid(True)

ax[0][1].set_title("Happiness Score vs Health (Life Expectancy) with respect to Region in 2018",size=15,pad=10, color="orange")

ax[0][1].set_ylabel("Happiness Score",size=12,color="orange")

ax[0][1].set_xlabel("Health (Life Expectancy)",size=12,color="orange")

ax[0][1].tick_params(axis="x", labelsize=12)

ax[0][1].tick_params(axis="y", labelsize=12)

ax[0][1].legend(prop={"size":8})



ax[1][0].grid(True)

ax[1][0].set_title("Happiness Score vs Freedom with respect to Region in 2018",size=15, color="orange")

ax[1][0].set_ylabel("Happiness Score",size=12,color="orange")

ax[1][0].set_xlabel("Freedom",size=12,color="orange")

ax[1][0].tick_params(axis="x", labelsize=12)

ax[1][0].tick_params(axis="y", labelsize=12)

ax[1][0].legend(prop={"size":8})



ax[1][1].grid(True)

ax[1][1].set_title("Happiness Score vs Trust (Government Corruption) with respect to Region in 2018",pad=10,size=15, color="orange")

ax[1][1].set_ylabel("Happiness Score",size=12,color="orange")

ax[1][1].set_xlabel("Trust (Government Corruption)",size=12,color="orange")

ax[1][1].tick_params(axis="x", labelsize=12)

ax[1][1].tick_params(axis="y", labelsize=12)

ax[1][1].legend(prop={"size":8})



plt.show()
f,ax=plt.subplots(ncols=2,figsize=(20,15))

rect_1=ax[0].bar(happiness_2019.head(10)["Country"],happiness_2019.head(10)["Happiness Score"],color=["red","yellow","orange","purple","pink","blue","cyan","brown","green","grey"])

rect_2=ax[1].bar(happiness_2019.tail(10)["Country"],happiness_2019.tail(10)["Happiness Score"],color=["red","yellow","orange","purple","pink","blue","cyan","brown","green","grey"])

autolabel(rect_1,ax[0])

autolabel(rect_2,ax[1])

f.tight_layout()

ax[0].set_title("Countries with Highest Happiness Rank in 2019",size=20,color="purple")

ax[1].set_title("Countries with Lowest Happiness Rank in 2019",size=20,color="purple")

ax[0].set_xlabel("Countries",size=15,color="purple")

ax[0].xaxis.set_label_coords(1.03,-0.09)

ax[0].set_ylabel("Happiness Rank",size=15,color="purple")



ax[0].tick_params(axis="x", labelsize=12,rotation=90)

ax[0].tick_params(axis="y", labelsize=12)



ax[1].tick_params(axis="x", labelsize=12,rotation=90)

ax[1].tick_params(axis="y", labelsize=12)



ax[0].set_ylim([0,10])

ax[1].set_ylim([0,10])



plt.show()
plt.figure(figsize=(15,15))

mask_2019 = np.triu(np.ones_like(happiness_2019.corr(), dtype=np.bool))

sns.heatmap(happiness_2019.corr(),annot=True,fmt=".2f",mask=mask_2019,linewidth=2,cmap="YlGnBu",vmax=1,vmin=-1)

plt.title("Correlations in Happiness 2019",size=15,color="r")
f,ax=plt.subplots(nrows=2,ncols=2,figsize=(20,20))



sns.scatterplot(data=happiness_2019,x="Economy (GDP per Capita)",y="Happiness Score",hue="Region",s=50,ax=ax[0][0])

sns.scatterplot(data=happiness_2019,x="Health (Life Expectancy)",y="Happiness Score",hue="Region",s=50,ax=ax[0][1])

sns.scatterplot(data=happiness_2019,x="Freedom",y="Happiness Score",hue="Region",s=50,ax=ax[1][0])

sns.scatterplot(data=happiness_2019,x="Trust (Government Corruption)",y="Happiness Score",hue="Region",s=50,ax=ax[1][1])





ax[0][0].grid(True)

ax[0][0].set_title("Happiness Score vs Economy with respect to Region in 2019",size=15, color="orange")

ax[0][0].set_ylabel("Happiness Score",size=12,color="orange")

ax[0][0].set_xlabel("Economy (GDP per Capita)",size=12,color="orange")

ax[0][0].tick_params(axis="x", labelsize=12)

ax[0][0].tick_params(axis="y", labelsize=12)

ax[0][0].legend(prop={"size":8})



ax[0][1].grid(True)

ax[0][1].set_title("Happiness Score vs Health (Life Expectancy) with respect to Region in 2019",size=15,pad=10, color="orange")

ax[0][1].set_ylabel("Happiness Score",size=12,color="orange")

ax[0][1].set_xlabel("Health (Life Expectancy)",size=12,color="orange")

ax[0][1].tick_params(axis="x", labelsize=12)

ax[0][1].tick_params(axis="y", labelsize=12)

ax[0][1].legend(prop={"size":8})



ax[1][0].grid(True)

ax[1][0].set_title("Happiness Score vs Freedom with respect to Region in 2019",size=15, color="orange")

ax[1][0].set_ylabel("Happiness Score",size=12,color="orange")

ax[1][0].set_xlabel("Freedom",size=12,color="orange")

ax[1][0].tick_params(axis="x", labelsize=12)

ax[1][0].tick_params(axis="y", labelsize=12)

ax[1][0].legend(prop={"size":8})



ax[1][1].grid(True)

ax[1][1].set_title("Happiness Score vs Trust (Government Corruption) with respect to Region in 2019",pad=10,size=15, color="orange")

ax[1][1].set_ylabel("Happiness Score",size=12,color="orange")

ax[1][1].set_xlabel("Trust (Government Corruption)",size=12,color="orange")

ax[1][1].tick_params(axis="x", labelsize=12)

ax[1][1].tick_params(axis="y", labelsize=12)

ax[1][1].legend(prop={"size":8})



plt.show()
#Creating MetaData

Data=pd.DataFrame(columns=["Region","Country","Happiness Rank","Economy (GDP per Capita)","Family","Health (Life Expectancy)","Freedom","Trust (Government Corruption)","Generosity","Happiness Score"])



#I am every column year infomation to differentiate happiness scores with years

happiness_2015.insert(loc=0,column="Year",value=2015)

happiness_2016.insert(loc=0,column="Year",value=2016)

happiness_2017.insert(loc=0,column="Year",value=2017)

happiness_2018.insert(loc=0,column="Year",value=2018)

happiness_2019.insert(loc=0,column="Year",value=2019)



#Adding every dataframes to our main data

Data=Data.append(happiness_2015)

Data=Data.append(happiness_2016)

Data=Data.append(happiness_2017)

Data=Data.append(happiness_2018)

Data=Data.append(happiness_2019)
#I will use pycountry to get iso alpha country codes of countries for plotting in world map 

import pycountry

alpha_codes=[]

countries=[]

for c in pycountry.countries:

    alpha_codes.append(c.alpha_3)

    countries.append(c.name)
countries_codes=pd.DataFrame(zip(countries,alpha_codes),columns=["Country","Code"])
#I am adding  ISO alpha codes in our dataset

Data_codes=pd.merge(left=Data,right=countries_codes,how="left",left_on='Country',right_on='Country')
Data_codes.head()
#Creating animation by plotly express to show happines scores of countries in years

import plotly.express as px

fig = px.choropleth(Data_codes, 

                    locations ="Code", 

                    color ="Happiness Score", 

                    hover_name ="Country",  

                    color_continuous_scale = px.colors.sequential.Plasma, 

                    scope ="world", 

                    animation_frame ="Year") 

fig.update_layout(transition = {'duration': 0})

fig.show()
f,ax=plt.subplots(nrows=2,ncols=2,figsize=(20,20))



sns.scatterplot(data=Data_codes,x="Economy (GDP per Capita)",y="Happiness Score",hue="Region",s=50,ax=ax[0][0])

sns.scatterplot(data=Data_codes,x="Health (Life Expectancy)",y="Happiness Score",hue="Region",s=50,ax=ax[0][1])

sns.scatterplot(data=Data_codes,x="Freedom",y="Happiness Score",hue="Region",s=50,ax=ax[1][0])

sns.scatterplot(data=Data_codes,x="Trust (Government Corruption)",y="Happiness Score",hue="Region",s=50,ax=ax[1][1])





ax[0][0].grid(True)

ax[0][0].set_title("Happiness Score vs Economy with respect to Region in General",size=15, color="orange")

ax[0][0].set_ylabel("Happiness Score",size=12,color="orange")

ax[0][0].set_xlabel("Economy (GDP per Capita)",size=12,color="orange")

ax[0][0].tick_params(axis="x", labelsize=12)

ax[0][0].tick_params(axis="y", labelsize=12)

ax[0][0].legend(prop={"size":8})



ax[0][1].grid(True)

ax[0][1].set_title("Happiness Score vs Health (Life Expectancy) with respect to Region in 2019",size=15,pad=10, color="orange")

ax[0][1].set_ylabel("Happiness Score",size=12,color="orange")

ax[0][1].set_xlabel("Health (Life Expectancy)",size=12,color="orange")

ax[0][1].tick_params(axis="x", labelsize=12)

ax[0][1].tick_params(axis="y", labelsize=12)

ax[0][1].legend(prop={"size":8})



ax[1][0].grid(True)

ax[1][0].set_title("Happiness Score vs Freedom with respect to Region in General",size=15, color="orange")

ax[1][0].set_ylabel("Happiness Score",size=12,color="orange")

ax[1][0].set_xlabel("Freedom",size=12,color="orange")

ax[1][0].tick_params(axis="x", labelsize=12)

ax[1][0].tick_params(axis="y", labelsize=12)

ax[1][0].legend(prop={"size":8})



ax[1][1].grid(True)

ax[1][1].set_title("Happiness Score vs Trust (Government Corruption) with respect to Region in General",pad=10,size=15, color="orange")

ax[1][1].set_ylabel("Happiness Score",size=12,color="orange")

ax[1][1].set_xlabel("Trust (Government Corruption)",size=12,color="orange")

ax[1][1].tick_params(axis="x", labelsize=12)

ax[1][1].tick_params(axis="y", labelsize=12)

ax[1][1].legend(prop={"size":8})



plt.show()
df_regions=Data.groupby("Region")["Happiness Score"].mean()

df_regions.head(10)
plt.figure(figsize=(15,15))

rec=sns.barplot(x=df_regions.index,y=df_regions.values)

plt.xticks(rotation=90,size=13)

plt.yticks(size=13)

plt.title("Average Happiness Score in Regions",size=20,color="purple")

plt.xlabel("Regions",color="red",size=20)

plt.ylabel("Happiness Score",color="red",size=20)



plt.plot()
columns=["Region","Country","Economy (GDP per Capita)","Family","Health (Life Expectancy)","Freedom","Trust (Government Corruption)","Generosity","Year","Happiness Score"]

Data=Data.reindex(columns=columns)
Data.head()
Data.info()
evaluation_df=pd.DataFrame(columns=["Model","Mean Absolute Error","Mean Squared Error","R Squared Error"])
X=Data["Economy (GDP per Capita)"].values.reshape(-1,1)

y=Data["Happiness Score"].values.reshape(-1,1)
from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
from sklearn.linear_model import LinearRegression

Lg=LinearRegression()

Lg.fit(X_train,y_train)
predictions=Lg.predict(X_test)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

eva_dict={"Model":"Linear Regression","Mean Absolute Error":mean_absolute_error(y_test,predictions),"Mean Squared Error":mean_squared_error(y_test,predictions),"R Squared Error":r2_score(y_test,predictions)}

evaluation_df=evaluation_df.append(eva_dict,ignore_index=True)
evaluation_df.head()
Data.head()
X=Data.drop("Country",axis=1).iloc[:,:-1]

y=Data.iloc[:, -1].values.reshape(-1,1)
X.head()
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

X=pd.get_dummies(X)
X.head()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
lg_1=LinearRegression()

lg_1.fit(X_train,y_train)
predictions_1=lg_1.predict(X_test)
eva_dict={"Model":"Multilinear Regression","Mean Absolute Error":mean_absolute_error(y_test,predictions_1),"Mean Squared Error":mean_squared_error(y_test,predictions_1),"R Squared Error":r2_score(y_test,predictions_1)}

evaluation_df=evaluation_df.append(eva_dict,ignore_index=True)
evaluation_df.head()
from sklearn.preprocessing import StandardScaler



Sc_X=StandardScaler()

Sc_y=StandardScaler()



scaled_X_train=Sc_X.fit_transform(X_train)

scaled_X_test=Sc_X.transform(X_test)

scaled_y_train=Sc_y.fit_transform(y_train)
from sklearn.svm import SVR

regressor=SVR(kernel="rbf")

regressor.fit(scaled_X_train,scaled_y_train)
predictions_2=Sc_y.inverse_transform(regressor.predict(scaled_X_test))
eva_dict={"Model":"Support Vector Regression","Mean Absolute Error":mean_absolute_error(y_test,predictions_2),"Mean Squared Error":mean_squared_error(y_test,predictions_2),"R Squared Error":r2_score(y_test,predictions_2)}

evaluation_df=evaluation_df.append(eva_dict,ignore_index=True)
evaluation_df.head()
from sklearn.tree import DecisionTreeRegressor

regressor=DecisionTreeRegressor()

regressor.fit(X_train,y_train)
predictions_3=regressor.predict(X_test)
eva_dict={"Model":"Decision Tree Regression","Mean Absolute Error":mean_absolute_error(y_test,predictions_3),"Mean Squared Error":mean_squared_error(y_test,predictions_3),"R Squared Error":r2_score(y_test,predictions_3)}

evaluation_df=evaluation_df.append(eva_dict,ignore_index=True)
evaluation_df.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

params_to_test = {

    'n_estimators':range(1,11)

}



rf_model=RandomForestRegressor()

grid_search=GridSearchCV(rf_model, param_grid=params_to_test)



grid_search.fit(X_train,y_train.ravel())
predictions_4=grid_search.predict(X_test)

eva_dict={"Model":"Random Forest Regression","Mean Absolute Error":mean_absolute_error(y_test,predictions_4),"Mean Squared Error":mean_squared_error(y_test,predictions_4),"R Squared Error":r2_score(y_test,predictions_4)}

evaluation_df=evaluation_df.append(eva_dict,ignore_index=True)
evaluation_df.head()
evaluation_df.head()