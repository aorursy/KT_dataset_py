#import of the necessary libaries and define their names 

import numpy as np 

import seaborn as sns 
sns.set_style("whitegrid")

import statsmodels.api as sm 


import pandas as pd 

import matplotlib.pyplot as plt
#to have the graphs in jupyter
%matplotlib inline
import matplotlib.image as mpimg # only neccesary for the picture 


#libaries for ML model (predictive analysis)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split




#upload a happy picture! 
img= mpimg.imread("../input/happiness/HappinessImage.jpeg")
plt.figure(figsize=(15,11)) #make picture larger 
imgplot = plt.imshow(img) 
plt.axis ("off") #remove the axis and tickers 
plt.show()
#import csv datasets and safe as pandas DataFrame
happy15_df = pd.read_csv ("../input/world-happiness/2015.csv")
happy16_df = pd.read_csv ("../input/world-happiness/2016.csv")
happy17_df = pd.read_csv ("../input/world-happiness/2017.csv")
happy18_df = pd.read_csv ("../input/world-happiness/2018.csv")
happy19_df = pd.read_csv ("../input/world-happiness/2019.csv")

#and we have a look at the last one (head is by default with the first 5 rows)
happy19_df.head()


#let's see the shape 
print (" Shape15:" , happy15_df.shape ,"\n",
       "Shape16:" , happy16_df.shape ,"\n",
       "Shape17:" , happy17_df.shape ,"\n",
       "Shape18:" , happy18_df.shape ,"\n",
       "Shape19:" , happy19_df.shape  )



#values and data types of the 5 datasets 
print ( happy15_df.info() 
       , happy16_df.info() 
       , happy17_df.info()
       , happy18_df.info() 
       , happy19_df.info())

print ("different data labels, and confirmation of different variables")
"""
renaming of year 2018 and 2019 
labeling as the base: 2015
Note: very important otherwise we 
will create new colums as soon as we merge the data
"""

#Country 

happy18_df = happy18_df.rename(columns= {"Country or region": "Country"})

happy18_df

happy19_df = happy18_df.rename(columns= {"Country or region": "Country"})

happy19_df


#Happiness Rank 

happy18_df = happy18_df.rename(columns= {"Overall rank": "Happiness Rank"})

happy18_df

happy19_df = happy18_df.rename(columns= {"Overall rank": "Happiness Rank"})

happy19_df



#Economy (GDP per Capita)

happy18_df = happy18_df.rename(columns= {"GDP per capita": "Economy (GDP per Capita)"})

happy18_df

happy19_df = happy18_df.rename(columns= {"GDP per capita": "Economy (GDP per Capita)"})

happy19_df


#Health (Life Expectancy)

happy18_df = happy18_df.rename(columns= {"Healthy life expectancy": "Health (Life Expectancy)"})

happy18_df

happy19_df = happy18_df.rename(columns= {"Healthy life expectancy": "Health (Life Expectancy)"})

happy19_df


#Freedom

happy18_df = happy18_df.rename(columns= {"Freedom to make life choices": "Freedom"})

happy18_df

happy19_df = happy18_df.rename(columns= {"Freedom to make life choices": "Freedom"})

happy19_df


#Trust (Government Corruption)

happy18_df = happy18_df.rename(columns= {"Perceptions of corruption": "Trust (Government Corruption)"})

happy18_df

happy19_df = happy18_df.rename(columns= {"Perceptions of corruption": "Trust (Government Corruption)"})

happy19_df


#Family 

happy18_df = happy18_df.rename(columns= {"Social support": "Family"})

happy18_df

happy19_df = happy18_df.rename(columns= {"Social support": "Family"})

happy19_df


#Happiness Score 

happy18_df = happy18_df.rename(columns= {"Score": "Happiness Score"})

happy18_df

happy19_df = happy18_df.rename(columns= {"Score": "Happiness Score"})

happy15_df


"""
renaming of year 2017
base label: 2015
"""

happy17_df = happy17_df.rename(columns = {"Happiness.Rank": "Happiness Rank"})

happy17_df = happy17_df.rename(columns = {"Happiness.Score": "Happiness Score"})

happy17_df = happy17_df.rename(columns = {"Economy..GDP.per.Capita.": "Economy (GDP per Capita)"})

happy17_df = happy17_df.rename(columns = {"Health..Life.Expectancy.": "Health (Life Expectancy)"})

happy17_df = happy17_df.rename(columns = {"Trust..Government.Corruption.": "Trust (Government Corruption)"})

happy17_df = happy17_df.rename(columns = {"Dystopia.Residual": "Dystopia Residual"})

#because we already know that we want to delete thouse filds 

happy17_df = happy17_df.drop(columns = "Whisker.high") 

happy17_df = happy17_df.drop(columns = "Whisker.low")

happy17_df
#insert year column at first position (index 0)

#2015
happy15_df.insert(0, "Year",value = "2015")


#2016
happy16_df.insert(0, "Year",value = "2016")


#2017
happy17_df.insert(0, "Year",value = "2017")


#2018
happy18_df.insert(0, "Year",value = "2018")

#2019
happy19_df.insert(0, "Year",value = "2019")

#check if it worked
happy18_df.head()
#creating empty dict 
region_dict ={}

#filling with values from DataFame happy15_df 

#index to have each row with both data, region and country 
region_dict = happy15_df [["Country","Region"]].to_dict("index")



[(key, value) for key, value in region_dict.items()]
#concatenating objects 

#defintion of all the sets we want to bring together
frames = [happy15_df, happy16_df,  happy17_df, happy18_df, happy19_df]

happiness = pd.concat (frames)
happiness.info()

#what we see is that "Trust" is not in one line yet
happiness = happiness.drop (["Lower Confidence Interval","Dystopia Residual", "Upper Confidence Interval", "Standard Error"], axis = 1)
happiness.head()


happiness.info()
#conversion of type str to float64 (alternatively: int)
happiness ["Year"] = happiness ["Year"].astype(int)

#convert object data to category data 
happiness ["Country"] = happiness ["Country"].astype("category")
#happiness ["Region"] = happiness ["Region"].astype("category")

happiness.info()
#yellow is what is missig and we see that is substential 

plt.figure (figsize=(10,7))

sns.heatmap(happiness.isnull(),yticklabels=False, cbar = False, cmap = "plasma")
plt.xlabel(xlabel = "variable names", rotation= 0, fontsize= 20)
plt.ylabel (ylabel= "missing data", fontsize = 20)
plt.title (label = "missing data per variable",  fontsize = 25)
plt.show()



#if we would not have the data 

happiness["Region"].fillna("unknown", inplace = True)

happiness.info()
try:
    for country in happiness.Country.unique():
        happiness.loc[happiness['Country']==str(country),
                              'Region']=happiness[happiness['Country']==str(country)].Region.mode()[0]
except IndexError:
    pass

happiness.info()


plt.figure (figsize=(10,7))

sns.heatmap(happiness.isnull(),yticklabels=False, cbar = False, cmap = "plasma")
plt.xlabel(xlabel = "variable names", rotation= 0, fontsize= 20)
plt.ylabel (ylabel= "missing data", fontsize = 20)
plt.title (label = "missing data per variable",  fontsize = 25)
plt.show()



#lets look for the "Region" that are still missing

happiness [happiness["Region"].isna()]

#we see that we only have 6 missing values and will therefore, assign the regions manually 
happiness.loc[[32,70], "Region"] =  "Eastern Asia"
happiness [happiness["Region"].isna()]
#lets look for the missing regions in the rest of the data 
happiness [happiness.Region == "Latin America and Caribbean"]

#replacing of missing values: 
happiness.loc[[37], "Region"] =  "Latin America and Caribbean"
happiness [happiness["Region"].isna()]
#lets look for the missing regions in the rest of the data 
happiness [happiness.Region == "Western Europe"]


#replacing of missing values: 
happiness.loc[[57], "Region"] =  "Western Europe"
happiness [happiness["Region"].isna()]


#we see that there are no missing variables in the region category! 




plt.figure (figsize=(10,7))

sns.heatmap(happiness.isnull(),yticklabels=False, cbar = False, cmap = "plasma")
plt.xlabel(xlabel = "variable names", rotation= 0, fontsize= 20)
plt.ylabel (ylabel= "missing data", fontsize = 20)
plt.title (label = "missing data per variable",  fontsize = 25)
plt.show()


#finding the missing trust variable 
happiness [happiness["Trust (Government Corruption)"].isna()]
"""
We indeed see that the variable missed two trust data in year 2018 and 2019

Replacing the data can be made in different ways, but because we are now 
    dealing with a numeric value we have to have a better understanding of 
    the data 
"""

round (happiness.describe(), 3)
#we find out a few variables to decisde with what we want to replace the missing values

#average of United Arab Emirates

year_trust = happiness.loc [:,"Trust (Government Corruption)":]


mean = year_trust.mean (axis=1)
print ("Mean","\n",mean[19])

median = year_trust.median (axis = 1)
print ("\n","Median","\n",median[19])


#so ahappiness.loc[[57], "Region"] =  "Western Europe"
happiness [happiness["Trust (Government Corruption)"].isna()]

# Replacing the is.na values 

happiness.loc[[19], "Trust (Government Corruption)"] =  0.18600

#we check if we have deleted all the null vales (worked!)
happiness [happiness["Trust (Government Corruption)"].isna()]

#one last time (hopefully)
plt.figure (figsize=(10,7))

sns.heatmap(happiness.isnull(),yticklabels=False, cbar = False, cmap = "plasma")
plt.xlabel(xlabel = "variable names", rotation= 0, fontsize= 20)
plt.ylabel (ylabel= "missing data", fontsize = 20)
plt.title (label = "missing data per variable",  fontsize = 25)
plt.show()

#we see that all the values have been replaced! 

#correlation heatmap

#definition of the correlation of all varialbe in DataFrame
corr = happiness.corr()

fig, ax = plt.subplots(figsize = (8.5,8.5))
ax = sns.heatmap(
    corr, 
    vmin = -1, vmax = 1, center= 0, 
    cmap= sns.cubehelix_palette (20), # insert any number larger than the correlation we want to observe! 
    square = True
)

ax.set_title ("Correlation Heatmap", fontsize = 20)


ax.set_yticklabels (
    ax.get_yticklabels(), 
    fontsize = 12
)

ax.set_xticklabels (
    ax.get_xticklabels(),
    rotation = 45,
    horizontalalignment = "right",
    fontsize = 12
)


#if you dont iclude this line you will have the chart but will all data on top...
plt.show()

#correlation heatmap without mirrowing


corr = happiness.corr()
dropself= np.zeros_like (corr)
dropself [np.triu_indices_from(dropself)] = True

fig, ax = plt.subplots(figsize = (9,9))
ax = sns.heatmap(
    corr, 
    vmin = -1, vmax = 1, center= 0, 
    cmap= sns.light_palette("purple"),
    square = True, 
    mask= dropself
)

ax.set_title ("Correlation Heatmap", fontsize = 20)

ax.set_yticklabels (
    ax.get_yticklabels(), 
    fontsize = 12
)

ax.set_xticklabels (
    ax.get_xticklabels(),
    rotation = 45,
    horizontalalignment = "right",
    fontsize = 12
)

plt.show()


#correlation heatmap with correlation values included 

def halfheatmap (df, mirrow, title ): 
    corr = df.corr()
    fig, ax = plt.subplots(figsize = (10,10))
    cmap= sns.light_palette("purple")
    ax.set_title(title, fontsize = 20)
    
    if mirrow == True:
        #Generate Heat Map, allow annotations and place floats in map
        sns.heatmap(corr, cmap=cmap, annot=True, fmt=".2f")
      #Apply xticks
        plt.xticks(range(len(corr.columns)), corr.columns);
      #Apply yticks
        plt.yticks(range(len(corr.columns)), corr.columns)
      #show plot
    
    else: 
        #drop selcorrelation
        dropself= np.zeros_like (corr)
        dropself [np.triu_indices_from(dropself)] = True
        colormap = sns.diverging_palette (200,10,as_cmap = True)
        sns.heatmap(corr, cmap=cmap, annot=True, fmt=".2f", mask=dropself)
      # Apply xticks
        plt.xticks(range(len(corr.columns)), corr.columns, rotation = 45, fontsize= 12);
      # Apply yticks
        plt.yticks(range(len(corr.columns)), corr.columns, fontsize= 12)
        
   # show plot
    plt.show()
    
halfheatmap(df = happiness, mirrow = False, title = "Correlation Heatmap")


#correlation heatmap with other colorcode

def halfheatmap (df, mirrow, title ): 
    corr = df.corr()
    fig, ax = plt.subplots(figsize = (10,10))
    colormap = sns.diverging_palette (220,10, as_cmap = True)
    ax.set_title(title, fontsize = 20)
    
    if mirrow == True:
        #Generate Heat Map, allow annotations and place floats in map
        sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2f")
      #Apply xticks
        plt.xticks(range(len(corr.columns)), corr.columns);
      #Apply yticks
        plt.yticks(range(len(corr.columns)), corr.columns)
      #show plot
    
    else: 
        #drop selcorrelation
        dropself= np.zeros_like (corr)
        dropself [np.triu_indices_from(dropself)] = True
        colormap = sns.diverging_palette (200,10,as_cmap = True)
        sns.heatmap(corr, cmap=colormap, annot=True, fmt=".2g", mask=dropself)
         # Apply xticks
        plt.xticks(range(len(corr.columns)), corr.columns, rotation = 45);
        # Apply yticks
        plt.yticks(range(len(corr.columns)), corr.columns)
        
    # show plot and save it 
    plt.savefig("Happiness.Correlation.Heatmap.png")
    plt.show()
    
halfheatmap(df = happiness, mirrow = False, title = "Correlation Heatmap")
#overview:
sns.pairplot(happiness)
plt.show()
happiness.info()

#we make the analysis with statsmodels.api as sm 

import statsmodels.api as sm 

x = happiness.iloc [:,[0,5,6,7,8,9,10]]

y = happiness["Happiness Score"]

X = sm.add_constant(x)
model = sm.OLS(y, X)
est = model.fit()

print(est.summary())
happiness_dummy = pd.get_dummies (happiness, columns = ["Region"])
happiness_dummy.head()
happiness_dummy.info()

import statsmodels.api as sm 

x = happiness_dummy.iloc [:,[0,4,5,6,7,8,9,10,11,12,13,14,15,16]]

y = happiness_dummy["Happiness Score"]

X = sm.add_constant(x)
model = sm.OLS(y, X)
est = model.fit()

print(est.summary())
#first sorting the scores based on the regions! 

grouped_happiness = happiness.groupby (["Region"])[["Happiness Score", "Year"]].aggregate(np.median).reset_index().sort_values ("Happiness Score")
grouped_happiness.info()
#plotting hte results (x and y are given like that to better allows to read the ocuntries)
chart = sns.barplot (x= grouped_happiness ["Happiness Score"],y = grouped_happiness ["Region"],saturation = 1.2, palette = "vlag"  )

chart.figure.set_size_inches (10,10)
plt.title (label = "Happiness Score by Region \n(2015 - 2019)", fontsize = 20)
plt.show()
#plotting hte results (x and y are given like that to better allows to read the ocuntries)
plt.figure (figsize=(10,10))
chart = sns.barplot (x= happiness ["Happiness Score"],y = happiness ["Region"], palette = ("BrBG"), hue=happiness ["Year"], saturation = 1.2)
                 
plt.title (label = "Happiness Score by Region \n(2015 - 2019)", fontsize = 20)
plt.show()
 
plt.figure (figsize=(10,10))

chart = sns.distplot (a= happiness ["Happiness Score"],bins= happiness ["Year"], color = "Red")
                 
plt.title (label = "Happiness Score distribution \n(2015 - 2019)", fontsize = 20)
plt.show()
plt.figure(figsize=(15,8))

sns.scatterplot(x='Happiness Score', y='Economy (GDP per Capita)', sizes = (("Economy (GDP per Capita)")*100), hue='Region',data=happiness)
plt.title ("Relationship Happiness Score and GDP per Capita \n(divided by Regions between 2015-2019)", fontsize = 20)
plt.xlabel('Happiness Score',size=12)
plt.ylim(0,2.5)
plt.xlim (2,8)
plt.ylabel('Economy (GDP per Capita)', size =12)
plt.show()
plt.figure(figsize=(12,10))
sns.scatterplot(x='Happiness Score', y='Economy (GDP per Capita)',data=happiness, sizes = (("Economy (GDP per Capita)")*100), alpha =0.8, color = "lime")

plt.title ("Relationship Happiness Score and GDP per Capita \n(between 2015-2019)", fontsize = 20)

plt.xlabel('Happiness Score',size=12)
plt.ylim(0,2.5)
plt.xlim (2,8)
plt.ylabel('Economy (GDP per Capita)', size =12)
plt.figure(figsize=(15,8))
plt.show()


plt.figure(figsize=(12,8))
sns.regplot (happiness ["Happiness Score"], happiness ["Economy (GDP per Capita)"], x_estimator=np.mean, ci = 80, color = "darkred")           
plt.title ("Relationship Happiness Score and GDP per Capita \n(between 2015-2019)", fontsize = 20)

plt.xlabel('Happiness Score',size=12)
plt.ylim(0,2.5)
plt.xlim (2,8)
plt.ylabel('Economy (GDP per Capita)', size =12)
plt.show()
#for the following we need at least version 3.4+ (so we have a look)

import sys 

print (sys.version)
#import additional libaries 

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score 
#definition of dependend (y= target) and indipendend variables (xns)
#note: we are not including everyhting (only the dummy and numerical values)

X= happiness_dummy.iloc [:,[0,4,5,6,7,8,9,10,11,12,13,15,16]]
y = happiness_dummy.iloc [:, 3] #target, what we are interested in
X.head() #check if it worked 
#establish training and test sets 

X_train, X_test, y_train, y_test = train_test_split (X,y, test_size =0.2)
#create linear regression

reg = LinearRegression()

reg.fit(X_train, y_train)
#predict the test data (80% of dataset)
y_pred = reg.predict (X_test)
#comute the root mean squared error (RMSE)
rmse = np.sqrt (mean_squared_error(y_test, y_pred))
print ("RMSE: {}". format(rmse))

#note between 0.45 and 0.52 that is a very high fluctratin rate! 
def regression_mode_cv( model, k = 8): 
    scores = cross_val_score (model, X, y, scoring = "neg_mean_squared_error", cv = k)
    
    rmse= np.sqrt(-scores)
    print ("Reg rmse:", rmse)
    print("Reg mean: ", rmse.mean())
    print("Reg mean:", rmse.mean())
#prints the 8 rsme and then the mean
regression_mode_cv(LinearRegression())
    
img= mpimg.imread("../input/overunderfitting/overfitting underfitting.png")
plt.figure(figsize=(15,11)) #make picture larger 
imgplot = plt.imshow(img) 
plt.axis ("off") #remove the axis and tickers 
plt.show()
#Ridge model

from sklearn.linear_model import Ridge
regression_mode_cv(Ridge())
#Lasso model 

from sklearn.linear_model import Lasso
regression_mode_cv(Lasso())
#barplot for economy 
figure = sns.violinplot(happiness_dummy["Year"], happiness_dummy["Happiness Score"])
figure.figure.set_size_inches (10,7)
#plt.savefig("Age_Attrition.png")
plt.show()

plt.title ("Infuence of time on Happiness Score", fontsize= 20)

figure = sns.regplot(happiness_dummy["Year"], happiness_dummy["Happiness Score"], x_estimator=np.mean, x_bins = 15, color = "darkred", ci = 90)
figure.figure.set_size_inches (10,7)
plt.xlabel("Year", fontsize = 15)
plt.ylabel("Happiness Score", fontsize = 15)
plt.show()
plt.title ("Infuence of Freedom on Happiness Score", fontsize= 20)

figure = sns.regplot(happiness_dummy["Freedom" ], happiness_dummy["Happiness Score"], x_estimator=np.mean, x_bins = 15, color = "darkred", ci = 90)
figure.figure.set_size_inches (10,7)
plt.xlabel("Freedom Score", fontsize = 15)
plt.ylabel("Happiness Score", fontsize = 15)
plt.show()
figure = sns.scatterplot(happiness_dummy["Family"],happiness_dummy["Happiness Score"], color = "darkred")

figure.figure.set_size_inches (10,7)

plt.show()
figure = sns.lineplot(happiness_dummy["Family"],happiness_dummy["Happiness Score"], color = "navy")

figure.figure.set_size_inches (10,7)

plt.show()
