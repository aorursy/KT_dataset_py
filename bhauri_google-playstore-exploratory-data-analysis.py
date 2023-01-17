# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from IPython.display import display
pd.options.display.max_rows=None
from scipy import stats
from statsmodels.formula.api import ols 
from IPython.display import display, Markdown
%matplotlib inline
playstore=pd.read_csv("../input/googleplaystore.csv")
display(playstore.info())
display(playstore.head())
playstore.drop(index=10472,inplace=True)
def change_size(size):
    kb= size.str.endswith("k")
    MB=  size.str.endswith("M")
    other= ~(kb|MB)
    size.loc[kb]= size[kb].str.replace("k","").astype("float")/1024
    size.loc[MB]= size[MB].str.replace("M","").astype("float")
    size.loc[other] = float(0.0)
change_size(playstore.Size)
playstore.columns= [x.replace(" ","_") for x in  playstore.columns]
playstore.Installs= np.log(playstore.Installs.str.replace("[+,]","").astype("int64")+1)
playstore.Reviews= np.log(playstore.Reviews.astype("int")+1)
playstore.Price= playstore.Price.str.replace("[$,]","").astype("float")
playstore.Size=playstore.Size.astype("float")
#playstore.Type= pd.get_dummies(playstore.Type,drop_first=True)
playstore.info()
total = playstore.isnull().sum().sort_values(ascending=False)
percent = (playstore.isnull().sum()/playstore.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
display(missing_data.head(6))
print("Before Cleaning")
display(playstore.shape)
playstore.Rating.fillna(method='ffill',inplace=True)
playstore.dropna(how ='any', inplace = True)
print("After Cleaning")
display(playstore.shape)
sns.pairplot(playstore,kind="scatter",hue='Type')
playstore.Category.value_counts().plot(kind='bar',figsize=(18,6))
playstore.groupby("Type")["Type"].value_counts().plot(kind='pie',autopct='%1.1f%%' );
sns.catplot(x="Type",y="Installs",kind='box',data=playstore,  height=4, aspect=2/1);
sns.catplot(x="Type",y="Rating",kind='box',data=playstore,  height=4, aspect=2/1);
#null Hypothisis- avg rating are same for paid and free app
model_name = ols('Rating ~ C(Type)', data=playstore).fit()
model_name.summary()
sns.catplot(x="Type",y="Size",kind='box',data=playstore, height=4, aspect=2/1);
sns.catplot("Price","Category",data=playstore[playstore.Price>0],height=10,aspect=1.5,hue="Content_Rating",s=10,alpha=.3);
playstore[playstore.Price>200][["Category","App","Price"]]
sns.catplot("Price","Category",data=playstore[playstore.Price>0],height=10,aspect=1.5,hue="Content_Rating",s=10,alpha=.3)
playstore[playstore.Price>0].groupby("Content_Rating")["App"].count().plot(kind='bar')
sns.catplot("Size","Category",data=playstore,height=10,aspect=2/1,c=1/1000,s=10,alpha=0.2,hue="Content_Rating")
sns.catplot(x="Category",y="Rating",kind='box',data=playstore, height=8, aspect=2/1);
plt.xticks(rotation=90);
model_name = ols('Rating ~ C(Category)', data=playstore).fit()
model_name.summary()
sns.catplot(x="Category",y="Installs",kind='violin',data=playstore, height=8, aspect=2/1);
sns.lineplot(x=range(0,len(playstore.Category.unique())),y=playstore["Installs"].mean(),)
plt.xticks(rotation=90);
playstore.Genres.value_counts().head(30).plot("bar",figsize=(18,6))
sns.regplot(playstore.Rating,playstore.Reviews,color="g",x_estimator=np.mean);
app_reviews= pd.read_csv("../input/googleplaystore_user_reviews.csv")
display(app_reviews.head())
display(app_reviews.info())
app_reviews.isnull().sum().sort_values()
app_reviews.dropna(inplace=True)
app_reviews.isnull().sum().sort_values()
app_reviews.Sentiment.value_counts()
combined_data= playstore[["App","Type","Category","Genres","Content_Rating"]].merge(app_reviews,how="inner",left_on="App",right_on="App")
combined_data.head()                                                                
temp_type=(combined_data.groupby(["Type","Sentiment"])["App"].count()/combined_data.groupby(
    ["Type"])["App"].count()).reset_index(level=[0,1])

greenBars= temp_type[temp_type.Sentiment=='Positive']["App"]
orangeBars = temp_type[temp_type.Sentiment=='Negative']["App"]
blueBars = temp_type[temp_type.Sentiment=='Neutral']["App"]
r= list(range(0,len(temp_type.Type.unique())))

barWidth = 0.85
names = temp_type.Type.unique()
# Create green Bars
plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth)
# Create orange Bars
plt.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth)
# Create blue Bars
plt.bar(r, blueBars, bottom=[i+j for i,j in zip(greenBars, orangeBars)], color='#a3acff', edgecolor='white', width=barWidth)
 
# Custom x axis
plt.xticks(r, names,rotation=90)
plt.xlabel("group")
#plt.legend() 
# Show graphic
plt.show()
temp_cat_paid=(combined_data[combined_data.Type=="Paid"].groupby(["Category","Sentiment"])["App"].count()/combined_data[combined_data.Type=="Paid"].groupby(
    ["Category"])["App"].count()).reset_index(level=[0,1])

greenBars= temp_cat_paid[temp_cat_paid.Sentiment=='Positive']["App"]
orangeBars = temp_cat_paid[temp_cat_paid.Sentiment=='Negative']["App"]
blueBars = temp_cat_paid[temp_cat_paid.Sentiment=='Neutral']["App"]
plt.figure(figsize=(15,8))

r= list(range(0,len(temp_cat_paid.Category.unique())))

barWidth = 0.85
names = temp_cat_paid.Category.unique()
# Create green Bars
plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth)
# Create orange Bars
plt.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth)
# Create blue Bars
plt.bar(r, blueBars, bottom=[i+j for i,j in zip(greenBars, orangeBars)], 
        color='#a3acff', edgecolor='white', width=barWidth)
plt.axhline(y=.75,color='r', linestyle='-')
# Custom x axis
plt.xticks(r, names,rotation=90)
plt.xlabel("group")
#plt.legend() 
# Show graphic
plt.show()


temp_cat_free=(combined_data[combined_data.Type=="Free"].groupby(["Category","Sentiment"])["App"].count()/combined_data[combined_data.Type=="Free"].groupby(
    ["Category"])["App"].count()).reset_index(level=[0,1])

greenBars= temp_cat_free[temp_cat_free.Sentiment=='Positive']["App"]
orangeBars = temp_cat_free[temp_cat_free.Sentiment=='Negative']["App"]
blueBars = temp_cat_free[temp_cat_free.Sentiment=='Neutral']["App"]

plt.figure(figsize=(15,8))

r= list(range(0,len(temp_cat_free.Category.unique())))

barWidth = 0.85
names = temp_cat_free.Category.unique()
# Create green Bars
plt.bar(r, greenBars, color='#b5ffb9', edgecolor='white', width=barWidth)
# Create orange Bars
plt.bar(r, orangeBars, bottom=greenBars, color='#f9bc86', edgecolor='white', width=barWidth)
# Create blue Bars
plt.bar(r, blueBars, bottom=[i+j for i,j in zip(greenBars, orangeBars)], 
        color='#a3acff', edgecolor='white', width=barWidth)
plt.axhline(y=.75,color='r', linestyle='-')
# Custom x axis
plt.xticks(r, names,rotation=90)
plt.xlabel("group")
#plt.legend() 
# Show graphic
plt.show()


sns.catplot("Type","Sentiment_Polarity",data= combined_data,alpha=.3);
sns.catplot("Category","Sentiment_Polarity",data= 
            combined_data[combined_data.Type=="Free"],alpha=.2,height=8,aspect=1.5);
plt.xticks(rotation=90);

