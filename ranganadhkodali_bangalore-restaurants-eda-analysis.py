# basic libraries used



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



pd.options.display.max_columns = None



# code presenting customer functions



from IPython.display import Markdown, display

def printmd(string):

    display(Markdown(string))

    

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



#reading Data 

data = pd.read_csv("../input/zomato.csv")



printmd("### initial look into data")



print(data.info())



data.head()
#reorganizing items in menu items to list



data["menu_item"]=data["menu_item"].str.strip("'[]'").str.split()



# Split the rates to the seperate

data["rate"]=data["rate"].str.split("/")

#data["rate"] = data.rate.str[0]



# Reorgnizing Rating field



data["rating"]=pd.to_numeric(data.rate.str[0],errors='coerce') 

data["rating_outof"]=pd.to_numeric(data.rate.str[1],errors='coerce') 



#Deleting unwanted columns 

data.drop(["address","url","phone"],axis=1,inplace=True)



#reattanging cost field 

data["costfortwo"]=data["approx_cost(for two people)"].str.replace(",",'').astype(float)
## Missing values analysis



_missing_counts=pd.DataFrame(round(data.isna().sum() / data.shape[0] ,2))



_missing_counts.columns =["Count"]

_missing_counts=_missing_counts.reset_index()



_missing_counts.sort_values(by="Count",ascending = False)
# from the rating understanded that if rating is like NEW can be considered as NEW or else existing restaurant



data["Category"] = np.where((data.rating.isna()==True) & (data.rate.isna()==False),"New","Existing")
data.groupby("name")["name"].count().sort_values(ascending =False)[:20]
# identified some estaurants indicating resturent chains. and these type of resturents represnts mostly same kind of standards hence. 



# hence for analysis on rating and votes we will prepare another data set by deleting all resturents with keeping only one Restaurants of any specfic chain



data_unique_Restaurants=(data.drop_duplicates(subset=["name"],keep="first")).copy()



data_unique_Restaurants.shape
# analysing Category of restaurants 



def dis_pie(field,num):

    #plt.subplot(num)

    _temp = pd.DataFrame(data.groupby(field)[field].count())

    #plt.title(field+" Descriptions")

    #plt.pie(_temp,labels = _temp.index,autopct='%1.1f%%', shadow=True, startangle=90)

    print(_temp)

    

plt.figure(figsize=(10,10))

dis_pie("Category",221)

dis_pie("online_order",222)

dis_pie("book_table",223)

plt.show()



def dis_barChart(field):

    plt.figure(figsize=(20,10))

    _temp = pd.DataFrame(data.groupby(field)[field].count())

    _temp.columns =["count"]

    _temp=_temp.reset_index()

    _temp.columns =[field,"count"]  

    _temp=_temp.sort_values(by="count",ascending =False)

    plt.title(field+" Descriptions")

    g=sns.barplot(data =_temp,x=field,y="count")

    plt.xticks(rotation=90)

    plt.show()

    

    

dis_barChart("listed_in(city)")

dis_barChart("rest_type")

dis_barChart('listed_in(type)')



dis_barChart('rating')

#_temp = pd.DataFrame(data.groupby())



#Derived columns for cuisines



#covert list variables as list

data_unique_Restaurants["cuisines_ls"]=data_unique_Restaurants.cuisines.str.split(",")



#find out distient elements of cuisines

_temp=data_unique_Restaurants['cuisines_ls'].apply(pd.Series)

for i in _temp.columns:

    _temp[i] = _temp[i].str.strip().str.lower()    

    

data_unique_Restaurants["cuisines_ls1"]=_temp.apply(lambda x:list(x),axis=1)

list_of_cuisines = _temp.stack().value_counts()
for i in list_of_cuisines[:10].index:

    data_unique_Restaurants[i] =data_unique_Restaurants.cuisines_ls1.astype(str).str.contains(i)

#display Rare cuisines surving restaruents

rare_cuisines = list_of_cuisines[-40:-1]

rare_cuisines
# identified restaurants serving where rare foods



data_unique_Restaurants["rare_cuisines"] =data_unique_Restaurants.cuisines_ls1.astype(str).str.contains("hot dogs")



for i in rare_cuisines.index:

    data_unique_Restaurants["rare_cuisines"] =(data_unique_Restaurants.cuisines_ls1.astype(str).str.contains(i)) | (data_unique_Restaurants["rare_cuisines"])
# derived colum defining type of cuisines surved by the restaurant

data_unique_Restaurants["number_of_different_cuisines"] = data_unique_Restaurants.cuisines_ls.str.len()
# attemped to identify how many of Restaurants serving each popular type of cuisnies + rare_cuisines



_temp=pd.DataFrame(data_unique_Restaurants[list_of_cuisines[:10].index.tolist() +["rare_cuisines"]].stack())

_temp=_temp.reset_index(level=1)

_temp.columns=["name","values"]





(_temp.groupby(["name","values"])["values"].count() / data_unique_Restaurants.shape[0]).unstack().plot(kind ="bar",stacked=True,figsize=(20,5),title="restaurants serving specific cuisine comparing with number of restaurants")

plt.show()
# multi cuisine Restaurants



_temp =data_unique_Restaurants.groupby('number_of_different_cuisines')["number_of_different_cuisines"].count() / data_unique_Restaurants.shape[0]



_temp =pd.DataFrame(_temp)

_temp.columns =["Per"]

_temp=_temp.reset_index()

_temp.columns =["noof_multi_cuisine","Per"]



plt.figure(figsize=(10,5))

ax=sns.barplot(x="noof_multi_cuisine",y="Per",data=_temp)

plt.title("Multi Cuisine Resturents in Banglore")



for index, row in _temp.iterrows():

    ax.text(index ,row.Per+0.001,str(round(row.Per,4)*100)+"%", color='black', ha="center")

    

    

plt.show()
## Review number of votes 



#data.votes.describe()

data_unique_Restaurants["votes_range"]=pd.cut(data_unique_Restaurants.votes,[0,2,5,7,10,15,20,50,100,200,1000,10000])



_temp =data_unique_Restaurants.groupby("votes_range")["name"].count() / data_unique_Restaurants.shape[0]

_temp=pd.DataFrame(_temp)

_temp=_temp.reset_index()

_temp.columns=["votes_range","Per"]



plt.figure(figsize=(20,5))

ax=sns.barplot(x="votes_range",y="Per",data =_temp)

plt.title("Resturent by votes")



for index, row in _temp.iterrows():

    ax.text(index ,row.Per+0.001,str(round(row.Per,4)*100)+"%", color='black', ha="center")

    

plt.show()

##data.head()
#_temp=data.pivot_table(index="votes_range",columns=["north indian"],values=["north indian"],aggfunc="count")



#_temp=_temp.reset_index(level =0)

_temp=pd.concat([data_unique_Restaurants[list_of_cuisines[:10].index.tolist() +["rare_cuisines"]].astype(int) , data_unique_Restaurants["votes_range"]],axis=1)



_temp=_temp.groupby("votes_range").sum() 

#_temp=_temp.reset_index()

plt.figure(figsize=(20,7))

plt.subplot(1,2,1)

plt.title("Votes by cusine type for all resturents (# Count)")

sns.heatmap(_temp,cmap="YlGnBu",vmin=0.4)

#plt.show()



_temp=_temp.apply(lambda x: x/sum(x),axis=0)



#_temp=_temp.reset_index()

#plt.figure(figsize=(10,7))

plt.subplot(1,2,2)

plt.title("% Votes by cusine type on the same segment of serving cuisine")

sns.heatmap(_temp,cmap="YlGnBu",vmin=0.01,annot=True)

plt.show()

#data[list_of_cuisines[:10].index.tolist() +["rare_cuisines"]].apply(lambda x: 1 if x else 0,axis=0)
#_temp=data.pivot_table(index="votes_range",columns=["north indian"],values=["north indian"],aggfunc="count")



#_temp=_temp.reset_index(level =0)

_temp=pd.concat([data_unique_Restaurants[list_of_cuisines[:10].index.tolist() +["rare_cuisines"]].astype(int) , data_unique_Restaurants["rating"]],axis=1)



_temp=_temp.groupby("rating").sum() 

#_temp=_temp.reset_index()

plt.figure(figsize=(20,7))

plt.subplot(1,2,1)

plt.title("Votes by cusine type for all resturents (# count)")

sns.heatmap(_temp,cmap="YlGnBu",vmin=0.4)



_temp=_temp.apply(lambda x: x/sum(x),axis=0)



#_temp=_temp.reset_index()

plt.subplot(1,2,2)

plt.title("Votes by cusine type on the same segment of serving cuisine (%)")

sns.heatmap(_temp,cmap="YlGnBu",vmin=0.01)

plt.show()

#data[list_of_cuisines[:10].index.tolist() +["rare_cuisines"]].apply(lambda x: 1 if x else 0,axis=0)
_temp=data_unique_Restaurants.loc[ data_unique_Restaurants["costfortwo"].isnull() ==False ,["votes_range","costfortwo","rating"]]



#_temp["costfortwo"]=_temp["approx_cost(for two people)"].str.replace(',', '').astype(float) 



plt.figure(figsize=(20,5))

plt.title("Number of votes vs cost for two")

sns.boxplot(data =_temp,x="votes_range",y="costfortwo")

plt.show()



plt.figure(figsize=(20,5))

plt.title("Rating vs cost for two")

sns.boxplot(data =_temp,x="rating",y="costfortwo")

plt.show()



#_temp["approx_cost(for two people)"]=_temp["approx_cost(for two people)"].str.replace(',', '').astype(float) 



#_temp[_temp["approx_cost(for two people)"].isnull()]



#data.iloc[1662]
# online orders vs votes

data_unique_Restaurants.groupby(["votes_range","online_order"])["name"].count().unstack().plot(kind="bar",stacked=True,figsize=(15,5),title="Option of Online Order vs Votes")

plt.show()



#online order vs rating

data_unique_Restaurants.groupby(["rating","online_order"])["name"].count().unstack().plot(kind="bar",stacked=True,figsize=(15,5),title="Option of Online Order vs rating")

plt.show()



# online orders vs votes

data_unique_Restaurants.groupby(["votes_range","book_table"])["name"].count().unstack().plot(kind="bar",stacked=True,figsize=(15,5),title="Option of Online Order vs Votes")

plt.show()



#online order vs rating

data_unique_Restaurants.groupby(["rating","book_table"])["name"].count().unstack().plot(kind="bar",stacked=True,figsize=(15,5),title="Option of Online Order vs rating")

plt.show()

_temp=data_unique_Restaurants.loc[ data_unique_Restaurants["approx_cost(for two people)"].isnull() ==False ,["book_table","approx_cost(for two people)","online_order"]]

_temp["costfortwo"]=_temp["approx_cost(for two people)"].str.replace(',', '').astype(float) 





plt.figure(figsize=(20,5))

plt.subplot(1,2,1)

plt.title("Booking Table option vs cost for two")

sns.boxplot(data =_temp,x="book_table",y="costfortwo")





plt.subplot(1,2,2)

plt.title("Online Order option vs cost for two")

sns.boxplot(data =_temp,x="online_order",y="costfortwo")

plt.show()


# Top of Restaurants of choice can be tried



data_unique_Restaurants.loc[(data_unique_Restaurants.rating>4.5) &(data_unique_Restaurants.

                                                                 costfortwo <=800),["name","cuisines","listed_in(city)","listed_in(type)","votes","rating",

                                                                                    "costfortwo"]].sort_values(by="costfortwo")








