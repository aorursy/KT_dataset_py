# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
restaurants=pd.read_csv('../input/zomato.csv',encoding = "ISO-8859-1")
countrycode=pd.read_excel('../input/Country-Code.xlsx')
restaurants.head()
restaurants.columns
countrycode
restaurants['Average Cost for two']=restaurants['Average Cost for two'].replace(0,np.nan)
restaurants.dropna(inplace=True)
restaurants=restaurants.merge(countrycode,on='Country Code')
restaurants.head()
indiaRes=restaurants[restaurants['Country']=='India']
singaporeRes=restaurants[restaurants['Country']=='Singapore']
AusRes=restaurants[restaurants['Country']=='Australia']
BrazRes=restaurants[restaurants['Country']=='Brazil']
CanadaRes=restaurants[restaurants['Country']=='Canada']
UAERes=restaurants[restaurants['Country']=='UAE']
UKRes=restaurants[restaurants['Country']=='United Kingdom']
USRes=restaurants[restaurants['Country']=='United States']
indiaRes.head()
indiaRes.City.value_counts()
DelhiTopRes=indiaRes[(indiaRes['City']=='New Delhi')&(indiaRes['Aggregate rating']>4.5)][['Aggregate rating','Restaurant Name','Votes']].sort_values(ascending=False,by='Votes')
DelhiTopRes.style.apply(lambda x: ['background: green' if x.name == 'Votes' else 'background: lightsteelblue' for i in x])
sns.jointplot('Votes','Aggregate rating',data=DelhiTopRes)
India=indiaRes['Restaurant Name'].nunique()
India
DelhiTopResOnline=indiaRes[(indiaRes['City']=='New Delhi')&(indiaRes['Aggregate rating']>4.5) &(indiaRes['Has Online delivery']=='Yes')][['Aggregate rating','Restaurant Name','Votes','Has Online delivery']].sort_values(ascending=False,by='Votes')
DelhiTopResOnline.style.apply(lambda x: ['background: green' if x.name == 'Has Online delivery' else 'background: Royalblue' for i in x])
sns.set(rc={'figure.figsize':(20,11)})
plt.xticks(rotation = 90)
sns.countplot(restaurants['Country'],hue=restaurants['Has Online delivery'])
Chennai=indiaRes[(indiaRes['City']=='Chennai')&(indiaRes['Aggregate rating']>4.5)&(indiaRes['Votes']>500)][['Average Cost for two','Restaurant Name','Aggregate rating','Votes']].sort_values(ascending=True,by=['Average Cost for two']).head()
Chennai.style.apply(lambda x: ['background: darkorange' if x.name == 'Average Cost for two' else 'background: lightsteelblue' for i in x])
IndianCuisines=restaurants[restaurants['Country']=='India']['Cuisines'].value_counts().head()
IndianCuisines
table=pd.pivot_table(data=restaurants, index = ('Country', 'City'), values="Aggregate rating")
cm = sns.light_palette("green", as_cmap=True)
table.style.background_gradient(cmap=cm,axis=0)
RestaurantratingEachCountry=pd.pivot_table(data=restaurants, index = ('Country', 'Rating text'), values=("Restaurant ID"),aggfunc="count")
RestaurantGroup=restaurants.groupby(by="Country")['Restaurant ID'].count()
RestaurantGroup.columns=['No of restaurants']
TotalRatingCountry=RestaurantGroup.groupby(by='Country').sum()
TotalRatingCountry.reset_index()
FinalRestPerc=pd.merge(TotalRatingCountry.reset_index(),RestaurantratingEachCountry.reset_index(),on='Country')
FinalRestPerc
FinalRestPerc['Percentage']=(FinalRestPerc['Restaurant ID_y']/FinalRestPerc['Restaurant ID_x'])*100
FinalRestPerc
sns.set(rc={'figure.figsize':(20,11)})
sns.barplot('Country', 'Percentage', data=FinalRestPerc, hue = 'Rating text')
plt.xticks(rotation = 90)
avgcostforindres=indiaRes['Average Cost for two']
aggratingofindres=pd.Categorical(values=indiaRes['Rating text'],categories=["Excellent", "Very Good", "Good", "Average", "Poor", "Not rated"], ordered=True)
sns.boxplot(aggratingofindres,avgcostforindres)
plt.figure(figsize=(10,10))
indiaRes['City'].value_counts().head().plot(kind='pie',autopct='%1.1f%%')
booking_avail=indiaRes[(indiaRes['City']=='New Delhi')&(indiaRes['Aggregate rating']>4.5) &(indiaRes['Has Table booking']=='Yes')][['Aggregate rating','Restaurant Name','Votes','Has Table booking']].sort_values(ascending=False,by='Votes')
booking_avail.style.apply(lambda x: ['background: darkorange' if x.name == 'Has Table booking' else 'background: lightsteelblue' for i in x])
plt.figure(figsize=(10,10))
indiaRes['Cuisines'].value_counts().head(10).plot(kind='pie',autopct='%1.1f%%')
