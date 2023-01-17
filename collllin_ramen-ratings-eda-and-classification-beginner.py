#Import files and packages

import os

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
#Load Data 

ramen_data=pd.read_csv('../input/ramen-ratings/ramen-ratings.csv')
#Data exploration

ramen_data.describe(include='all')
#Noticed that style has 2 missing values

ramen_data.info()
#Previewing the data

ramen_data.head()
#Replaced the missing Ramen Styles with None

ramen_data['Style']=ramen_data['Style'].fillna('None')
#Reordered dataframe by Brands into a descending order

ramen_data.sort_values(by='Brand', ascending=True)
#Noticed that some Brands were not spelled in the same manner. 

#ex.) Goku Uma -> Goku Uma

ramen_data_brand=ramen_data['Brand'].unique()

print(sorted((ramen_data_brand)))
#Fixed the naming of brands

ramen_data.loc[ramen_data['Brand'] == 'Goku-Uma', 'Brand'] ='Goku Uma'

ramen_data.loc[ramen_data['Brand'] == 'Lishan', 'Brand'] ='Lishan Food Manufacturing'

ramen_data.loc[ramen_data['Brand'] == 'MAMA', 'Brand'] ='Mama'

ramen_data.loc[ramen_data['Brand'] == 'Prima', 'Brand'] ='Prima Taste'

ramen_data.loc[ramen_data['Brand'] == 'Sakurai', 'Brand'] ='Sakurai Foods'

ramen_data.loc[ramen_data['Brand'] == 'Samyang', 'Brand'] ='Samyang Foods'

ramen_data.loc[ramen_data['Brand'] == 'Unif Tung-I', 'Brand'] ='Unif / Tung-I'

ramen_data.loc[ramen_data['Brand'] == 'Wu-Mu', 'Brand'] ='Wu Mu'
#Noticed that there were Unrated Ramens, gave them a rating of 0

ramen_data['Stars'].unique()

#Investigated the Top Ten Column



ramen_data_copy=ramen_data

ramen_data_copy['Top Ten'].unique()

#Identified the values within the Top Ten Column, most of the rated ramen were not classified as a Top Ten

#created a new column to indicate which ones qualified in the Top Tens.

ramen_data.loc[(ramen_data['Top Ten'].notna()), 'Top_Ten'] = 1

ramen_data.loc[(ramen_data['Top Ten'].isna()), 'Top_Ten'] = 0

ramen_data_copy['Top_Ten'].value_counts()
#Converted the 'Stars' column to numeric

ramen_data.loc[ramen_data['Stars'] == 'Unrated', 'Stars'] = '0'

ramen_data['Stars'] = pd.to_numeric(ramen_data['Stars'])

ramen_data['Stars'].describe()
#Found out the distribution of total ramen ratings in each bin.

#appears that 29.36% are rated between 3.5~4.

#appears that 20% are rated between 3.0~3.5.

#appears that 17.64% are rated between 4.5~5.

#If we assume that all ramen within the 2.5~3.0 star rating are okay, that would mean at 67% of the rated ramen are

#above average.

count_of_stars= pd.cut(ramen_data['Stars'],bins=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5])

count_of_stars.value_counts(normalize=True)
#See the distribution of the Ramen Ratings.



plt.grid(False)

plt.style.use('tableau-colorblind10')

plt.hist(ramen_data['Stars'],bins=[0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5], color='limegreen', edgecolor='black')

plt.xlabel('Ramen Ratings')

plt.ylabel('Total Count')

plt.title('Distribution of Ramen Ratings')

plt.show()
#Wanted to see which brands are the top 10 producers of ramen. Nissin is the biggest producer of ramen, 

#followed by Mama & Nongshim

ramen_brand_counts=ramen_data['Brand'].value_counts().head(10)

labels = ['Nissin','Mama','Nongshim','Maruchan','Samyang Foods','Paldo','Myojo','Indomie','Ottogi','Lucky Me!']





plt.figure(figsize=(15,5))

plt.bar(labels,ramen_brand_counts,color='limegreen',edgecolor='black')

plt.grid(False)

plt.xlabel('Brand')

plt.ylabel('Total Products')

plt.title('Total Ramen Products By Brand')



plt.show()



#Wanted to the average rating for the top 10 Ramen producers.

#The top 8 ramen producing companies have an average ramen rating of 3.5+.

#Also Nissin, does not have the highest average ramen rating, despite being the top producer.

labels = ['Nissin','Mama','Nongshim','Maruchan','Samyang Foods','Paldo','Myojo','Indomie','Ottogi','Lucky Me!']



Nissin = ramen_data.loc[ramen_data.Brand == 'Nissin']['Stars'].mean()

Mama = ramen_data.loc[ramen_data.Brand == 'Mama']['Stars'].mean()

Nongshim = ramen_data.loc[ramen_data.Brand == 'Nongshim']['Stars'].mean()

Maruchan = ramen_data.loc[ramen_data.Brand == 'Maruchan']['Stars'].mean()

Samyang_Foods = ramen_data.loc[ramen_data.Brand == 'Samyang Foods']['Stars'].mean()

Paldo = ramen_data.loc[ramen_data.Brand == 'Paldo']['Stars'].mean()

Myojo = ramen_data.loc[ramen_data.Brand == 'Myojo']['Stars'].mean()

Indomie = ramen_data.loc[ramen_data.Brand == 'Indomie']['Stars'].mean()

Ottogi = ramen_data.loc[ramen_data.Brand == 'Ottogi']['Stars'].mean()

Lucky_Me = ramen_data.loc[ramen_data.Brand == 'Lucky Me!']['Stars'].mean()



plt.figure(figsize=(15,5))



plt.xlabel('Brands')

plt.ylabel('Average Ramen Rating')

plt.title('Average Ramen Ratings for the Top 10 Ramen Producing Brand')

plt.bar(labels,[Nissin,Mama,Nongshim,Maruchan,Samyang_Foods,Paldo,Myojo,Indomie,Ottogi,Lucky_Me],color='limegreen',

       edgecolor='black')

plt.grid(False)

plt.show()
#created a new dataframe that consists of only ramen that made it onto the yearly top-ten lists

# Made a barchart to see which Brands had the most products that made it on the top-ten lists

# The top three in a descending order: Prima Taste, Indomie and then Mama.

# One thing to take into consideration is that brands may produce ramen from different countries,

#this chart does not account for this fact.

ramen_data_only_tt=ramen_data[(ramen_data['Top_Ten']==1)]

ramen_data_only_tt['Brand'].unique()

labels =['Prima Taste', 'Indomie', 'Mama', 'Nongshim','Paldo','MyKuali','Sapporo Ichiban','Myojo','Mamee',

         'Wugudaochang','Doll','Nissin','Samyang Foods','CarJEN','A-Sha Dry Noodle','Tseng Noodles', 'Maruchan',

         'Mi Sedaap', 'Koka']

ramen_data_only_tt_counts=ramen_data_only_tt['Brand'].value_counts(sort=True)





plt.figure(figsize=(20,10))

plt.ylabel('Count of Ramen in the Top Ten Lists')

plt.xlabel('Brand')

plt.title('Brands and Ramen Products in the Top Ten Lists')

plt.xticks(rotation=90)

plt.bar(labels,ramen_data_only_tt_counts, color='limegreen',edgecolor='black')

plt.grid(False)

plt.show()

#Find average rating of these 3 Brands that are renowned for top-ten ramen.

#Surprisingly Prima Taste has the maximum rating of 5 for the average rating of its ramen.

#Indomie has an average rating of 4, while Mama has 3.7.

#Based on this information, I assume that Prima Taste is seeking to produce higher quality ramen instead of rushing

#out the total quantity of ramen products.

Prima_Taste=ramen_data.loc[ramen_data.Brand == 'Prima Taste']['Stars'].mean()

Indomie=ramen_data.loc[ramen_data.Brand == 'Indomie']['Stars'].mean()

Mama=ramen_data.loc[ramen_data.Brand == 'Mama']['Stars'].mean()





labels =['Prima Taste','Indomie','Mama']



plt.figure(figsize=(4,6))

plt.bar(labels,[Prima_Taste,Indomie,Mama], color='limegreen',edgecolor='black')

plt.grid(False)

plt.xlabel('Brand')

plt.ylabel('Average Rating of Ramen')

plt.title('Top Ten Ramen Producers and Average Rating of Ramen')





plt.show()
ramen_data.loc[ramen_data.Brand == 'Prima Taste']['Stars'].value_counts()
#Did a pie chart to see where which countries produced top ten ramens.

#Shows each country's count of top ten ramen and percentage that it contributes to the total.

#Surprising Singapore and South Korea are tied at first place with 7 top ten rated ramens.

#Japan, Malaysia, and Indonesia are tied for second place



#Count of top ten ramen created in each country

Myanmar=ramen_data_only_tt.loc[ramen_data_only_tt['Country']=='Myanmar'].count()[0]

Singapore=ramen_data_only_tt.loc[ramen_data_only_tt['Country']=='Singapore'].count()[0]

Taiwan=ramen_data_only_tt.loc[ramen_data_only_tt['Country']=='Taiwan'].count()[0]

China=ramen_data_only_tt.loc[ramen_data_only_tt['Country']=='China'].count()[0]

USA=ramen_data_only_tt.loc[ramen_data_only_tt['Country']=='USA'].count()[0]

Malaysia=ramen_data_only_tt.loc[ramen_data_only_tt['Country']=='Malaysia'].count()[0]

Japan=ramen_data_only_tt.loc[ramen_data_only_tt['Country']=='Japan'].count()[0]

Thailand=ramen_data_only_tt.loc[ramen_data_only_tt['Country']=='Thailand'].count()[0]

South_Korea=ramen_data_only_tt.loc[ramen_data_only_tt['Country']=='South Korea'].count()[0]

Indonesia=ramen_data_only_tt.loc[ramen_data_only_tt['Country']=='Indonesia'].count()[0]

Hong_Kong=ramen_data_only_tt.loc[ramen_data_only_tt['Country']=='Hong Kong'].count()[0]



values = [Myanmar,Singapore,Taiwan,China,USA,Malaysia,Japan,Thailand,South_Korea,Indonesia,Hong_Kong]

labels = ['Myanmar','USA','Singapore','Taiwan','Malaysia','China','Japan','Thailand','South Korea',

          'Hong Kong','Indonesia']

explode = (0.2,0,0.1,0.4,0.6,0,0,0,0,0,0.4)



def new_autopct(values):

    def percentage_autopct(pct):

        total = sum(values)

        val = int(round(pct*total/100.0))

        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)

    return percentage_autopct



colours = ['magenta','tab:orange','tab:green','tab:red','tab:purple','tab:cyan','tab:olive','gold','tomato',

         'aqua','lawngreen']



plt.figure(figsize=(10,10))

plt.pie([Myanmar,USA,Singapore,Taiwan,Malaysia,China,Japan,Thailand,South_Korea,Hong_Kong,Indonesia],labels=labels,

        autopct=new_autopct(values), pctdistance=0.75,colors=colours)

plt.title('Countries That Have Produced Top Ten Ramen')

plt.show()

#I wanted to see which styles of ramen are produced the most.

#Pack is the most popular, followed by Bowl and Cup.

ramen_data['Style'].value_counts(normalize=True)
#Decided to see if different styles of ramen affected the star rating.

# Pack, Bowl, Cup are the most produced styles and there doesn't appear to be much influence on the rating.

# Pack probably has more outliers than the other 2 styles because it is the most produced style.

ramen_data['Style'].unique()



plt.figure(figsize=(10,5))

labels =['Cup','Pack','Tray','Bowl','Box','Can','Bar','None']



Cup = ramen_data.loc[ramen_data.Style == 'Cup']['Stars']

Pack = ramen_data.loc[ramen_data.Style == 'Pack']['Stars']

Tray = ramen_data.loc[ramen_data.Style == 'Tray']['Stars']

Bowl = ramen_data.loc[ramen_data.Style == 'Bowl']['Stars']

Box = ramen_data.loc[ramen_data.Style == 'Box']['Stars']

Can = ramen_data.loc[ramen_data.Style == 'Can']['Stars']

Bar = ramen_data.loc[ramen_data.Style == 'Bar']['Stars']

NaN = ramen_data.loc[ramen_data.Style == 'None']['Stars']



outliers = dict(markerfacecolor='r',marker='o')

plt.title('Comparison of Ramen Production Styles and Ramen Ratings')

plt.ylabel('Style')

plt.xlabel('Ramen Rating')





plt.boxplot([Cup,Pack,Tray,Bowl,Box,Can,Bar,NaN], labels=labels, vert=False, flierprops=outliers)

plt.grid(False)

plt.show()
#Created a new column to practice Machine Learning Binary Classification





#If a ramen has a rating of 3.5 or higher it gets a value of 1 else it is 0.



ramen_data['Buy/Pass'] = [1 if x >= 3.5 else 0 for x in ramen_data['Stars']]

ramen_data=ramen_data.drop(['Top Ten', 'Top_Ten'], axis=1)
ramen_data.head()

from sklearn import preprocessing

data_binary = ramen_data.apply(preprocessing.LabelEncoder().fit_transform)
Xb = data_binary.drop(['Review #','Buy/Pass'], axis=1)

yb = data_binary['Buy/Pass']



#Split the data for training and testing



from sklearn.model_selection import train_test_split

Xb_train,Xb_test,yb_train,yb_test = train_test_split(Xb,yb, test_size=0.2, random_state=42)

#Import logistic regression model and scale the data





from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import scale

logreg = LogisticRegression(random_state =42)



#Fitting the scaled data into logistic regression model

Xb = scale(Xb)

Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, yb, test_size=0.2, random_state=42)

logreg.fit(Xb_train,yb_train)



yb_logreg_pred =logreg.predict(Xb_test)

from sklearn.metrics import confusion_matrix

confusion_matrix(yb_test,yb_logreg_pred)
from sklearn.metrics import classification_report



print(classification_report(yb_test,yb_logreg_pred))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(Xb_train,yb_train)
yb_knn_pred = knn.predict(Xb_test)
confusion_matrix(yb_test,yb_knn_pred)
print(classification_report(yb_test,yb_knn_pred))

from sklearn.model_selection import cross_val_score



#Find Max distance

k_range = range(1,51)

k_scores =[]

#loop through reasonable values of k

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors= k)

    scores = cross_val_score(knn,Xb,yb,cv= 10,scoring='accuracy')

    k_scores.append(scores.mean())

print(k_scores)
k=pd.DataFrame(k_scores,columns= ['distance'])

k[k['distance']== k['distance'].max()]
plt.plot(k_range,k_scores)

plt.title('K values vs Cross Validated Accuracy')

plt.xlabel('K values for KNN')

plt.ylabel('Cross validated Accuracy')

plt.show()

knn = KNeighborsClassifier(n_neighbors= 12)

knn.fit(Xb_train,yb_train)
yb_knn_pred = knn.predict(Xb_test)
confusion_matrix(yb_test,yb_knn_pred)
print(classification_report(yb_test,yb_knn_pred))
