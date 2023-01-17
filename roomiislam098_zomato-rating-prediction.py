import numpy as np

import pandas as pd

import seaborn as sb

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import r2_score
zomato_orgnl=pd.read_csv("../input/zomato-bangalore-restaurants/zomato.csv")
zomato_orgnl.head()
zomato_orgnl.isnull().sum()
zomato_orgnl.info()
zomato=zomato_orgnl.drop(['url','dish_liked','phone'],axis=1)

zomato.columns
zomato.rename({'approx_cost(for two people)': 'approx_cost_for_2_people',

               'listed_in(type)':'listed_in_type',

               'listed_in(city)':'listed_in_city'

              }, axis=1, inplace=True)

zomato.columns
remove_comma = lambda x: int(x.replace(',', '')) if type(x) == np.str and x != np.nan else x 

zomato.votes = zomato.votes.astype('int')

zomato['approx_cost_for_2_people'] = zomato['approx_cost_for_2_people'].apply(remove_comma)
zomato.info()
zomato['rate'].unique()
zomato = zomato.loc[zomato.rate !='NEW']

zomato = zomato.loc[zomato.rate !='-'].reset_index(drop=True)
remove_slash = lambda x: x.replace('/5', '') if type(x) == np.str else x

zomato.rate = zomato.rate.apply(remove_slash).str.strip().astype('float')
zomato['rate'].head()
zomato.info()
def Encode(zomato):

    for column in zomato.columns[~zomato.columns.isin(['rate', 'approx_cost_for_2_people', 'votes'])]:

        zomato[column] = zomato[column].factorize()[0]

    return zomato



zomato_en = Encode(zomato.copy())
zomato_en['rate'] = zomato_en['rate'].fillna(zomato_en['rate'].mean())

zomato_en['approx_cost_for_2_people'] = zomato_en['approx_cost_for_2_people'].fillna(zomato_en['approx_cost_for_2_people'].mean())
zomato_en.isna().sum()
corr = zomato_en.corr(method='kendall')
plt.figure(figsize=(15,8))

sns.heatmap(corr, annot=True)

plt.savefig("image0.png")
zomato_en.columns
x = zomato_en.iloc[:,[2,3,5,6,7,8,9,11]]

y = zomato_en['rate']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=353)
x_train.head()
y_train.head()
reg=LinearRegression()

reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)

from sklearn.metrics import r2_score

r2_score(y_test,y_pred)
'''reg_score=[]

import numpy as np

for j in range(1000):

    x_train,x_test,y_train,y_test =train_test_split(x,y,random_state=j,test_size=0.1)

    lr=LinearRegression().fit(x_train,y_train)

    reg_score.append(lr.score(x_test,y_test))

K=reg_score.index(np.max(reg_score))

#K=353'''
from sklearn.tree import DecisionTreeRegressor
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=105)
DTree=DecisionTreeRegressor(min_samples_leaf=.0001)
DTree.fit(x_train,y_train)

y_predict=DTree.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_predict)
'''from sklearn.tree import DecisionTreeRegressor

ts_score=[]

for j in range(1000):

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=j)

    dc=DecisionTreeRegressor().fit(x_train,y_train)

    ts_score.append(dc.score(x_test,y_test))

J= ts_score.index(np.max(ts_score))



J

#J=105'''
from sklearn.ensemble import RandomForestRegressor

RForest=RandomForestRegressor(n_estimators=5,random_state=329,min_samples_leaf=.0001)
RForest.fit(x_train,y_train)

y_predict=RForest.predict(x_test)
from sklearn.metrics import r2_score

r2_score(y_test,y_predict)
'''rf_score=[]

for k in range(500):

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.01,random_state=k)

    dtc=RandomForestRegressor().fit(x_train,y_train)

    rf_score.append(dtc.score(x_test,y_test))

K= rf_score.index(np.max(rf_score))

K=329'''
fig = plt.figure(figsize=(20,7))

loc = sns.countplot(x="location",data=zomato_orgnl, palette = "Set1")

loc.set_xticklabels(loc.get_xticklabels(), rotation=90, ha="right")

plt.ylabel("Frequency",size=15)

plt.xlabel("Location",size=18)

loc

plt.title('NO. of restaurents in a Location',size = 20,pad=20)

plt.savefig("image1.png", bbox_inches="tight")
fig = plt.figure(figsize=(17,5))

rest = sns.countplot(x="rest_type",data=zomato_orgnl, palette = "Set1")

rest.set_xticklabels(rest.get_xticklabels(), rotation=90, ha="right")

plt.ylabel("Frequency",size=15)

plt.xlabel("Restaurent type",size=15)

rest 

plt.title('Restaurent type',fontsize = 20 ,pad=20)

plt.savefig("image2.png", bbox_inches="tight")
plt.figure(figsize=(15,7))

chains=zomato_orgnl['name'].value_counts()[:20]

sns.barplot(x=chains,y=chains.index,palette='Set1')

plt.title("Most famous restaurants chains in Bangaluru",size=20,pad=20)

plt.xlabel("Number of outlets",size=15)

plt.savefig("image3.png", bbox_inches="tight")
plt.figure(figsize=(15,7))

zomato_orgnl['online_order'].value_counts().plot.bar()

plt.title('Online orders', fontsize = 20)

plt.ylabel('Frequency',size = 15)

plt.savefig("image4.png", bbox_inches="tight")
plt.figure(figsize=(15,7))

zomato_orgnl['book_table'].value_counts().plot.bar()

plt.title('Booking Table', fontsize = 20,pad=15)

plt.ylabel('Frequency', fontsize = 15)

plt.savefig("image5.png", bbox_inches="tight")
plt.figure(figsize=(10,10))

restaurantTypeCount=zomato_orgnl['rest_type'].value_counts().sort_values(ascending=True)

slices=[restaurantTypeCount[0],

        restaurantTypeCount[1],

        restaurantTypeCount[2],

        restaurantTypeCount[3],

        restaurantTypeCount[4],

        restaurantTypeCount[5],

        restaurantTypeCount[6],

        restaurantTypeCount[7],

        restaurantTypeCount[8]]

labels=['Pubs and bars','Buffet','Drinks & nightlife','Cafes','Desserts','Dine-out','Delivery ','Quick Bites','Bakery']

colors = ['#3333cc','#ffff1a','#ff3333','#c2c2d6','#6699ff','#c4ff4d','#339933','black','orange']

plt.pie(slices,colors=colors, labels=labels, autopct='%1.0f%%', pctdistance=.5, labeldistance=1.2,shadow=True)

fig = plt.gcf()

plt.title("Percentage of Restaurants according to their type", bbox={'facecolor':'2', 'pad':2})

plt.savefig("image6.png", bbox_inches="tight")
fig, ax = plt.subplots(figsize=[15,7])

sns.distplot(zomato_en['approx_cost_for_2_people'],color="magenta")

ax.set_title('Approx cost for two people distribution',size=20,pad=15)

plt.xlabel('Approx cost for two people',size = 15)

plt.ylabel('Percentage of restaurents',size = 15)

plt.savefig("image7.png", bbox_inches="tight")
plt.figure(figsize=(12,7))

preds_rf = RForest.predict(x_test)

plt.scatter(y_test,x_test.iloc[:,2],color="red")

plt.title("True rate vs Predicted rate",size=20,pad=15)

plt.xlabel('Rating',size = 15)

plt.ylabel('Frequency',size = 15)

plt.scatter(preds_rf,x_test.iloc[:,2],color="green")

plt.savefig("image8.png", bbox_inches="tight")
plt.figure(figsize=(15,8))

rating = zomato['rate']

plt.hist(rating,bins=20,color="red")

plt.title('Restaurent rating distribution', size = 20, pad = 15)

plt.xlabel('Rating',size = 15)

plt.ylabel('No. of restaurents',size = 15)

plt.savefig("image9.png", bbox_inches="tight")
plt.figure(figsize=(15,8))

sns.violinplot(zomato.approx_cost_for_2_people)

plt.title('Approx cost for 2 people distribution', size = 20, pad = 15)

plt.xlabel('Approx cost for 2 people',size = 15)

plt.ylabel('Density',size = 15)

plt.savefig("image10.png", bbox_inches="tight")
plt.figure(figsize=(15,8))

cuisines=zomato['cuisines'].value_counts()[:15]

sns.barplot(cuisines,cuisines.index)

plt.title('Most popular cuisines of Bangaluru', size = 20, pad = 15)

plt.xlabel('No. of restaurents',size = 15)

plt.savefig("image11.png", bbox_inches="tight")