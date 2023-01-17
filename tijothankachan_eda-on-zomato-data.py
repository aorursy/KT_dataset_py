#import the necessary libraries

from collections import Counter

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
# load data to pandas dataframe

data=pd.read_csv('../input/zomato-bangalore-restaurants/zomato.csv')
#rename column

data1=data.rename(columns={'approx_cost(for two people)':'cost','listed_in(type)':'food_type','reviews_list':'review','listed_in(city)':'city'}) #rename columns

print("Percentage null or na values in df")

((data1.isnull() | data1.isna()).sum() * 100 / data1.index.size).round(2)
# getting some info abut data

data1.info()
#converting to string

data1['rate']=data1['rate'].astype('str')
#finding the unique words

data1['rate'].unique()
#replacing by nan

data1.rate.replace(('NEW','-'),np.nan,inplace =True)  
data1['rate']=data1['rate'].astype('str')

data1['rate']=data1['rate'].apply(lambda x: x.replace('/5','').strip())
data1.dropna(how='any',inplace = True)
data1=data1.loc[data1['votes']!='nan']
# deleting unnecessary column

column_to_drop = ['address','url', 'phone']

data1.drop(columns=column_to_drop, axis=1,inplace=True)
#checking duplicates

data1.duplicated().sum() 
#removing duplicates

data1.drop_duplicates(inplace=True)              
data1['rate']=data1['rate'].astype('float')
data1.shape
#which restaurant has more rating

test1=data1.groupby('name',as_index=False)['rate'].mean()

test1.sort_values('rate',ascending=False)[:10]             #descending

data1['rest_type'].unique()
plt.rcParams['figure.figsize'] = 6,8

rest=data1['rest_type'].value_counts()[:20]

sns.barplot(rest,rest.index)

plt.title('preferred rest_type', fontdict={'fontweight':'bold', 'fontsize': 18})

plt.xlabel("count")
data1['rate']=data1['rate'].astype('float')

bins =[0,2,3,4,5]

labels =['not recommended','average','good','highly recommended']

data1['rate_range'] = pd.cut(data1['rate'], bins=bins,labels=labels)

data1.loc[:5,['rate','rate_range']]

ct= pd.crosstab(data1['food_type'],data1['rate_range'])

ct.plot.bar(stacked=True)

plt.legend(title='rate')                               #stack

plt.title('preferring food type', fontdict={'fontweight':'bold', 'fontsize': 18})   #font style

plt.show()
plt.rcParams['figure.figsize'] = (15, 9)

x = pd.crosstab(data1['cost'], data1['book_table'])

x.div(x.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = True,color=['red','yellow'])

plt.title('Table booking vs cost', fontweight = 30, fontsize = 20)

plt.legend(loc="upper right")

plt.show()
plt.rcParams['figure.figsize'] = 15,8

# sns.barplot(y=data0['location'].value_counts()[:2].index,color='#abcdef')

rest=data1['location'].value_counts()[:20]

sns.barplot(rest.index,rest)

plt.title('food lovers belongs to', fontdict={'fontweight':'bold', 'fontsize': 18})

plt.xticks(rotation='vertical')

plt.show()
plt.rcParams['figure.figsize'] = 15,8

plt.subplot(2,1,1)

# plt.rcParams['figure.figsize'] = 15,8

sns.countplot('location',hue='online_order',data=data1)

plt.title('preferring online order', fontdict={'fontweight':'bold', 'fontsize': 18})

plt.xticks(rotation='vertical')

plt.show()



plt.subplot(2,1,2)

# plt.rcParams['figure.figsize'] = 15,8

sns.countplot('location',hue='book_table',data=data1)

plt.title('preferring book table ', fontdict={'fontweight':'bold', 'fontsize': 18})

plt.xticks(rotation='vertical')

plt.show()
#expensive food_type



temp=data1[['food_type','cost']].sort_values('cost',ascending=False).reset_index(drop=True)  

temp=temp.head(20)

temp.style.background_gradient(cmap='Blues')


plt.rcParams['figure.figsize'] = 15,8

sns.countplot(y='rate',hue='online_order',data=data1)

plt.title('rate vs online', fontdict={'fontweight':'bold', 'fontsize': 18})

plt.xticks(rotation='vertical')

plt.show()
data1['cost'].unique()
data1['cost']=data1['cost'].astype('str')
data1['cost']=data1['cost'].apply(lambda x: x.replace(',','').strip())
data1['cost']=data1['cost'].astype('int')
plt.rcParams['figure.figsize'] = 8,15

sns.distplot(data1['cost'],color='#abcdef',kde=False)

plt.title('peferred price rate', fontdict={'fontweight':'bold', 'fontsize': 18})

plt.xticks(rotation='vertical')

plt.show()
plt.rcParams['figure.figsize'] = 18,15

sns.scatterplot(x='cost',y='location',hue='online_order',data=data1)

plt.title('affordable cost per location', fontdict={'fontweight':'bold', 'fontsize': 18})

plt.xticks(rotation='vertical')

plt.grid()

plt.show()
from palettable.colorbrewer.qualitative import Pastel1_7

plt.rcParams['figure.figsize'] = 9,6

plt.subplot(1, 2, 1)

data1.online_order.value_counts().plot('pie',colors=Pastel1_7.hex_colors)

# add white circle to male donut plot

w_circle=plt.Circle( (0,0), 0.7, color='white')

p=plt.gcf()

p.gca().add_artist(w_circle)

plt.title('Online Order Service',weight ='bold')



plt.subplot(1, 2, 2)

data1.book_table.value_counts().plot('pie',colors=Pastel1_7.hex_colors)

# add white circle to male donut plot

w_circle=plt.Circle( (0,0), 0.7, color='white')

p=plt.gcf()

p.gca().add_artist(w_circle)

plt.title('Book Table Service',weight ='bold')

plt.tight_layout()

plt.rcParams['figure.figsize'] = 9,6

plt.show()

# BEST DISH

from collections import Counter 

lst=[]

for line in data1['dish_liked']:

    word=line.split(',')

    for i in range(0,len(word)):

        lst.append(word[i])

        

Counter = Counter(lst) 

most_occur = Counter.most_common(10) 

print(most_occur)
#extracting rate from review column

data1['review_rate']=''

lst2=[]

for index,row in data1.iterrows():

    lst1=[]

    b=0

#     print(row['reviews'])

    for  i in eval(row['review']):

        if i[0] is not None:

                a=float(i[0][-3:])

                lst1.append(a)

        else:

            b=0

#             print(b)

    

    if(len(lst1)>0):

                b=sum(lst1)/len(lst1)

                b="%.2f" % b

                        

    

    data1.loc[index,'review_rate']=b

       

            

            
#extracting text only from review column 

data1['review_only']=''



for index,row in data1.iterrows():

    a=''

      

    for  i in eval(row['review']):



        a += i[1].replace('RATED\n','').strip()

            

    data1.loc[index,'review_only']=a

data1[['review_rate','review_only']]
collection=[]

for index,row in data1.iterrows():

        line = [x.strip() for x in row['dish_liked'].split(',')]          

        for i in line:

            collection.append(i)

#             print(i)

menu_set=set(collection)

menu_set.intersection(data1.review_only[1000].split(' '))
from collections import Counter

line=[x.strip() for x in ','.join(data1['dish_liked']).split(',')]          #stripping and splitting  in python

counter=Counter(line)

counter=counter.most_common(20)

dish_count=pd.DataFrame(counter, columns = ['dish', 'count'])
dish_count=dish_count.head(10)

plt.rcParams['figure.figsize'] = 8,6

sns.barplot(x='dish',y='count',data=dish_count)

plt.title('most loved dishes', fontdict={'fontweight':'bold', 'fontsize': 18})

plt.xticks(rotation='vertical')

plt.grid()

plt.show()
from collections import Counter

loc_dish = data1.groupby('location')['dish_liked'].value_counts()

ind = loc_dish.index.levels[0]

location=[]

dish=[]

count=[]

for i in ind:

    dishes=[x.strip() for x in ','.join(loc_dish[i].index).split(',')]

    counter=Counter(dishes)

    counter=counter.most_common(1)

    for  j in counter:

        location.append(i)

        dish.append(j[0])

        count.append(j[1])
loc_dish_df = pd.DataFrame({'location':location,'top_dish':dish,'count':count}).head(20)    

loc_dish_df