from IPython.display import HTML

%config IPCompleter.greedy=True

import pandas as pd

from collections import Counter

import numpy as nm

import matplotlib.pyplot as plt

import warnings

import random

import datetime

warnings.filterwarnings("ignore")

from IPython.display import display, Image



display(Image(filename='../input/picture/olist.png'))



customers=pd.read_csv("../input/brazilian-ecommerce/olist_customers_dataset.csv")

customers.name= "customers"

geo=pd.read_csv("../input/brazilian-ecommerce/olist_geolocation_dataset.csv")

geo.name= "geolocation"

items=pd.read_csv("../input/brazilian-ecommerce/olist_order_items_dataset.csv")

items.name="order items"

payments=pd.read_csv("../input/brazilian-ecommerce/olist_order_payments_dataset.csv")

payments.name= "order payments"

reviews=pd.read_csv("../input/brazilian-ecommerce/olist_order_reviews_dataset.csv")

reviews.name= "order reviews"

orders=pd.read_csv("../input/brazilian-ecommerce/olist_orders_dataset.csv")

orders.name='orders'

products=pd.read_csv("../input/brazilian-ecommerce/olist_products_dataset.csv")

products.name="products"

sellers=pd.read_csv("../input/brazilian-ecommerce/olist_sellers_dataset.csv")

sellers.name="sellers"
def exploreFrequencies(customers):

    print("{0:30} {1:25} {2:25}".format(customers.name, "unique values", "missing values"))

    for i in customers:

        print("{0:30} {1:20} {2:20}".format(i, customers[i].nunique(),customers[i].isna().sum()))

    print("------------------------------------")
exploreFrequencies(customers)

exploreFrequencies(items)

exploreFrequencies(payments)

exploreFrequencies(reviews)

exploreFrequencies(orders)

exploreFrequencies(products)

exploreFrequencies(sellers)

exploreFrequencies(geo)

k=pd.DataFrame({'customers':customers['customer_state'].value_counts(),'sellers':sellers['seller_state'].value_counts()})

print("-------Customers and sellers location per state-----")

k=k.sort_values(by='customers',ascending=False)

k=k.fillna(0)

print(k)

k['sellers']= k['sellers'].apply( lambda x:x/k['sellers'].sum())

k['customers']= k['customers'].apply( lambda x:x/k['customers'].sum())


labels = k.T.columns

sel = k['sellers']

cus = k['customers']



x = nm.arange(len(labels))  # the label locations

width = 0.2  # the width of the bars



fig, ax = plt.subplots(figsize=(20,10))



rects1 = ax.bar(x - width/2, sel, width, label='Sellers')

rects2 = ax.bar(x + width/2, cus, width, label='Customers')



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Percentage of total per state')

ax.set_xlabel('States')

ax.set_title('Customers and sellers location by state')

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()
plt.boxplot(items.groupby(by='order_id')['price'].sum(), showfliers=False)

plt.title("Order total without outliers")

plt.ylabel("currency units")
plt.boxplot(items['price'], showfliers=False)

plt.title("Item price without outliers")

plt.ylabel("currency units")
d=pd.DataFrame(items.groupby(by='seller_id').size().sort_values(0,ascending=False))

d.head(5)

plt.boxplot(d.T,showfliers=False)

plt.title("Number of items sold by one seller")

plt.ylabel("Number of items")
z=pd.DataFrame(payments['payment_type'].value_counts())

plt.bar(z.index, z.payment_type,tick_label=z.T.columns)

plt.title("Payment methods")

plt.ylabel("Number of transactions used a specific method")

ax=pd.DataFrame(items.groupby(by='order_id')['order_item_id'].size().value_counts()).apply(lambda x:x/items['order_item_id'].sum()).plot(kind="bar", title="Items number per order", rot=0)

ax.legend("")

ax.set_xlabel("Items per order")

ax.set_ylabel("Percentage")





orders["year"]= orders['order_purchase_timestamp'].str[:4]

col=[orders["year"]==2017]

sc=orders["year"].value_counts()

print(sc)



ax=sc.plot(kind="bar", title="Number of orders per year", rot=0)

#will remove

orders.pivot_table(index=['customer_id'], aggfunc='size').value_counts()
#reviews per order

a=reviews.groupby('order_id').size()

dd=a.value_counts()

print(dd)

az=dd.plot(kind="bar", title= "Number of reviews per order", rot=0)

az
o=reviews['review_score'].value_counts()

ax=o.plot(kind='bar', title= " Distribution of review scores", rot=90)

ax.set_xlabel("Review Score")

ax.set_ylabel("Order number for a specific review score")

items[['price','freight_value']].describe()
plt.scatter(items['price'], items['freight_value'])

plt.title("relationship between price and freight cost")

plt.ylabel("Freight cost")

plt.xlabel("Item price")
f=products[['product_id','product_description_lenght','product_photos_qty']]

i=pd.DataFrame(items.groupby(by='product_id').size()).reset_index()

f= f.merge(i, how="left", on='product_id')

f['quantity_sold']=f[0]

f=f.drop(0, axis=1)

print(f[['quantity_sold','product_description_lenght']].describe())

plt.scatter(f['quantity_sold'], f['product_description_lenght'])

plt.title("relationship between description length and sold items")

plt.ylabel("Description length")

plt.xlabel("Quantity sold")
print(f[['quantity_sold','product_photos_qty']].describe())

plt.scatter(f['quantity_sold'], f['product_photos_qty'])

plt.title("relationship between photos quantity and number of sold items")

plt.ylabel("Photos number")

plt.xlabel("Quantity sold")

plt.yticks(nm.arange(0, 21, step=1))

plt.show()
s=pd.DataFrame(customers.groupby('customer_unique_id').size().reset_index())

s['nbOrders']=s[0]

s.drop(0,axis=1)

f=s['nbOrders'].value_counts()

f
d=s.merge(customers.drop_duplicates('customer_unique_id',keep="last"),how='left',on="customer_unique_id")

co=d['nbOrders'].value_counts()

#plt.bar(co.index,co.values)

ax=co.plot(kind='bar', title="Number of orders per customer", rot=0)

ax.set_xlabel("Number of customers per specific number of orders")

ax.set_ylabel("Number of orders per customer")

d=d.drop(0,axis=1)

#keep max 10 orders

d['nbOrders']=d['nbOrders'].apply(lambda s:s if s<4 else 3)

ordersList=customers.groupby('customer_unique_id')['customer_id'].apply(list).to_dict()



#list of customer_ids associated with customer_unique_ids. 1 customer unique id can have many ids      
'''

############################################################

###############        Extecuted code     ##################

############################################################

money= pd.DataFrame(columns=["customer_unique_id","total"])

for i in ordersList:

    total=0

    orderCounter=0

    for u in ordersList[i]:

        ordersPerCs=orders.loc[orders['customer_id']==u].values

        for orde in ordersPerCs:

            orderCounter=orderCounter+1

            

            itemsOrder=items.loc[items['order_id']==orde[0]].values

            for ea in itemsOrder:

                #print(ea[1])

            #if(len(itemsOrder['price'])>0):

                #total+=itemsOrder['price'].values[0]  

                total+=ea[5]

    if orderCounter>0:

        v=total/orderCounter

    else:

        v=0

        print(total)

    

    money=money.append({'customer_unique_id':i, 'total':v},ignore_index=True)

money.to_csv("averagePerOrderMoneySpent.csv", encoding='utf-8', index=False)

'''
money=pd.read_csv("../input/olistnewfeatures/averagePerOrderMoneySpent.csv")

print(money['total'].describe())

money['total'].plot(kind='box', showfliers=False, title="Average money spent by customer per order")

money['total']=money['total'].apply(lambda s:s if s<300 else 300)
'''

############################################################

###############        Extecuted code     ##################

############################################################

revs= pd.DataFrame(columns=["customer_unique_id","reveiwScore"])

for i in ordersList:

    score=0

    count=0

    

    for u in ordersList[i]:

        ordersPerCs=orders.loc[orders['customer_id']==u].values

        for orde in ordersPerCs:

            reviewsOrder=reviews.loc[reviews['order_id']==orde[0]].values

            for r in reviewsOrder:

                score=score+r[2]

                count=count+1

                

    revs=revs.append({'customer_unique_id':i, 'reviewScore':score/count},ignore_index=True)

revs.to_csv("reviewsCustomers.csv", encoding='utf-8', index=False)

 '''   

    
revs=pd.read_csv("../input/olistnewfeatures/reviewsCustomers.csv")

print(revs['reviewScore'].describe())

revs['reviewScore'].plot(kind='box', title="Average score left by customer")
#consider 5 star order as happy customer ready to return= 1

#2= loyals

revs=revs.merge(d[['customer_unique_id','nbOrders']],how='left',on='customer_unique_id')

revs['retentionScore']=revs['reviewScore']/5*revs['nbOrders']

print(revs['retentionScore'].describe())

#revs['retentionScore'].plot(kind='box', title="Retention score pre customer review score * Nb of orders")

revs['retentionScore']=revs['retentionScore'].apply(lambda s:s if s<3 else 3 )

revs['retentionScore'].plot(kind='box', title="Retention score pre customer review score * Nb of orders")

revs=revs.drop('nbOrders', axis=1)



'''

############################################################

###############        Extecuted code     ##################

############################################################

vendeurs= pd.DataFrame(columns=["customer_unique_id","totalSellers"])

for i in ordersList:

    total=[]

    

    for u in ordersList[i]:

        ordersPerCs=orders.loc[orders['customer_id']==u].values

        for orde in ordersPerCs:

            itemsOrder=items.loc[items['order_id']==orde[0]]

            if(len(itemsOrder['seller_id'])>0):

                

                sstr=str(itemsOrder['seller_id'].values)

                

                if (sstr not in total):

                    

                    total.append(sstr)

    vendeurs=vendeurs.append({'customer_unique_id':i, 'totalSellers':len(total)},ignore_index=True)

vendeurs.to_csv("resellersPerCustomer.csv", encoding='utf-8', index=False)



'''
vend=pd.read_csv("../input/olistnewfeatures/resellersPerCustomer.csv")

print(vend['totalSellers'].value_counts())

vend['totalSellers'].value_counts().plot(kind='bar', title="Number of resellers per customer",rot=0)

vend['totalSellers']=vend['totalSellers'].apply(lambda s:s if s<6 else 5)
from math import sin, cos, sqrt, atan2, radians



def distance(a,b):

    R = 6373.0

    aa=geo.loc[geo['geolocation_zip_code_prefix'].values == a].head(1)

    bb= geo.loc[geo['geolocation_zip_code_prefix'].values == b].head(1)

    if len(aa)==0:

        a1= geo['geolocation_lat'].mean()

        a2= geo['geolocation_lng'].mean()

    else:

        a1=aa['geolocation_lat'].values

        a2=aa['geolocation_lng'].values

    if len(bb)==0:

        b1= geo['geolocation_lat'].mean()

        b2= geo['geolocation_lng'].mean()

    else:

        b1=bb['geolocation_lat'].values

        b2=bb['geolocation_lng'].values

            

   

    lat1 = radians(a1)

    lon1 = radians(a2)

    lat2 = radians(b1)

    lon2 = radians(b2)



    dlon = lon2 - lon1

    dlat = lat2 - lat1



    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2

    c = 2 * atan2(sqrt(a), sqrt(1 - a))



    return R * c

'''

############################################################

###############        Extecuted code     ##################

############################################################

distances= pd.DataFrame(columns=["customer_unique_id","maximumDistance"])

for i in ordersList:

    total=0

    customerZip=customers.loc[customers['customer_unique_id']==i]['customer_zip_code_prefix'].values[0]

    

    for u in ordersList[i]:

        ordersPerCs=orders.loc[orders['customer_id']==u].values

        for orde in ordersPerCs:

            itemsOrder=items.loc[items['order_id']==orde[0]]

            if(len(itemsOrder['seller_id'])>0):

                    for sal in itemsOrder['seller_id']:

                        zips= sellers.loc[sellers['seller_id']==sal]

                        sellerZip=zips['seller_zip_code_prefix'].values

                        #print(customerZip)

                        #print(int(sellerZip))

                        if(int(customerZip)>0 and int(sellerZip)>0):

                            dist=distance(int(customerZip),int(sellerZip))

                            if dist>total:

                                total=dist

                            

    distances=distances.append({'customer_unique_id':i, 'maximumDistance': total},ignore_index=True)

distances.to_csv("MaximumDistancePerCustomer.csv", encoding='utf-8', index=False)



'''
distance=pd.read_csv("../input/olistnewfeatures/MaximumDistancePerCustomer.csv")

print(distance['maximumDistance'].describe())

distance['maximumDistance'].plot(kind='box', title="Maximum distance customer purchased from", showfliers=False)

distance['maximumDistance']=distance['maximumDistance'].apply(lambda s:s if s<1750 else 1750)
'''

############################################################

###############        Extecuted code     ##################

############################################################

itemsNumber= pd.DataFrame(columns=["customer_unique_id","nbItems"])

#get all customer ids

for i in ordersList:

    totalItems=[]

    

    

    #get all customer orders corresponding to ids

    for u in ordersList[i]:

        ordersPerCs=orders.loc[orders['customer_id']==u].values

        #get all orders items corresponding to ids

        

        for orde in ordersPerCs:

            

            itemsOrder=items.loc[items['order_id']==orde[0]]

            if(itemsOrder.shape[0]>0):

                totalItems.append(itemsOrder.shape[0])

    

    if len(totalItems)>0:

        c = Counter(totalItems)

        v=c.most_common(1)[0][0]

        

    else:

        v=0

        

    itemsNumber=itemsNumber.append({'customer_unique_id':i, 'nbItems': v},ignore_index=True)

    

itemsNumber.to_csv("itemsNumberPerCustomer.csv", encoding='utf-8', index=False)

'''
itemsNumber=pd.read_csv("../input/olistnewfeatures/itemsNumberPerCustomer.csv")

vv=itemsNumber['nbItems'].value_counts()

print(vv.head(5))

vv.plot(kind='bar', title="Items frequency per customer orders",rot=0)

itemsNumber['nbItems']=itemsNumber['nbItems'].apply(lambda s:s if s<6 else 5)
'''

############################################################

###############        Extecuted code     ##################

############################################################

from collections import Counter



paymentMethod= pd.DataFrame(columns=["customer_unique_id","paymentMethod"])

#get all customer ids

for i in ordersList:

    totalItems=[]

    

    

    #get all customer orders corresponding to ids

    for u in ordersList[i]:

        ordersPerCs=orders.loc[orders['customer_id']==u].values

        #get all orders items corresponding to ids

        

        for orde in ordersPerCs:

            

            method=payments.loc[payments['order_id']==orde[0]]

            if(method.empty!=True):

                

                totalItems.append(method.values[0][2])

    if len(totalItems)>0:

        c = Counter(totalItems)

        v=c.most_common(1)[0][0]

    else:

        v='not known'

    

    paymentMethod=paymentMethod.append({'customer_unique_id':i, 'paymentMethod': v},ignore_index=True)

    

paymentMethod.to_csv("paymentMethod.csv", encoding='utf-8', index=False)

'''
paymentMethod=pd.read_csv("../input/olistnewfeatures/paymentMethod.csv")

v=paymentMethod['paymentMethod'].value_counts()

print(v)

ax=v.plot(kind='barh', title="Prefereed payment method")

ax
paymentMethod['paymentMethodScore']=paymentMethod['paymentMethod'].replace(v.index,v.values)

paymentMethod['paymentMethod']=paymentMethod.drop('paymentMethod',axis=1)
'''

############################################################

###############        Extecuted code     ##################

############################################################

category= pd.DataFrame(columns=["customer_unique_id","category"])

#get all customer ids

for i in ordersList:

    totalItems=[]

    #get all customer orders corresponding to ids

    for u in ordersList[i]:

        ordersPerCs=orders.loc[orders['customer_id']==u].values

        #get all orders items corresponding to ids

        

        for orde in ordersPerCs:

            

            itemsOrder=items.loc[items['order_id']==orde[0]].values

            for ea in itemsOrder:

                

                produits=products.loc[products['product_id']==ea[2]].values

                

                if (len(produits[0])>0):

                    totalItems.append(produits[0][1])

                        

                                      

    if len(totalItems)>0:

        c = Counter(totalItems)

        c=c.most_common(1)[0][0]

        

    else:

        c='not known'

    

    category=category.append({'customer_unique_id':i, 'category': c},ignore_index=True)

category.to_csv("categoryCustomers.csv", encoding='utf-8', index=False)

'''
category=pd.read_csv("../input/olistnewfeatures/categoryCustomers.csv")

print("Number of preferred categories: "+str(len(category['category'].unique())))

print("First 10 categories represent :"+str(category['category'].value_counts().head(10).values.sum()/len(category['category'])*100)+ "% of customers choice")



ax=category['category'].value_counts().head(10).plot(kind='barh', title="10 top prefered categories per customers")

ax
k=category['category'].value_counts()

category['categoryScore']=category['category'].replace(k.index,k.values)

category=category.drop('category', axis=1)
'''

############################################################

###############        Extecuted code     ##################

############################################################

frequentReseller= pd.DataFrame(columns=["customer_unique_id","frequent reseller"])

#get all customer ids

for i in ordersList:

    totalSellers=[]

    #get all customer orders corresponding to ids

    for u in ordersList[i]:

        ordersPerCs=orders.loc[orders['customer_id']==u].values

        #get all orders items corresponding to ids

        

        for orde in ordersPerCs:

            

            itemsOrder=items.loc[items['order_id']==orde[0]].values

            for ea in itemsOrder:

                

                totalSellers.append(ea[3])

                                    

    if len(totalSellers)>0:

        c = Counter(totalSellers)

        c=c.most_common(1)[0][0]

        

    else:

        c='not known'

    

    frequentReseller=frequentReseller.append({'customer_unique_id':i, 'frequent reseller': c},ignore_index=True)

frequentReseller.to_csv("frequentResellers.csv", encoding='utf-8', index=False)

'''
frequentReseller=pd.read_csv("../input/olistnewfeatures/frequentResellers.csv")

print("Number of preferred resellers: "+str(len(frequentReseller['frequent reseller'].unique())))

print("Top 200 resellers represent :"+str(frequentReseller['frequent reseller'].value_counts().head(200).values.sum()/len(frequentReseller['frequent reseller'])*100)+ "% of customers choice")

frequentReseller['frequent reseller'].value_counts().head(10).plot(kind='barh', title="10 top resellers")

k=frequentReseller['frequent reseller'].value_counts()

frequentReseller['frequentResellerScore']=frequentReseller['frequent reseller'].replace(k.index,k.values)

frequentReseller=frequentReseller.drop('frequent reseller', axis=1)
'''

############################################################

###############        Extecuted code     ##################

############################################################

purchaseCity= pd.DataFrame(columns=["customer_unique_id","frequent purchase city"])

#get all customer ids

for i in ordersList:

    totalAreas=[]

    #get all customer orders corresponding to ids

    for u in ordersList[i]:

        ordersPerCs=orders.loc[orders['customer_id']==u].values

        #get all orders items corresponding to ids

        

        for orde in ordersPerCs:

            

            itemsOrder=items.loc[items['order_id']==orde[0]].values

            for ea in itemsOrder:

                zips= sellers.loc[sellers['seller_id']==ea[3]].values

                zipo= zips[0][3]

                

                totalAreas.append(zipo)

                                  

    if len(totalAreas)>0:

        c = Counter(totalAreas)

        c=c.most_common(1)[0][0]

        

    else:

        v='not known'

    

    purchaseCity=purchaseCity.append({'customer_unique_id':i, 'frequent purchase city': c},ignore_index=True)

purchaseCity.to_csv("frequentState.csv", encoding='utf-8', index=False)

'''
purchaseCity=pd.read_csv("../input/olistnewfeatures/frequentState.csv")

c=purchaseCity['frequent purchase city'].value_counts().head(5)

print(c)

c.plot(kind="barh", title="5 top prefered purchase states")
k=purchaseCity['frequent purchase city'].value_counts()

purchaseCity['frequentPurchaseStateScore']=purchaseCity['frequent purchase city'].replace(k.index,k.values)

purchaseCity= purchaseCity.drop('frequent purchase city', axis=1)
'''

############################################################

###############        Extecuted code     ##################

############################################################

import datetime

date_time_str = '2019-01-01 01:01:01'

dateNow = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')

purchaseYear= pd.DataFrame(columns=["customer_unique_id","lastPurchase"])

#get all customer ids

for i in ordersList:

    totalMonths=100

    #get all customer orders corresponding to ids

    for u in ordersList[i]:

        ordersPerCs=orders.loc[orders['customer_id']==u].values

        #get all orders items corresponding to ids

        

        for orde in ordersPerCs:

            if pd.isna(orde[4])!=True:

                date_time_str = orde[4]

                dateBefore = datetime.datetime.strptime(date_time_str, '%Y-%m-%d %H:%M:%S')

                dist=round((dateNow - dateBefore).days/30)                                       

                

                if(dist<totalMonths):

                    totalMonths=dist

                    

                else:

                    totalMonths=100

    if(totalMonths==100):

        totalMonths="not known"

    purchaseYear=purchaseYear.append({'customer_unique_id':i, 'lastPurchase': totalMonths},ignore_index=True)

purchaseYear.to_csv("lastPurchase.csv", encoding='utf-8', index=False)

'''
purchaseYear=pd.read_csv("../input/olistnewfeatures/lastPurchase.csv")

y=purchaseYear[purchaseYear["lastPurchase"] != "not known"]

median=y['lastPurchase'].median() 

purchaseYear["lastPurchase"].replace({"not known": median }, inplace=True)

purchaseYear["lastPurchase"].astype(float).plot(kind="box")
k=d['customer_state'].value_counts()

d['customerStateScore']=d['customer_state'].replace(k.index,k.values)

d=d.merge(money, on='customer_unique_id', how='left')

d=d.merge(revs, on='customer_unique_id', how='left')

d=d.merge(vend, on='customer_unique_id', how='left')

d=d.merge(distance, on='customer_unique_id', how='left')

d=d.merge(itemsNumber, on='customer_unique_id', how='left')

d=d.merge(paymentMethod, on='customer_unique_id', how='left')

d=d.merge(purchaseYear, on='customer_unique_id', how='left')

d=d.merge(purchaseCity, on='customer_unique_id', how='left')

d=d.merge(category, on='customer_unique_id', how='left')

d=d.merge(frequentReseller, on='customer_unique_id', how='left')

d=d.drop(['reviewScore',"customer_id",'customer_state'],axis=1)

d.columns


l=d[[ 'nbOrders', 

        'customerStateScore',

       'total', 'retentionScore', 'totalSellers', 'maximumDistance', 'nbItems',

        'paymentMethodScore', 'lastPurchase',

       'frequentPurchaseStateScore',  'frequentResellerScore']]

from sklearn import preprocessing



x = l.values #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

l = pd.DataFrame(x_scaled, columns=l.columns)



l.describe()
from sklearn.cluster import KMeans



num_clusters = 8

kmeans_tests = [KMeans(n_clusters=i, init='random', n_init=10) for i in range(1, num_clusters)]

score = [kmeans_tests[i].fit(l).score(l) for i in range(len(kmeans_tests))]

plt.plot(range(1, num_clusters),score)

plt.xlabel('Number of Clusters')

plt.ylabel('Score')

plt.title('Elbow Curve')

plt.show()

from sklearn.model_selection import KFold

from sklearn.metrics import silhouette_samples, silhouette_score

n_clusters=8

score=[]

for r in range(2,n_clusters): 

    

    clusterer = KMeans(n_clusters=r, random_state=10)

    cluster_labels = clusterer.fit_predict(l)

    silhouette_avg = silhouette_score(l, cluster_labels)

    score.append(silhouette_avg)

plt.plot(range(2, n_clusters),score)

plt.xlabel('Number of Clusters')

plt.ylabel('Silhouette Score')

plt.title('Elbow Curve Silhouette Score')

plt.show()
g=KMeans(n_clusters=4, init='random', n_init=10)

b=g.fit(l).labels_

#from sklearn.model_selection import cross_val_score

from sklearn.metrics import make_scorer,adjusted_rand_score

#j=b.reshape(-1, 1)



sc=make_scorer(adjusted_rand_score)

km=KMeans(n_clusters=4)

from sklearn.model_selection import GridSearchCV

parameters = {'init':('k-means++', 'random'),'max_iter':[100,200,300],'n_init':[5,10,15]}

clf = GridSearchCV(km, parameters,scoring=sc)

clf.fit(l,b)

indx=clf.cv_results_['rank_test_score'].tolist().index(1)

print("ARI score mean :"+str(clf.cv_results_['mean_test_score'][indx]))

print("Best parameters: "+str(clf.best_params_))


'''

#####################################################

### TESTING CUSTOM GRID SEARCH ######################

#####################################################



io={"parameter":{'init':['k-means++', 'random']}}



def customGrid1(parameter=None):

    

    average=[]

    for k,v in parameter.items():

        

        for g in v:

            kw={k:g}

            op = KMeans(n_clusters=4,**kw)

            sam=l

            lb= sam['cluster']

            sam= sam.drop('cluster',axis=1)

        

            kfold = KFold( n_splits=10)

            sco=[]



            for train_index, test_index in kfold.split(sam):

                train = sam.iloc[train_index]

                trainY=lb.iloc[train_index]

                test=sam.iloc[test_index]

                testY=lb.iloc[test_index]

                r1=op.fit(train).labels_

                r2=op.fit(test).labels_

                mm=adjusted_rand_score(r1,trainY)

                nn=adjusted_rand_score(r2,testY)

                mm=(mm+nn)/2

                sco.append(mm)

               

            ss=pd.DataFrame(sco).mean()

            

            average.append({g:ss})

    return average



print(customGrid1(**io))



####################################################################################################

##############                 Result            ###################################################

####################################################################################################

#[{'k-means++': 0    1.0

#dtype: float64}, {'random': 0    0.92968

#dtype: float64}]

####################################################################################################

'''



# Library of Functions for the OpenClassrooms Multivariate Exploratory Data Analysis Course



import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection

import numpy as np

import pandas as pd

from scipy.cluster.hierarchy import dendrogram

from pandas.plotting import parallel_coordinates

import seaborn as sns





palette = sns.color_palette("bright", 10)



def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):

    """Display correlation circles, one for each factorial plane"""



    # For each factorial plane

    for d1, d2 in axis_ranks: 

        if d2 < n_comp:



            # Initialise the matplotlib figure

            fig, ax = plt.subplots(figsize=(20,20))



            # Determine the limits of the chart

            if lims is not None :

                xmin, xmax, ymin, ymax = lims

            elif pcs.shape[1] < 30 :

                xmin, xmax, ymin, ymax = -1, 1, -1, 1

            else :

                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])



            # Add arrows

            # If there are more than 30 arrows, we do not display the triangle at the end

            if pcs.shape[1] < 30 :

                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),

                   pcs[d1,:], pcs[d2,:], 

                   angles='xy', scale_units='xy', scale=1, color="grey")

                # (see the doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)

            else:

                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]

                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))

            

            # Display variable names

            if labels is not None:  

                for i,(x, y) in enumerate(pcs[[d1,d2]].T):

                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :

                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)

            

            # Display circle

            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')

            plt.gca().add_artist(circle)



            # Define the limits of the chart

            plt.xlim(xmin, xmax)

            plt.ylim(ymin, ymax)

        

            # Display grid lines

            plt.plot([-1, 1], [0, 0], color='grey', ls='--')

            plt.plot([0, 0], [-1, 1], color='grey', ls='--')



            # Label the axes, with the percentage of variance explained

            plt.xlabel('PC{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))

            plt.ylabel('PC{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            nr=d1+1

            plt.title("Correlation Circle (PC{} and PC{})".format(d1+1, d2+1))

            plt.show(block=False)

            d = {'values': pca.components_[d1], 'factors': labels}

            df1= pd.DataFrame(d)

            df1.set_index('factors')

            df2=df1.sort_values(by='values', ascending=False)

            df3=df1.sort_values(by='values', ascending=True)

            print("Principal Component" + str(nr)+ " Presenting Values")

            print(df2.head(3))

            print(df3.head(3))

            

            nr=d2+1

            

            d = {'values': pca.components_[d2], 'factors': labels}

            df1= pd.DataFrame(d)

            df1.set_index('factors')

            df2=df1.sort_values(by='values', ascending=False)

            df3=df1.sort_values(by='values', ascending=True)

            print("Principal Component" + str(nr)+ " Presenting Values")

            print(df2.head(3))

            print(df3.head(3))



def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):

    '''Display a scatter plot on a factorial plane, one for each factorial plane'''



    # For each factorial plane

    for d1,d2 in axis_ranks:

        if d2 < n_comp:

 

            # Initialise the matplotlib figure      

            fig = plt.figure(figsize=(7,6))

        

            # Display the points

            if illustrative_var is None:

                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)

            else:

                illustrative_var = np.array(illustrative_var)

                for value in np.unique(illustrative_var):

                    selected = np.where(illustrative_var == value)

                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)

                plt.legend()



            # Display the labels on the points

            if labels is not None:

                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):

                    plt.text(x, y, labels[i],

                              fontsize='14', ha='center',va='center') 

                

            # Define the limits of the chart

            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1

            plt.xlim([-boundary,boundary])

            plt.ylim([-boundary,boundary])

        

            # Display grid lines

            plt.plot([-100, 100], [0, 0], color='grey', ls='--')

            plt.plot([0, 0], [-100, 100], color='grey', ls='--')



            # Label the axes, with the percentage of variance explained

            plt.xlabel('PC{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))

            plt.ylabel('PC{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))



            plt.title("Projection of points (on PC{} and PC{})".format(d1+1, d2+1))

            #plt.show(block=False)



def display_scree_plot(pca):

    '''Display a scree plot for the pca'''



    scree = pca.explained_variance_ratio_*100

    plt.bar(np.arange(len(scree))+1, scree)

    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')

    plt.xlabel("Number of principal components")

    plt.ylabel("Percentage explained variance")

    plt.title("Scree plot")

    plt.show(block=False)



def append_class(df, class_name, feature, thresholds, names):

    '''Append a new class feature named 'class_name' based on a threshold split of 'feature'.  Threshold values are in 'thresholds' and class names are in 'names'.'''

    

    n = pd.cut(df[feature], bins = thresholds, labels=names)

    df[class_name] = n



def plot_dendrogram(Z, names, figsize=(10,25)):

    '''Plot a dendrogram to illustrate hierarchical clustering'''



    plt.figure(figsize=figsize)

    plt.title('Hierarchical Clustering Dendrogram')

    plt.xlabel('distance')

    dendrogram(

        Z,

        labels = names,

        orientation = "left",

    )

    #plt.show()



def addAlpha(colour, alpha):

    '''Add an alpha to the RGB colour'''

    

    return (colour[0],colour[1],colour[2],alpha)



def display_parallel_coordinates(df, num_clusters):

    '''Display a parallel coordinates plot for the clusters in df'''



    # Select data points for individual clusters

    cluster_points = []

    for i in range(num_clusters):

        cluster_points.append(df[df.cluster==i])

    

    # Create the plot

    fig = plt.figure(figsize=(20, 25))

    title = fig.suptitle("Parallel Coordinates Plot for the Clusters", fontsize=18)

    fig.subplots_adjust(top=0.95, wspace=0)



    # Display one plot for each cluster, with the lines for the main cluster appearing over the lines for the other clusters

    for i in range(num_clusters):    

        plt.subplot(num_clusters, 1, i+1)

        for j,c in enumerate(cluster_points): 

            if i!= j:

                pc = parallel_coordinates(c, 'cluster', color=[addAlpha(palette[j],0.2)])

        pc = parallel_coordinates(cluster_points[i], 'cluster', color=[addAlpha(palette[i],0.5)])



        # Stagger the axes

        ax=plt.gca()

        for tick in ax.xaxis.get_major_ticks()[1::2]:

            tick.set_pad(20)        





def display_parallel_coordinates_centroids(df, num_clusters):

    '''Display a parallel coordinates plot for the centroids in df'''



    # Create the plot

    fig = plt.figure(figsize=(12, 5))

    title = fig.suptitle("Parallel Coordinates plot for the Centroids", fontsize=18)

    fig.subplots_adjust(top=0.9, wspace=0)



    # Draw the chart

    parallel_coordinates(df, 'cluster', color=palette)



    # Stagger the axes

    ax=plt.gca()

    for tick in ax.xaxis.get_major_ticks()[1::2]:

        tick.set_pad(5)    



l['cluster']=b

sample=l.sample(frac=0.1, replace=True, random_state=1)

# Display parallel coordinates plots, one for each cluster

display_parallel_coordinates(sample, 4)
l['cluster'].value_counts()
from sklearn.manifold import TSNE

lsample= l.sample(frac=0.3)

X_embedded = TSNE(n_components=2).fit_transform(lsample)

dataV=pd.DataFrame(X_embedded, columns=["D1","D2"])

dataV['cluster']=lsample['cluster'].tolist()

fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)

scatter = ax.scatter(dataV['D1'],dataV['D2'],

                  c=dataV['cluster'],   s=50)

ax.set_title('K-Means Clustering')

ax.set_xlabel('Dimention 1')

ax.set_ylabel('Dimention 2')

plt.colorbar(scatter)
#for more than 2 dimensions: minPts=2*dim (Sander et al., 1998)

from sklearn.cluster import OPTICS, cluster_optics_dbscan

import matplotlib.gridspec as gridspec

X=l

X=X.drop('cluster', axis=1)

clust = OPTICS(min_samples=2*len(X.columns), xi=.05, min_cluster_size=.05)



# Run the fit

labelsDB=clust.fit(X).labels_

X['cluster']= labelsDB

X['cluster'].value_counts()
X['cluster']= labelsDB

sample=X.sample(frac=0.1, replace=True, random_state=1)

# Display parallel coordinates plots, one for each cluster

display_parallel_coordinates(sample, 6)

#Grid Search for OPTICS based model

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import KFold

sc=make_scorer(adjusted_rand_score)

io={"parameter":{"xi": [.05,.01,.1] }}

ia={"parameter":{"min_cluster_size": [.05,.01,.1] }}



def customGrid(X, parXi, parameter=None):

    sam=X.sample(frac=0.1)

    lb= sam['cluster'] 

    sam= sam.drop('cluster',axis=1)

    average={}

    for k,v in parameter.items():

        

        for g in v:

            kw={k:g}

            if(parXi!=None):

                op = OPTICS(**kw,xi=parXi)

            else:

                op = OPTICS(**kw)

            kfold = KFold( n_splits=10)

            sco=[]



            for train_index, test_index in kfold.split(sam):

                train = sam.iloc[train_index]

                trainY=lb.iloc[train_index]

                test=sam.iloc[test_index]

                testY=lb.iloc[test_index]

                r1=op.fit(train).labels_

                r2=op.fit(test).labels_

                mm=adjusted_rand_score(r1,trainY)

                nn=adjusted_rand_score(r2,testY)

                mm=(mm+nn)/2

                sco.append(mm)

               

            ss=pd.DataFrame(sco).mean()

            

            average[g]=ss

    

    return average



def bestParams(firstPar):

    

    major=0

    sel=0

    for k,v in firstPar.items():

        if v.values>major:

            major=v.values

            sel=k

    print("Best params: "+ str(sel)+" : "+str(major))

    return sel

firstPar=customGrid(X,None, **io)

print("--------------------------------------------------")

sel= bestParams(firstPar)

dd=customGrid(X, sel,**ia)

y=bestParams(dd)
#Silhouette Score for OPTICS based model



sam1=X

lb1= sam1['cluster'] 

sam1=sam1.drop('cluster',axis=1)

ssc=silhouette_score(sam1,lb1)

ssc
lsample= X.sample(frac=0.3)

X_embedded = TSNE(n_components=2).fit_transform(lsample)

dataV=pd.DataFrame(X_embedded, columns=["D1","D2"])

dataV['cluster']=lsample['cluster'].tolist()

dataV=dataV.loc[dataV["cluster"]>-1]

fig = plt.figure(figsize=(10, 10))

ax = fig.add_subplot(111)

scatter = ax.scatter(dataV['D1'],dataV['D2'],

                  c=dataV['cluster'],   s=50)

ax.set_title('OPTICS Clustering')

ax.set_xlabel('Dimention 1')

ax.set_ylabel('Dimention 2')

plt.colorbar(scatter)