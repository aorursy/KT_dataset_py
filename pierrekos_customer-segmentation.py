# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
d=pd.read_csv("../input/data.csv", encoding='ISO-8859-1')

d.head()

d.info()
miss = []

for col in d.columns:

    i=d[col].isnull().sum()

    miss_v_p = i*100/d.shape[0]

    miss.append(miss_v_p)

    print ('{} -----> {}%'.format(col, 100-i*100/d.shape[0]))



dico = {'columns': d.columns, 'filling rate': 100-np.array(miss), 'taux nan': miss}

#print(miss, dico['taux de remplissage'])

tr=pd.DataFrame(dico)

r = range(tr.shape[0])

barWidth=0.85

plt.figure(figsize=(20,8))

plt.bar(r, tr['filling rate'], color='#a3acff', edgecolor='white', width=barWidth)

plt.bar(r, tr['taux nan'], bottom=tr['filling rate'], color ='#b5ffb9', edgecolor= 'white', width=barWidth)

plt.title('fill rate representation')

plt.xticks(r, tr['columns'], rotation='vertical')

plt.xlabel('columns')

plt.ylabel('filling rate')

plt.margins(0.01)
def count_lab(d, lab='Country'):

    """Compte le nombre d'éléments identique dans la colonne 'lab'"""

    r=d.groupby(lab).count()

    r['nb']=r.iloc[:,0]

    r.sort_values(by= 'nb', ascending=False, inplace=True)

    dico={lab: r.index, 'nb':r['nb']}

    return pd.DataFrame(dico)
label='Description'

g1=count_lab(d, lab=label).head(30)

#display(g1)

plt.figure(figsize=(8,30))

plt.barh(g1['Description'].apply(str), g1['nb'])

plt.gca().invert_yaxis()

plt.axvline(x=2000, color='b')

plt.text(2082, -1, '>2000', color='b')

plt.axvline(x=1000, color='r')

plt.text(780, -1, '<1000', color='r')

plt.grid(True)

plt.title('the 30 Most Present Products')
desc_price=d.loc[:,['Description', 'UnitPrice']].groupby('Description').mean()

import seaborn as sns



desc_price.boxplot('UnitPrice')

plt.title('unit price distribution')

plt.ylim((0,70))
print('Rq: Missing Customer ID and Negative Quantities transactions seem to be related to delivery issues:')

d[(d['CustomerID'].isnull())&(d['Quantity']<0)].sample(8)
print(d[d['CustomerID'].isnull()].shape, d.shape, )

d.dropna(axis=0, subset=['CustomerID'], inplace=True)
#the price of the transaction for a product

d['TotalPrice']=d['UnitPrice']*d['Quantity']



#The total of the purchase invoice

Invoice = d.loc[:,['InvoiceNo', 'TotalPrice']].groupby('InvoiceNo').sum()

Invoice.rename(columns={'TotalPrice':'InvoiceTotal'}, inplace=True)

d=d.merge(Invoice, on='InvoiceNo')#rq: Here, merge remove any missing values from the pivot category.



# the total amount spent in the DataFrame per customer

total_y = d.loc[:,['CustomerID', 'TotalPrice']].groupby('CustomerID').sum()

total_y.rename(columns={'TotalPrice': 'TotalYear'}, inplace=True)

d=d.merge(total_y, on='CustomerID')





d.sample(5)



"""Fonctions relatives au variables de coûts"""

def cost_info(d):

    """Add to 'd' informative variables on cost. In case these variables already exist in 'd', they are reset."""

    for col in ['TotalPrice', 'nb_inv', 'TotalYear', 'InvoiceTotal']:

        try:

            d.drop(columns=[col], inplace=True)

            print('Resetting the cost variable {} succeeded'.format(col))

        except:

            print('Column {} not present'.format(col))

    

    #prix du produit * quantité achetée

    d['TotalPrice']=d['UnitPrice']*d['Quantity']

    

    

    if type(d)==pd.Series:

        d=pd.DataFrame([d])

        

    

    #total par commande

    Invoice = d.loc[:,['InvoiceNo', 'TotalPrice']].groupby('InvoiceNo').sum()

    Invoice.rename(columns={'TotalPrice':'InvoiceTotal'}, inplace=True)

    d=d.merge(Invoice, on='InvoiceNo')

        

    #Total dépensé par le client

    total_y = d.loc[:,['CustomerID', 'TotalPrice']].groupby('CustomerID').sum()

    total_y.rename(columns={'TotalPrice': 'TotalYear'}, inplace=True)

    d=d.merge(total_y, on='CustomerID')

        

    #nombre de commandes sur l'année

    invoices=d.loc[:, ['CustomerID', 'InvoiceNo', 'TotalYear']].groupby('InvoiceNo').mean()

    nb_invoices = invoices.groupby('CustomerID').count()

    nb_invoices.rename(columns={'TotalYear':'nb_inv'}, inplace=True)

    d=d.merge(nb_invoices, on='CustomerID')

        

    

    return d

"""Fonctions relatives au variables de coûts"""

def cost_info(d):

    """Add to 'd' informative variables on cost. In case these variables already exist in 'd', they are reset."""

    for col in ['TotalPrice', 'nb_inv', 'TotalYear', 'InvoiceTotal']:

        try:

            d.drop(columns=[col], inplace=True)

            print('Resetting the cost variable {} succeeded'.format(col))

        except:

            print('Column {} not present'.format(col))

    

    #prix du produit * quantité achetée

    d['TotalPrice']=d['UnitPrice']*d['Quantity']

    

    

    if type(d)==pd.Series:

        d=pd.DataFrame([d])

        

    

    #total par commande

    Invoice = d.loc[:,['InvoiceNo', 'TotalPrice']].groupby('InvoiceNo').sum()

    Invoice.rename(columns={'TotalPrice':'InvoiceTotal'}, inplace=True)

    d=d.merge(Invoice, on='InvoiceNo')

        

    #Total dépensé par le client

    total_y = d.loc[:,['CustomerID', 'TotalPrice']].groupby('CustomerID').sum()

    total_y.rename(columns={'TotalPrice': 'TotalYear'}, inplace=True)

    d=d.merge(total_y, on='CustomerID')

        

    #nombre de commandes sur l'année

    invoices=d.loc[:, ['CustomerID', 'InvoiceNo', 'TotalYear']].groupby('InvoiceNo').mean()

    nb_invoices = invoices.groupby('CustomerID').count()

    nb_invoices.rename(columns={'TotalYear':'nb_inv'}, inplace=True)

    d=d.merge(nb_invoices, on='CustomerID')

        

    

    return d

import datetime as dt

d['InvoiceDate']=pd.to_datetime(d['InvoiceDate'])

now=d['InvoiceDate'].max()

now
d['date']=d['InvoiceDate'].apply(lambda x : x.date())

d['hour']=d['InvoiceDate'].apply(lambda x : x.time())
d_POS = d[d['Quantity']>0]#To not consider the refunds as transactions
invoices=d_POS.loc[:, ['CustomerID', 'InvoiceNo', 'TotalYear']].groupby('InvoiceNo').mean()

nb_invoices = invoices.groupby('CustomerID').count()

nb_invoices.rename(columns={'TotalYear':'nb_inv'}, inplace=True)

#display(nb_invoices)

nb_invoices.sort_values(by='nb_inv', ascending=False, inplace=True)
plt.hist(nb_invoices['nb_inv'], bins=200)#plt.plot(nb_invoices, '.')

plt.title('Number of customers for each number of orders')

plt.xlim((0,50))
d=d.merge(nb_invoices, on='CustomerID')

d.head()


first_date = d.loc[:,['CustomerID', 'InvoiceDate']].groupby('CustomerID').min().rename(columns={ 'InvoiceDate' : 'first_date'})

last_date = d_POS.loc[:,['CustomerID', 'InvoiceDate']].groupby('CustomerID').max().rename(columns={ 'InvoiceDate' : 'last_date'})# we use d_POS to not consider refunds

display(last_date.info())



date = first_date.merge(last_date, on='CustomerID')

#display(second_last)

#display(date)

date['since_last']=date['last_date'].apply(lambda x: x.date()-now.date())

date['since_first']=date['first_date'].apply(lambda x: x.date()-now.date())

date.head(2)
not_unique=date['since_last']-date['since_first']>dt.timedelta(days=0)





not_u=date.loc[not_unique,:]



d_not_u = pd.DataFrame(index=not_u.index)

#d_not_u['CustomerID']=not_u.index







invoice=d[d['Quantity']>0].groupby('InvoiceNo').nth(0)#We do not count refunds

invoice['InvoiceNo']=invoice.index





d_not_u=d_not_u.merge(invoice, on='CustomerID', how='inner')

d_not_u=d_not_u.sort_values(['CustomerID','InvoiceDate'], ascending=False)

second_last=d_not_u.loc[:,['CustomerID', 'InvoiceDate']].groupby('CustomerID').nth(1)



date=pd.concat([date, second_last.rename(columns={ 'InvoiceDate' : 'second_last'})], axis=1)

date['since_sec_last']=date['second_last'].apply(lambda x: x.date()-now.date())

date.head(2)
#Converting the number of days to integers

for col in ['since_last', 'since_first', 'since_sec_last']:

    date[col]=date[col].apply(lambda x: pd.to_timedelta(x).days)

date.head()   
nb_unique = date['second_last'].isnull().sum()

nb_cust= date.shape[0]

print("{}% of customers only made one involved during the year, {} customers on {}".format(round(nb_unique*100/nb_cust,2),

                                                                                                               nb_unique,

                                                                                                               nb_cust

                                                                                                              ))
nb_t_p=d_POS['Description'].nunique()

print('total number of products : ',nb_t_p)
#quantity of each product per customers

quantity = d_POS.loc[:,['CustomerID', 'Description', 'Quantity']].groupby(['CustomerID', 'Description']).sum()

col_ind = quantity.index.get_level_values(0)

quantity.reset_index(level=[0,1], inplace=True)



# Let add the Unit price per product

UnitPrice=d_POS.loc[:, ['Description', 'UnitPrice']].groupby('Description').mean()

quant_price = quantity.merge(UnitPrice, on='Description')



display(quant_price.head())



"""Becouse we focus on positive quantity, we must used the total monetary value per customer ('TotalYear') only on positive quantity : 

Refunds can skew our entropy, so we have to recalculate TotalYear on positive quantity only"""

for c in quantity['CustomerID'].unique():

    

    c_quanti = quant_price[quant_price['CustomerID']==c]

    date.loc[c,'quantity_t']=c_quanti.loc[:,'Quantity'].sum()

    date.loc[c,'nb_prod_diff']=c_quanti.shape[0]

    """cancellations can skew our entropy, so we have to recalculate TotalYear"""

    moyT = (c_quanti['Quantity']*c_quanti['UnitPrice']).sum()/date.loc[c,'nb_prod_diff']

    

    date.loc[c, 'entropy_corrige']=np.sqrt(np.square(np.log(1+c_quanti['Quantity']*c_quanti['UnitPrice'])-np.log(1+moyT)).sum()/date.loc[c,'nb_prod_diff'])

date['ind_div']=((date['nb_prod_diff'])/date['quantity_t'])
date.head()
def entropie_prod(d):

    d_POS=d[d['Quantity']>0]#Here we avoid the case where customer have a negative total quantity.

    

    #quantity of each product per customers

    quantity = d_POS.loc[:,['CustomerID', 'Description', 'Quantity']].groupby(['CustomerID', 'Description']).sum()

    col_ind = quantity.index.get_level_values(0)

    quantity.reset_index(level=[0,1], inplace=True)

    

    # Let add the Unit price per product

    UnitPrice=d_POS.loc[:, ['Description', 'UnitPrice']].groupby('Description').mean()

    quant_price = quantity.merge(UnitPrice, on='Description')

    

    # the DataFrame we 'll return

    date=pd.DataFrame(index=col_ind.unique(), columns=['quantity_t'])

    

    #Entropy Produit

    """Becouse we focus on positive quantity, we must used the total monetary value per customer ('TotalYear') only on positive quantity : 

Refunds can skew our entropy, so we have to recalculate TotalYear on positive quantity only"""

    for c in quantity['CustomerID'].unique():

    

        c_quanti = quant_price[quant_price['CustomerID']==c]

        date.loc[c,'quantity_t']=c_quanti.loc[:,'Quantity'].sum()

        date.loc[c,'nb_prod_diff']=c_quanti.shape[0]

        

        

        """cancellations can skew our entropy, so we have to recalculate TotalYear"""

        moyT = (c_quanti['Quantity']*c_quanti['UnitPrice']).sum()/date.loc[c,'nb_prod_diff']

    

        date.loc[c, 'entropy_corrige']=np.sqrt(np.square(np.log(1+c_quanti['Quantity']*c_quanti['UnitPrice'])-np.log(1+moyT)).sum()/date.loc[c,'nb_prod_diff'])

    

    #Diversity index

    date['ind_div']=((date['nb_prod_diff'])/date['quantity_t'])

    

    return date
from collections import Counter

import nltk
d_POS.head() #data restrict to positive quantity (not a refund or cancellation...)


def weights_words(d, label='Description', sep=" ", nb=None, min_occ=None):

    """

    Associate with each word present in the description of a product, a score that corresponds to the money spent by the customer in this product (quantity purchased * Unit price).

     Rq: In practice d will be the DataFrame for a client only.

     

    Associe à chaque mots presents dans la description d'un produit, un score qui correspond à l'argent dépenser du client dans ce produit ( quantité acheté * prix Unitaire).

    Rq: En pratique d sera le DataFrame pour un client uniquement.

    """

    #count=dict()

    words=[]

    for ind in d.index:

        #price=d.loc[ind, 'TotalPrice']

        #if

        words+=str(d.loc[ind, label]).split(sep)*int(d.loc[ind, 'TotalPrice']*10)#le rapport sera arrondie à 10 centimes près

        #else:

        #for w_neg in str(d.loc[ind, label]).split(sep)*int((-price)*10):

                #words=words.remove(w_neg)

       

    count=Counter(words)        

    if nb==None:

        if min_occ==None:

            return count

        else:

            c_words=pd.Series(count)

            rare = c_words[c_words<min_occ].index

            c_words.drop(index=rare, inplace=True)

            return dict(c_words)

    else:

        return(dict(count.most_common(nb)))
def df_cust_them(d):

    """

    Returns a Dataframe that each client associates a proportion for each keyword.

     This proportion represents the presence of the word on the customer's market share:

         Each product has a description in which keywords are extracted.

         These words are then weighted by the (price of the product) * (quantity purchased) / (total of the customer's expenses)

         I thus obtain a score between 0 and 1. which corresponds to the basis of the proportion of purchases of a product for the customer.

         As the products are broken down into words, I implicitly add up the scores (proportions) of the words that

         are found in several products.

         We have an indication of the interest (a posteriori) converted from the customer.

     Rq: The hypothetical case where score> 1 is theoretically possible if the word appears several times in the description of the same product

    

    

    Retourne un Dataframe qui a chaque client associe une proportion pour chaque mots clefs.

    Cette proportion représente la presence du mot sur la part de marché du client:

        A chaque produit correspond une description dans laquelle on extrait des mots clefs. 

        Ces mots sont ensuite pondérés par le (prix du produit)*(quantité achetée)/(total des dépenses du client)

        J'obtiens donc un score entre 0 et 1. qui correspond à la base aux proportions d'achats d'un produit pour le client.

        Les produits étant décomposés en mots, j'additionne implicitement les scores (proportions) des mot qui 

        se retrouvent dans plusieurs produit.

        On a une indication de l'interet (à posteriori) converti du client.

    Rq: Le cas hypothétique où score >1 est théoriquement possible ssi le mot apparait plusieur fois dans la description d'un meme produit.

    """

    dt=d[d['Quantity']>0]

    print('We keep positive quantities : ')

    dt=cost_info(dt)

    cust_them_dft = pd.DataFrame(index=dt['CustomerID'].unique())

    

    #Associate for each customer an affinity score on the most present words in the product description

    #Associe pour chaque client un score d'affinité sur les mots les plus présent dans la description des produits.

    for c in dt['CustomerID'].unique():

        d_cust = dt[dt['CustomerID']==c]

        y_total=d_cust.loc[:, 'TotalYear'].mean()

        

        l=weights_words(d_cust, min_occ=2)#None

        

        

        for lab in l.keys():

            cust_them_dft.loc[c, lab]=l[lab]

        cust_them_dft.loc[c, :]=cust_them_dft.loc[c, :]/(y_total*10)

        # Here I divide by the total quantities purchased from the customer

        #rq: The factor 10 has been added to get an accuracy of 10 cents.

        #Indeed the prices have been converted in integer * 10

    return cust_them_dft





def cust_them_clean(cust_them, w_all_min=16, affiner=True, keep_max=False):

    """Cleans the DataFrame 'cust_them' by removing unnecessary words (columns).

     it also removes columns with a total weight less than 'w_all_min'.

    """

    try:

        cust_them.drop(columns=[''], inplace=True)

    except:

        print("Column '' is not in the DataFrame")

        

    #deletes words that are not nouns

    is_not_noun = lambda pos: pos[:2]!='NN'

    not_noun=[]

    cust_them.columns = map(str.lower, cust_them.columns)

    l_w_pos = nltk.pos_tag(cust_them.columns)

    not_noun=[w for (w, pos) in l_w_pos if is_not_noun(pos)]    

    

    if affiner:

        #list of arbitrary words to add:

        l_words_int = ['photo' , 'girl', 'ceramic', 't-shirt', 

                   'origami', 'xmas', 'garden', 'gift', 'lantern', 'paint', 'marmalade', 

                   'poncho', 'bonbon', 'ivy', 'guitar', 'laser', 'boys', 'halloween', 'cloth', 't-light', 'baby', 'doormat']



    for a in l_words_int:

        try:

        

            not_noun.remove(a)

            

        except:

            print(a, 'n is not in the list')

    #List of words to remove (This corresponds to a retro active setting of words creating clusters when they are not significant enough (in the interpretation of the generated cluster)

    if keep_max==False:

        l_w_not_int=[]#'heart', 'retrospot', 'metal', 'design', 'holder', 'polkadot', 'regency','vintage',

        for r in l_w_not_int:

            not_noun.append(r)

    #commit:

    try:

        cust_them = cust_them.drop(columns=not_noun)

    except:

        print('word not remove')

    

    cust_them.fillna(0, inplace=True)

    

    #Nous retirons les mots les moins présent:

    w_importance = cust_them.sum().sort_values(ascending=False)

    w_importance = w_importance[w_importance>w_all_min]

    cust_them=cust_them.loc[:, w_importance.index]

    

    return cust_them





def for_cust_them(d, w_all_min=16, keep_max=False):

    them = df_cust_them(d)

    them_clean = cust_them_clean(them, w_all_min=w_all_min)

    return them_clean
%%time

cust_them = for_cust_them(d)
from sklearn import cluster, metrics
nb_clu = [i for i in range(3, 13)]

silh=[]

for n in nb_clu:

    km_init = cluster.KMeans(n_clusters=n, random_state=0)

    km_init.fit(cust_them)

    s=metrics.silhouette_score(cust_them, km_init.labels_)

    silh.append(s)

    print('OK',n)

    

plt.plot(range(3, 13), silh)

print(silh.index(max(silh))+3, max(silh))

print('7 clu :',silh[7-3])


n=8

km_init = cluster.KMeans(n_clusters=n, random_state=0)#4ou7 rds 6 8clu ou 10 clusters

km_init.fit(cust_them)



clusKM = pd.DataFrame(km_init.labels_, cust_them.index, columns=['km_them_t'])

#display(clusKM)

them_clu=pd.concat([clusKM, cust_them], axis=1)

#clusKM['CustomerID']=cust_them.index

#d=d.merge(clusKM, on='CustomerID')



them_clu.head()
dico_cl = dict()

for k in range(n):

    print('cluster ',k)

    temp=them_clu[them_clu['km_them_t']==k].describe().T

    temp=temp.iloc[1:, :]

    temp=temp[temp['50%']>0]

    display(temp)

    dico_cl[k]=temp

    l1 = plt.plot(temp['mean'])

    l2 = plt.plot(temp['50%'])

    l3 = plt.plot(temp['25%'])

    plt.legend(['mean','median', '25%'])

    plt.xticks(rotation=40)

    #plt.legend((l1, l2, l3), ('mean', 'median', '25%'))

    plt.show()
"""Let name these Clusters"""

#dico_them={6:'MetalSign', 1: 't-Light', 3:'Thermos', 2:'Bags', 0:'ChismasTime', 5:'Unspecified', 4:'RegencyTea', 7:'Doormat'}

dico_them={0:'RegencyTea', 1:'MetalSign', 2:'Unspecified', 3:'thermos', 4:'t-Light', 5:'Bags', 6:'HeartDeco', 7:'ChrismasVintage'}
from sklearn import decomposition
#n = nb of clusters

nmf=decomposition.NMF(n_components=n, random_state=0)

w=nmf.fit_transform(cust_them)

w_clean = pd.DataFrame(w, index=cust_them.index)
ordered_c=pd.DataFrame(km_init.predict(cust_them), index=cust_them.index, columns=['km_them'])

#display(ordered_c)

ref_nmf = pd.concat([ordered_c ,w_clean], axis=1, sort=True)

ref_nmf.head(1)
ref_nmf['legend']=ref_nmf['km_them'].apply(lambda x: dico_them[x])



import seaborn as sns

for i in range(7):

    plt.figure(figsize=(20,20))

    

    sns.pairplot(ref_nmf.loc[:,['km_them', 'legend', i, i+1]] , hue='legend', 

                 palette=sns.color_palette())  #"husl", 8             

    plt.title('plan factoriel {} et {}'.format(i+1, i+2))    

    plt.xlim((-0.001,0.5))

    plt.ylim((-0.001,0.5))

    plt.show()

    
l=['HeartDeco', 'Bags', 'ChrismasVintage', 'Thermos', 'Unspecified', 't-Light', 'MetalSign', 'RegencyTea']

#l=['Unspecified','t-Light', 'Bags', 'Thermos','RegencyTea', 'MetalSign','ChismasTime','DoormatChismas']

for i in range(8):

    ref_nmf.rename(columns={i : l[i]},inplace=True)

ref_nmf.head()   
d_clu = pd.concat([date.loc[:, ['entropy_corrige','ind_div']], ref_nmf.loc[:,l]], axis=1, sort=True)#'quantity_t','nb_prod_diff',
d_clu.head()
from sklearn.preprocessing import StandardScaler

scale_v=StandardScaler()

d_clu.loc[:,:]=scale_v.fit_transform(d_clu.loc[:,:])

print(d_clu.shape)

nb_clu = [i for i in range(3, 23)]

silh=[]

for n in nb_clu:

    km_gen= cluster.KMeans(n_clusters=n, random_state=1)

    km_gen.fit(d_clu)

    s=metrics.silhouette_score(d_clu, km_gen.labels_)

    silh.append(s)

    #print('OK pour ',n)

    

plt.plot(nb_clu, silh)

print(silh.index(max(silh))+3, 'silhouette :', max(silh))
n_cl=14

km_gen = cluster.KMeans(n_clusters=n_cl, random_state=1)#4ou6

km_gen.fit(d_clu)



#clusKM = pd.DataFrame(np.array([km_gen.labels_, d_clu.index]).T, columns=['categ', 'CustomerID'])

#them_clu=clusKM.merge(d_clu, on='CustomerID')



clusKM = pd.DataFrame(km_gen.labels_, index=d_clu.index, columns=['categ'])

them_clu=pd.concat([clusKM, d_clu], axis=1, sort=True)

#clusKM['CustomerID']=cust_them.index

#d=d.merge(clusKM, on='CustomerID')



them_clu.head()
for k in range(n_cl):

    print('cluster No.',k)

    temp=them_clu[them_clu['categ']==k].describe().T

    #temp.drop('CustomerID', inplace=True)

    temp=temp.iloc[1:, :]

    #temp=temp[temp['50%']>0]

    display(temp)

    plt.figure(figsize=(10,4))

    l1 = plt.plot(temp['mean'])

    l2 = plt.plot(temp['50%'])

    l3 = plt.plot(temp['25%'])

    l4 = plt.plot(temp['75%'])

    plt.axhline(y=0, c='black', ls='--')

    plt.xticks(rotation=40)

    plt.title('No.{} (nb = {})'.format(k, temp.iloc[0,0]))

    plt.legend(['mean', 'median', '25%', '75%'])

    plt.show()