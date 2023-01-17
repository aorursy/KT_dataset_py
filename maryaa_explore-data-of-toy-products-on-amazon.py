# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import pylab

from wordcloud import WordCloud

sns.set_style("whitegrid")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df =  pd.read_csv('../input/amazon_co-ecommerce_sample.csv')

df.head(2)
df.dtypes
df.shape
df['number_available_in_stock'][0]
df['customers_who_bought_this_item_also_bought'][0]
df['product_description'][0]
df['items_customers_buy_after_viewing_this_item'][0]
df['customer_questions_and_answers'][2]
df['sellers'][0]
df.describe()
#---------------------------------------

# clean functions

#---------------------------------------



# change the price column to float

# usage:  df['price'] = df['price'].map(mapprice)

def mapprice(v): 

    if pd.isnull(v):

        return  [ 0, 0]

    try:

        vv = v.split('-')



        if(len(vv)==1):

            p0 = vv[0].strip()[1:].replace(",","")

            p1 = p0

        else: 

            p0 = vv[0].strip()[1:].replace(",","")

            p1 = vv[1].strip()[1:].replace(",","")     

        return  [float(p0),float(p1)]

    except ValueError:

        return  [ 0, 0]

    



# change the price column to float

# dx = df[0:2]

# print(pd.isnull(dx.number_available_in_stock.head(1)))

# test: dfclone.number_available_in_stock.str.split('\xa0')

# usage:  df['count1'],df['class1'] = df['number_available_in_stock'].map(mapnumber_available_in_stock)[0]

def mapnumber_available_in_stock(v): 

    if pd.isnull(v):

        return np.NaN  ,np.NaN  

    try:

        vv = v.split('\xa0')

        return int(vv[0]),vv[1]

    except ValueError:        

        return np.NaN ,np.NaN    



    

def mapnumber_of_reviews(v): 

    if pd.isnull(v):

        return 0

    try:

        vv = v.replace(",","")

        return int(vv)

    except ValueError:

        return 0



    

def mapaverage_review_rating(v): 

    if pd.isnull(v):

        return 0

    try:

        vv = v.split('out')[0][0].strip()

        return float(vv)

    except ValueError:        

        return 0    

    

# read json data of seller    

def myseller(v):

    if pd.isnull(v):

        return 0

    try:

        vv = v.replace('=>',':')

        djson = pd.read_json(vv,orient='records')   

        return djson

    except ValueError:        

        return 0      



#split category    

def mapcategories(srs):

    if pd.isnull(srs):

        return []

    else:

        return [cat.strip() for cat in srs.split(">")]    



    

#---------------------------------------

# plot functions

#---------------------------------------



#example: sns.countplot(df['number_of_reviews'],ax=ax[0],palette='Set1').set_title('number_of_reviews')

def mycountplot(col1,idx1):

    plt.subplot(idx1)

    sns.countplot(df[col1],  palette='Set1') #.set_title(col1)

    

def myjitterplot(col,idx1):

    plt.subplot(idx1)

    sns.stripplot(data = df, y = col, jitter = True,  color='goldenrod')

    plt.xlabel(col, fontsize=10)

    plt.ylabel('Cases', fontsize=10)   

    

# generate the word cloud    

def wordcloudify( dataset):

    return WordCloud().generate(''.join(dataset))



def mywordcloudshow2(idx,key,value):

    data = [item.strip() for item in df[key].dropna()]

    axarr[idx].imshow(wordcloudify(data), 

                    interpolation='nearest', aspect='auto')

    axarr[idx].axis("off")

    axarr[idx].set_title(value, fontsize=16)     

    

def wordcloudifycat(cat):

    return WordCloud().generate(

        " ".join(products.query('leaf_category == "{0}"'.format(cat))['product_name'].values)

    )





def mywordcloudshowcat(idx,key,value):

    # Display the generated image:

    axarr[idx].imshow(wordcloudifycat(key), 

                    interpolation='nearest', aspect='auto')

    axarr[idx].axis("off")

    axarr[idx].set_title(value, fontsize=16)

        

    
foo = lambda x: pd.Series([i for i in mapnumber_available_in_stock(x)])

dfin_stock= df.loc[:]['number_available_in_stock']

dfin_stock.head(5)

rev = dfin_stock.apply(foo)

rev.columns = ['inumber_available_in_stock','class_available_in_stock']

df['inumber_available_in_stock'],df['class_available_in_stock']  = rev['inumber_available_in_stock'],rev['class_available_in_stock']

df.head(2)
foo = lambda x: pd.Series([i for i in mapprice(x)])

dfprice = df.loc[:]['price']

rev = dfprice.apply(foo)

rev.columns = ['pricemin','pricemax']

df['pricemin'] = rev['pricemin']

df['pricemax'] = rev['pricemax']

df['pricerange'] = rev['pricemax'] - rev['pricemin'] 
df.iloc[3193]
# Reformatting the number_of_reviews and price columns.

# exception data: df[df['number_of_reviews'].str.contains(',') == True]

if df['number_of_reviews'].dtype != 'int64':  

    df['number_of_reviews'] = df['number_of_reviews'].map(mapnumber_of_reviews)



# check sample data

# df.iloc[9315]['number_of_reviews']
if df['average_review_rating'].dtype != 'float64':

    df['average_review_rating'] = df['average_review_rating'].map(mapaverage_review_rating) 
dfbuyafter = df.loc[:,['uniq_id','items_customers_buy_after_viewing_this_item']].drop('items_customers_buy_after_viewing_this_item', axis=1)

dfbuyafter = dfbuyafter.join(df['items_customers_buy_after_viewing_this_item'].str.split('|', expand=True).stack().map(str.strip).reset_index(level=1, drop=True).rename('items_customers_buy_after_viewing_this_item'))

dfbuyafter.shape
dfbuyalso = df.loc[:,['uniq_id','customers_who_bought_this_item_also_bought']].drop('customers_who_bought_this_item_also_bought', axis=1)

dfbuyalso = dfbuyalso.join(df['customers_who_bought_this_item_also_bought'].str.split('|', expand=True).stack().map(str.strip).reset_index(level=1, drop=True).rename('customers_who_bought_this_item_also_bought'))

dfbuyalso.shape
dfx = df['sellers'][0:2][1]

dfx
#split the sellers data, put it into another 

dx = df['sellers'].map(myseller)[0]

for i,dy in enumerate(dx.seller):

    #print(dy)

    for k,v in dy.items():

        print(k[0:-2],v)

    break         
plt.subplots(figsize=(12,10))

toplot = ["pricemin","inumber_available_in_stock","number_of_reviews","number_of_answered_questions" ]

plotrc = 221;# 2*2

for idx, col in enumerate(toplot): 

     rc = plotrc + idx;

     myjitterplot(col,rc)

 

plt.show()
df.plot.kde()

plt.show()
#fig,ax=plt.subplots(2,2,figsize=(12,10))   

plt.subplots(figsize=(12,10))

toplot = ["class_available_in_stock", 

          "average_review_rating", "number_of_answered_questions",'pricerange']

plotrc = 321;# 2*2

for idx, col in enumerate(toplot): 

     rc = plotrc + idx;

     mycountplot(col,rc)

  

plt.ylabel('')

plt.xticks(rotation=90)

plt.show()
df_count_buyalso = dfbuyalso.groupby('uniq_id')['customers_who_bought_this_item_also_bought'].count().reset_index()

max_df_count_buyalso = df_count_buyalso['customers_who_bought_this_item_also_bought'].max()

#df_top_products_buyalso  = df.loc[:,['uniq_id','product_name','manufacturer','price']]

df_top_products_buyalso = df[df['uniq_id'].isin(

    df_count_buyalso[df_count_buyalso.customers_who_bought_this_item_also_bought == 

       max_df_count_buyalso]['uniq_id'])]



print('Top introducers when being bought: ',df_top_products_buyalso.shape[0],' products')

df_top_products_buyalso[['uniq_id','product_name']]
df_item_also_bought_all = dfbuyalso['customers_who_bought_this_item_also_bought'].value_counts()



print('Top 20 items that customers who bought this item also bought:\n',df_item_also_bought_all[0:20])
df_count_buyafter = dfbuyafter.groupby('uniq_id')['items_customers_buy_after_viewing_this_item'].count().reset_index()

df_count_buyafter['items_customers_buy_after_viewing_this_item'].value_counts()
df_item_also_bought_all = df_item_also_bought_all.reset_index()

plt.subplots(figsize=(7,3))

sns.stripplot(data = df_item_also_bought_all, y = "customers_who_bought_this_item_also_bought", jitter = True,  color='goldenrod')

plt.xlabel("items", fontsize=10)

plt.ylabel('Chances', fontsize=10)       

plt.title('Distribution of Chances of items that customers who bought a product also bought', fontsize=10)       

plt.show()
df_item_buyafter_all = dfbuyafter['items_customers_buy_after_viewing_this_item'].value_counts().reset_index()

plt.subplots(figsize=(7,3))

sns.stripplot(data = df_item_buyafter_all, y = "items_customers_buy_after_viewing_this_item", jitter = True,  color='goldenrod')

plt.xlabel("items", fontsize=10)

plt.ylabel('Chances', fontsize=10)       

plt.title('Distribution of Chances of items that customers who bought after viewing another product', fontsize=10)       

plt.show()
plt.subplots(figsize=(7,3))

dfbuyafter['items_customers_buy_after_viewing_this_item'].value_counts()[0:20].plot.barh(width=0.9,color='goldenrod')

plt.title('Top 20 items that customers bought after viewing a product')

plt.show()
plt.subplots(figsize=(7,3))

dfbuyalso['customers_who_bought_this_item_also_bought'].value_counts()[0:20].plot.barh(width=0.9,color='goldenrod')

plt.title('Top 20 items that customers who bought an item also bought')

plt.show()
plt.subplots(figsize=(7,3))

sns.stripplot(data = df_count_buyalso, y = "customers_who_bought_this_item_also_bought", jitter = True,  color='goldenrod')

plt.xlabel("products", fontsize=10)

plt.ylabel('Counts', fontsize=10)       

plt.title('Distribution of counts of items that customers who bought this item also bought', fontsize=10)       

plt.show()
plt.subplots(figsize=(7,3))

sns.stripplot(data = df_count_buyafter, y = "items_customers_buy_after_viewing_this_item", jitter = True,  color='goldenrod')

plt.xlabel("products", fontsize=10)

plt.ylabel('Counts', fontsize=10)       

plt.title('Distribution of counts of items that customers buy after viewing this item', fontsize=10)       

plt.show()
f, ax = plt.subplots(figsize=(7, 3))

sns.countplot(x="items_customers_buy_after_viewing_this_item", data=df_count_buyafter, color='goldenrod')

plt.title('Counts of items customers buy after viewing this item')

plt.xlabel('Count of items that customers buy after viewing a product')

plt.ylabel('Count of this kind of products')

plt.show()
f, ax = plt.subplots(figsize=(7, 3))

sns.countplot(x="customers_who_bought_this_item_also_bought", data=df_count_buyalso, color='goldenrod')

plt.title('Counts of items customers also bought after bought this item')

plt.xlabel('Count of items that customers also bought after bought a product')

plt.ylabel('Count of this kind of products')

plt.show()
dfx = df['pricemin'].sort_values(axis=0,ascending = False)[0:10]

df[df.index.isin(dfx.index)][['manufacturer','price','number_available_in_stock','number_of_reviews','number_of_answered_questions','average_review_rating']]
g = sns.PairGrid(df, y_vars=["pricemin"], x_vars=["average_review_rating", "number_of_reviews","inumber_available_in_stock"], size=4)

g.map(sns.regplot, color=".3")

g.set(ylim=(-1, 1000))

plt.title('Distribution regarding price < 1000')

plt.show()
sns.lmplot(x="number_of_reviews", y="pricemin", hue="average_review_rating", data=df)

plt.show()
sns.lmplot(x="number_of_reviews", y="number_of_answered_questions",  data=df)

plt.title('Correlation of number_of_reviews and number_of_answered_questions')

plt.show()
dfx = df[["pricemin","inumber_available_in_stock"]].query('pricemin <= 10').query('pricemin > 0').query('inumber_available_in_stock < 50')

g = sns.jointplot(x="pricemin", y="inumber_available_in_stock", data=dfx, kind="kde", color="m",axis=0)

g.plot_joint(plt.scatter, c="w", s=30, linewidth=0.2, marker=".")

g.ax_joint.collections[0].set_alpha(0.5)



g.set_axis_labels("$pricemin$", "$number_available_in_stock$")

plt.title('Just a graph of a small part of the data')

plt.show()
sns.stripplot(data = df, y = "inumber_available_in_stock", jitter = True,  color='goldenrod')

plt.xlabel("Products", fontsize=10)

plt.ylabel('number', fontsize=10)       

plt.title('Distribution of number available in stock', fontsize=10)       

plt.show()
sns.stripplot(data = df, y = "number_of_reviews", jitter = True,  color='goldenrod')

plt.xlabel("Products", fontsize=10)

plt.ylabel('number', fontsize=10)       

plt.title('Distribution of number_of_reviews', fontsize=10)       

plt.show()
df_soldbest = df[df["number_of_reviews"]>400]

plt.subplots(figsize=(12,4))

sns.stripplot(y="number_of_reviews", x="pricemin", data=df_soldbest)

plt.show()
plt.subplots(figsize=(18,6))

sns.stripplot(x="number_of_reviews", y="pricemin", data=df);

plt.title('Distribution of price and number of reviews')

plt.ylabel('number_of_reviews', fontsize=12)

plt.xlabel('pricemin', fontsize=12)

plt.xticks(rotation=90)

plt.show()
sns.set()



r = df['number_of_reviews']

dfr = pd.DataFrame({'r': r,  '4 Times': 4 * r ,'Current': r})



# Convert the dataframe to long-form or "tidy" format

dfr = pd.melt(dfr, id_vars=['r'], var_name='number_of_reviews', value_name='theta')



# Set up a grid of axes with a polar projection

g = sns.FacetGrid(dfr, col="number_of_reviews", hue="number_of_reviews",

                  subplot_kws=dict(projection='polar'), size=4.5,

                  sharex=False, sharey=False, despine=False)



# Draw a scatterplot onto each axes in the grid

g.map(plt.scatter, "theta", "r")

plt.show()
df.groupby('manufacturer')['pricemin'].mean().plot(figsize=(12,6),color='goldenrod')

plt.show()
top_expensive_manufacturer = df.groupby('manufacturer')['pricemin'].mean().sort_values(axis = 0,ascending=False) [0:20]

df_top_expensive_manufacturer = df[df['manufacturer'].isin(top_expensive_manufacturer.index)]

plt.subplots(figsize=(12,6))

plt.subplot(221)

sns.stripplot(x="manufacturer", y="number_of_reviews", data=df_top_expensive_manufacturer)

plt.xticks(rotation=90)

plt.ylabel('Number of reviews')

plt.subplot(222)

sns.stripplot(x="manufacturer", y="pricemin", data=df_top_expensive_manufacturer)

plt.xticks(rotation=90)

plt.ylabel('price')

plt.subplot(223)

sns.stripplot(x="manufacturer", y="inumber_available_in_stock", data=df_top_expensive_manufacturer)

plt.xticks(rotation=90)

plt.ylabel('number available in stock')

plt.subplot(224)

sns.stripplot(x="manufacturer", y="average_review_rating", data=df_top_expensive_manufacturer)

plt.xticks(rotation=90)

plt.ylabel('average review rating')

plt.show()
df['manufacturer'].value_counts()[:10].plot(kind='pie',figsize=(10,6),autopct='%1.1f%%',shadow=True,explode=[0.1,0,0,0,0,0,0,0,0,0])

plt.title('Distribution Of Top 10 manufacturers')

plt.show()
df.groupby('manufacturer')['uniq_id'].count().sort_values(ascending=False)[0:50].plot(kind='bar',figsize=(12,6),color='goldenrod')

plt.title('Product count per Top 50 manufacturers')

plt.show()
top_kindsofproducts_manufacturer = df['manufacturer'].value_counts(ascending=False)[0:10]

print("Manufacturers provides largest number of products:\n",top_kindsofproducts_manufacturer)
df_top_manufacturer = df[df['manufacturer'].isin(top_kindsofproducts_manufacturer.index)]

plt.subplots(figsize=(12,6))

plt.subplot(221)

sns.stripplot(x="manufacturer", y="number_of_reviews", data=df_top_manufacturer)

plt.xticks(rotation=90)

plt.ylabel('Number of reviews')

plt.subplot(222)

sns.stripplot(x="manufacturer", y="pricemin", data=df_top_manufacturer)

plt.xticks(rotation=90)

plt.ylabel('price')

plt.subplot(223)

sns.stripplot(x="manufacturer", y="inumber_available_in_stock", data=df_top_manufacturer)

plt.xticks(rotation=90)

plt.ylabel('number available in stock')

plt.subplot(224)

sns.stripplot(x="manufacturer", y="average_review_rating", data=df_top_manufacturer)

plt.xticks(rotation=90)

plt.ylabel('average review rating')

plt.show()



# should adjust the verticle space between graphs
plt.subplots(figsize=(12,20))

max_manufacturer=df.groupby('manufacturer')['manufacturer'].count()

max_manufacturer.sort_values(ascending=False,inplace=True)

mean_df=df[df['manufacturer'].isin(max_manufacturer[0:10].index)]

abc=mean_df.groupby(['manufacturer','number_of_reviews'])['inumber_available_in_stock'].mean().reset_index() 

abc=abc.pivot('number_of_reviews','manufacturer','inumber_available_in_stock')

sns.heatmap(abc,annot=True,cmap='RdYlGn',linewidths=0.4)

 





plt.title('Top 10 Manufacturers with number_of_reviews,average number available_in_stock')



plt.show()
df['sellers'].head(2)
df_seller = df['sellers'].map(myseller)

df_seller[0]
df_cat = df['amazon_category_and_sub_category'].value_counts().sort_values(ascending=False)[0:10]

df_cat.plot.barh(width=0.9,color='goldenrod')

fig=plt.gcf() 

fig.set_size_inches(12,5)



plt.xlabel('Counts')

plt.title('Counts of items per category')

plt.show()
# Word cloud of some attributes



toplot = {'customer_reviews':'customer reviews',

          'customer_questions_and_answers':'customer questions and answers',

          'description':'description',

          'product_information':'product information',

          'product_description':'product description',

          'items_customers_buy_after_viewing_this_item':'items customers buy after viewing this item'

         }



  

f, axarr = plt.subplots(len(toplot), 1, figsize=(8, 10))

# f.subplots_adjust(hspace=1)



ii = 0

for key, value in toplot.items():      

    mywordcloudshow2(ii,key,value)

    ii = ii +1

    

    

plt.show()