#Copied

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #Cuz why not

import seaborn as sns #Not using it now



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#Lets just see the November data

df = pd.read_csv("/kaggle/input/ecommerce-events-history-in-cosmetics-shop/2019-Nov.csv")



#Formalities time conversion

df['event_time'] = pd.to_datetime(df['event_time'],infer_datetime_format=True)



##Encode? Lets not do that now. categorical/label encoding of the session IDs (instead of string - save memory/file size):

#Actually its new to me to know that it reduces the size so thanks

#df['user_session'] = df['user_session'].astype('category').cat.codes



#Check how many row and column

print(df.shape)

#Formalities see the topmost data

df.head()
df.info()
#Oh so this is how you count cat value occurence. That's neat

ev_count = df["event_type"].value_counts()

#Might as well draw a pie chart

#func to get percent string to show to  

def func(pct, allvals):

    absolute = int(pct/100.*np.sum(allvals))

    return "{:.1f}%\n({:d} entry)".format(pct, absolute)

ax = ev_count.plot.pie(figsize=(6,6),radius=2,autopct=lambda pct: func(pct, ev_count),textprops=dict(color="w"),legend=True)

ax.legend(loc=2)
#Might do the same to product_id and others

prid_s = df["product_id"].value_counts()

#Of course not, the product_id is long you'd waste space printing it 

print('no. of products: ',len(prid_s))

#Use describe instead

print(prid_s.describe())

#So what we count here is how active ea product is, it doesnt matter what the activity though might be only viewing

#carting purchasing and even removing. But we get the big picture here that how active each product was on Nov 2019

#So we gonna step up to see which is most popular

prid_s = prid_s.sort_values(ascending=False) 

print('Top 10 Active Products')

print(prid_s.head(10))

#Sadly we dont know what the real product which lies on top of the list

#Lets step up to see the product access count distribution (with histogram of course)

prid_s.plot.hist(bins=100)

#That doesnt look good right,try this

#prid_s.plot.hist(bins=[0,10,20,50,100,200,500,1000,2000,5000,10000,20000],logx=True,logy=True)
category_dict=df[['category_id','product_id']].drop_duplicates()

print('no. of category',len(df['category_id'].drop_duplicates()))

cat_counts=category_dict['category_id'].value_counts().sort_values(ascending=False)

print('Top 10 Category')

print(cat_counts.head(10))

print('Last 10 Category')

print(cat_counts.tail(10))

cat_counts.plot.bar(logy=True)
print(type(cat_counts)) #Check the type of cat_count oh no its a series

cat_act_c=df['category_id'].value_counts()

print(cat_act_c.head(10))

print(cat_counts.head(10))

print(len(cat_act_c.index))

print(len(cat_counts.index))

print(cat_act_c.index[:10])

print(cat_counts.index[:10])

#we can see the number of cat is same however it isnt aligned

#if we join

df_act_cat=pd.DataFrame(cat_act_c)

df_act_cat.rename(columns={'category_id':'activeness'}, inplace=True)

df_cnt_cat=pd.DataFrame(cat_counts)

df_cnt_cat.rename(columns={'category_id':'n_product'}, inplace=True)

print(df_act_cat.head(10))

print(df_cnt_cat.head(10))

joined = df_act_cat.join(df_cnt_cat)

joined.head(10)

# Oh it actually joined well because the index is already the category number

joined.plot.scatter(x='n_product',y='activeness',figsize=(16,8))
print(df['category_code'].value_counts())

print('allRows =',len(df.index))

print('nulls =',len(df.index)-sum(df['category_code'].value_counts()))

##df.drop(["event_time"],axis=1).nunique()
print(df['brand'].value_counts())

print('allRows =',len(df.index))

print('nulls =',len(df.index)-sum(df['brand'].value_counts()))

print(df['brand'].value_counts().head(20))

print('brand accessed ', sum(df['brand'].value_counts())/len(df.index)*100)
c_p_act = df['price'].value_counts()

c_p_act.sort_index(inplace=True)

print(c_p_act.head(10))

c_p_act.plot.line()# Huh there are negative price??? And zero prices oh come on wth
#Looks like a detective job here

print(df.loc[df['price']<0][['product_id','price']].drop_duplicates())
#So we're down to five lets see the transaction of each product

print(df.loc[df['product_id']==5716855][['event_time','event_type','product_id','price','user_id']])

#Straight purchase huh

print(df.loc[df['product_id']==5716859][['event_time','event_type','product_id','price','user_id']])

print(df.loc[df['product_id']==5716857][['event_time','event_type','product_id','price','user_id']])

print(df.loc[df['product_id']==5716861][['event_time','event_type','product_id','price','user_id']])

print(df.loc[df['product_id']==5670257][['event_time','event_type','product_id','price','user_id']])
print(len(df[['product_id','price']].drop_duplicates()))

print(df[['product_id','price']].drop_duplicates().sort_values(by='product_id').head(20))

print(df.drop(["event_time"],axis=1).nunique())

#oh there may be a change of price, so, does the aforementioned thing is pure error?

#Lets see the most changing product

price_list=df[['product_id','price']].drop_duplicates()

print(price_list['product_id'].value_counts().head(10))
#lets see how product 5900886 

print(df.loc[df['product_id']==5900886][['event_time','event_type','product_id','price','user_id']].iloc[100:150])
def day_far(series):

    time_span=series.max()-series.min()

    if time_span.days==0: 

        return 1

    else:

        return time_span.days

pvt_pp=pd.pivot_table(df,values=['event_time','event_type'],index=['product_id','price'],aggfunc={'event_time':day_far,'event_type':len})
pvt_pp['act_p_day']=pvt_pp['event_type']/pvt_pp['event_time']

print(pvt_pp.head(10))

pvt_pp.reset_index().plot.scatter(x='price',y='act_p_day',figsize=(16,8))
print(pvt_pp.head(10))

pvt_pp.loc[5900886].reset_index().plot.line(x='price',y='act_p_day',figsize=(16,8))

pvt_pp.loc[5906079].reset_index().plot.line(x='price',y='act_p_day',figsize=(16,8))

pvt_pp.loc[5816649].reset_index().plot.line(x='price',y='act_p_day',figsize=(16,8))

pvt_pp.loc[5788139].reset_index().plot.line(x='price',y='act_p_day',figsize=(16,8))

pvt_pp.loc[5900579].reset_index().plot.line(x='price',y='act_p_day',figsize=(16,8))

pvt_pp.loc[5901864].reset_index().plot.line(x='price',y='act_p_day',figsize=(16,8))

pvt_pp.loc[5900883].reset_index().plot.line(x='price',y='act_p_day',figsize=(16,8))
df['user_id'].value_counts()
#oh 527021202 is the most active user lets see whats done

#df.loc[(df['user_id']==527021202)&(df['event_time']>pd.to_datetime('2019-11-06'))&(df['event_time']<pd.to_datetime('2019-11-08'))].tail(60)

df.loc[(df['user_id']==527021202)&(df['event_type']=='purchase')].tail(60)

#Oh no the most active user doesnt purchase anything 
cross_usev=pd.crosstab(df['user_id'],df['event_type'])

print(cross_usev.sort_values(by='purchase',ascending=False).head(20))
#so 557790271 is seemingly the most information rich user, it also has a good proportion of each event type

#lets see when they did purchase 

df.loc[(df['user_id']==557790271)&(df['event_type']=='purchase')]

#big purchase list lets see whats going on 13 Nov

data_snippet = df.loc[(df['user_id']==557790271)&(df['event_time']>pd.to_datetime('2019-11-12'))&(df['event_time']<pd.to_datetime('2019-11-14'))]

#that what we typically want to see a session(multiple sessions actually) started with viewing and ended with purchase 

#im curious of how it went for each product

data_snippet.sort_values(by=['product_id','event_time']).head(31)
data_snippet.sort_values(by=['product_id','event_time']).tail(31)
fig, ax = plt.subplots()

cross_usev.plot(kind='scatter',x='view',y='cart',c='purchase',s=8,colormap='PiYG', figsize=(16,8), xlim=(0,4000), ylim=(0,1000), ax=ax)
#purchasers

purchaser=cross_usev.loc[cross_usev['purchase']>0]

print(purchaser.describe())

purchaser.plot(kind='scatter',x='view',y='cart',c='purchase',s=8,colormap='PiYG', figsize=(16,8), xlim=(0,4000), ylim=(0,1000))

big_shot=cross_usev.loc[cross_usev['purchase']>100]

big_shot.plot(kind='scatter',x='view',y='cart',c='purchase',s=20,colormap='PiYG', figsize=(16,8), xlim=(0,4000), ylim=(0,1000))

typical_purchaser=cross_usev.loc[(cross_usev['purchase']<20)&(cross_usev['purchase']>0)]

typical_purchaser.plot(kind='scatter',x='view',y='cart',c='purchase',s=20,colormap='PiYG', figsize=(16,8), xlim=(0,4000), ylim=(0,1000))

non_purchaser=cross_usev.loc[cross_usev['purchase']==0]

non_purchaser.plot(kind='scatter',x='view',y='cart',c='black',s=20, figsize=(16,8), xlim=(0,4000), ylim=(0,1000))
cross_usev['delta_cart']=cross_usev['cart']-cross_usev['remove_from_cart']

cross_usev.plot(kind='scatter',x='view',y='delta_cart',c='purchase',s=8,colormap='PiYG', figsize=(16,8), xlim=(0,4000), ylim=(-600,600))

typical_purchaser=cross_usev.loc[(cross_usev['purchase']<20)&(cross_usev['purchase']>0)]

typical_purchaser.plot(kind='scatter',x='view',y='delta_cart',c='purchase',s=20,colormap='PiYG', figsize=(16,8), xlim=(0,4000), ylim=(-600,600))

no_purchaser=cross_usev.loc[(cross_usev['purchase']==0)]

no_purchaser.plot(kind='scatter',x='view',y='delta_cart',c='black',s=20, figsize=(16,8), xlim=(0,4000), ylim=(-600,600))

print(df.drop(["event_time"],axis=1).nunique())

print(df[['product_id','price','brand']].loc[df['product_id']==5809910].drop_duplicates())
len(cross_usev.loc[cross_usev['purchase']>0].index)
# 1. Get Activity of each product

cross_prev = pd.crosstab(df['product_id'],df['event_type'])

print(cross_prev.head(5))

# 2. Get Price Average

pivot_prodprice = pd.pivot_table(df,index=['product_id'],values=['price'],aggfunc=np.mean)

print(pivot_prodprice.head(5))

# 3. Get Category

unique_cat_pr = df[['category_id','product_id']].drop_duplicates()

cat_size = unique_cat_pr['category_id'].value_counts()

print(cat_size.head(5))

# 4. Get number of unique user

unique_user_event=df[['product_id','user_id','event_type']].drop_duplicates()

cross_prevuser=pd.crosstab(unique_user_event['product_id'],unique_user_event['event_type'])

print(cross_prevuser.head(5))

# 5. Isbranded

df_prbr=df[['product_id','brand']].drop_duplicates()

print(len(df_prbr.index)) #oh there are some product that brand added later

df_prbranded=df.loc[df['brand'].notnull(),['product_id']].drop_duplicates()

np_prbrand=df_prbranded.to_numpy()

print(len(df_prbranded.index)/43419)

df_prcat = df[['product_id','category_id']].drop_duplicates()

#trytrylah = np.where((df_prcat['product_id'] in [232423,121212]), True, False)

def isBrandPoduct(series):

    if series['product_id'] in np_prbrand:

        return True

    else:

        return False

df_prcat['isBranded'] = df_prcat.apply(isBrandPoduct,axis='columns')    

print(df_prcat.head(25))

# Wow so long for just labeling this, help me shorten this plz

# now lets jjjjoin
#Use df_prcat as the base

join_1 = df_prcat.join(cross_prev,on='product_id')

#print(join_1.head(5))

#print(cross_prev.loc[[5802432,5844397,5837166,5876812,5826182]])

#for k in [5802432,5844397,5837166,5876812,5826182]:

#    print(df_prcat.loc[df_prcat['product_id']==k])

join_2 = join_1.join(pivot_prodprice,on='product_id')

join_2.rename(columns={"price": "avg_price"},inplace=True)

#print(join_2.head(5))

join_3 = join_2.join(cat_size,on='category_id',rsuffix='_new')

join_3.rename(columns={"category_id_new": "competitor"},inplace=True)

#print(join_3.head(5))

join_4 = join_3.join(cross_prevuser,on='product_id',rsuffix='_user')

#print(join_4.head(5))

#print(cross_prev.loc[5802432])

#print(cross_prevuser.loc[5802432])

join_4.to_csv("product_score.csv.gz",index=False,compression="gzip")
#Time is a bit confusing

#lets experiment a little bit

#remember this

#def day_far(series):

#    time_span=series.max()-series.min()

#    if time_span.days==0: 

#        return 1

#    else:

#        return time_span.days

span = df['event_time'].max()-df['event_time'].min()

print(type(span))

# so type is pandas Timedelta

print(span.value,'nanoseconds')

print(span.days,'days')

print(span.seconds,'seconds')

actual_seconds=span.value/1000000000

print('actual_seconds',actual_seconds)

#how many seconds a day?

day_sec=60*60*24

#span second should be same as

second_left = actual_seconds%day_sec

print('seconds_left',second_left)

#on the mark

sekon=span.seconds%60

minut=((span.seconds-sekon)/60)%60

hourz=(span.seconds-sekon-60*minut)/3600

print('so there should be',hourz,'hours',minut,'minute',sekon,'seconds')
#Now Getting easy customer scores



# 1. Get Activity of each customer

cross_usev = pd.crosstab(df['user_id'],df['event_type'])

print(cross_usev.head(5))

# 2. Get n unique product

unique_us_pr = df[['user_id','product_id']].drop_duplicates()

user_reach = unique_us_pr['user_id'].value_counts()

print(user_reach.head(5))

# 3. Get unique product on each event

unique_us_pr_ev=df[['user_id','product_id','event_type']].drop_duplicates()

cross_usevprod=pd.crosstab(unique_us_pr_ev['user_id'],unique_user_event['event_type'])

print(cross_usevprod.head(5))

# 4. Get n unique category

unique_us_cat = df[['user_id','category_id']].drop_duplicates()

user_diver = unique_us_cat['user_id'].value_counts()

print(user_diver.head(5))

# 5. Get unique category on each event

unique_us_cat_ev=df[['user_id','category_id','event_type']].drop_duplicates()

cross_usevcat=pd.crosstab(unique_us_cat_ev['user_id'],unique_user_event['event_type'])

print(cross_usevcat.head(5))