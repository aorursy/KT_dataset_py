import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns 

import squarify

import matplotlib

import re

from wordcloud import WordCloud, STOPWORDS
zomato_df = pd.read_csv('/kaggle/input/zomato-bangalore-restaurants/zomato.csv')
zomato_df.head()
zomato_df.info()
g = zomato_df.groupby('address')

g1 = g.filter(lambda x: len(x) > 1)[g.filter(lambda x: len(x) > 1)['name']=='Jalsa']

g1[g.filter(lambda x: len(x) > 1)[g.filter(lambda x: len(x) > 1)['name']=='Jalsa'].address=='942, 21st Main Road, 2nd Stage, Banashankari, Bangalore']
zomato_df1 = zomato_df.drop(['url', 'listed_in(type)', 'listed_in(city)'], axis=1).reset_index().drop(['index'], axis=1).drop_duplicates().copy()
plt.figure(figsize=(10, 7))

sns.set_style('white')

restaurants = zomato_df1.groupby(['address','name'])

chains= restaurants.name.nunique().index.to_frame()['name'].value_counts()[:15]

ax = sns.barplot(x= chains, y = chains.index, palette='Blues_d')

sns.despine()

plt.title('Top 15 restaurant chains in Bangalore')

plt.xlabel('Number of outlets')

plt.ylabel('Name of restaurants')

for p in ax.patches:

    width = p.get_width()

    ax.text(width+0.007, p.get_y() + p.get_height() / 2. + 0.2, format(width), 

            ha="left", color='black')

plt.show()
#Preprocessing cuisines

cuisines_p = zomato_df1.groupby(['address','cuisines']).cuisines.nunique().index.to_frame()

tmp = pd.DataFrame()

tmp = cuisines_p.cuisines.str.strip().str.split(',', expand=True)
cuisines=pd.DataFrame()

cuisines=pd.concat([tmp.iloc[:,0].str.strip(), tmp.iloc[:,1].str.strip(), tmp.iloc[:,2].str.strip(), tmp.iloc[:,3].str.strip(), tmp.iloc[:,4].str.strip(), tmp.iloc[:,5].str.strip(), tmp.iloc[:,6].str.strip(), tmp.iloc[:,7].str.strip() ]).value_counts()
plt.figure(figsize=(10, 7))

sns.set_style('white')

cuisine= cuisines[:15]

ax = sns.barplot(x= cuisine, y = cuisine.index, palette='Blues_d')

sns.despine()

plt.title('Top 15 cuisines served in Bangalore')

plt.xlabel('Number of restaurants')

plt.ylabel('Name of cuisines')

total = len(cuisines_p)

for p in ax.patches:

        percentage = '{:.1f}%'.format(100 * p.get_width()/total)

        x = p.get_x() + p.get_width() + 0.02

        y = p.get_y() + p.get_height()/2

        ax.annotate(percentage, (x, y),

        ha="left", color='black')

plt.show()
cuisines[-15:]
cuisines[cuisines.index=='Healthy Food']
#Preprocessing Restaurant Types

rest_t = zomato_df1.groupby(['address','rest_type']).rest_type.nunique().index.to_frame()

tmp_r = pd.DataFrame()

tmp_r = rest_t.rest_type.str.strip().str.split(',', expand=True)

tmp_r.shape
rest_types=pd.DataFrame()

rest_types=pd.concat([tmp_r.iloc[:,0].str.strip(), tmp_r.iloc[:,1].str.strip()]).value_counts()

rest_types
plt.figure(figsize=(10, 7))

sns.set_style('white')

restaurant_types = rest_types[:15]

ax = sns.barplot(x= restaurant_types, y = restaurant_types.index, palette='Blues_d')

sns.despine()

plt.title('Top 15 restaurant types in Bangalore')

plt.xlabel('Number of restaurants')

plt.ylabel('Restaurant types')

total = len(rest_t)

for p in ax.patches:

        percentage = '{:.1f}%'.format(100 * p.get_width()/total)

        x = p.get_x() + p.get_width() + 0.02

        y = p.get_y() + p.get_height()/2

        ax.annotate(percentage, (x, y),

        ha="left", color='black')

plt.show()
loc_t = zomato_df1.groupby(['address','location']).location.nunique().index.to_frame()

print (loc_t['location'].value_counts()[loc_t['location'].value_counts().index.str.contains('Koramangala')]

,"\n","Total number of restaurants in Koramangala",sum(loc_t['location'].value_counts()[loc_t['location'].value_counts().index.str.contains('Koramangala')]))
plt.figure(figsize=(10, 7))

sns.set_style('white')

locations= loc_t['location'].value_counts()[:15]

ax = sns.barplot(x= locations, y = locations.index, palette='Blues_d')

sns.despine()

plt.title('Top 15 locations for foodies in Bangalore')

plt.xlabel('Number of restaurants')

plt.ylabel('Name of Location')

for p in ax.patches:

    width = p.get_width()

    ax.text(width+0.007, p.get_y() + p.get_height() / 2. + 0.2, format(width), 

            ha="left", color='black')

plt.show()
df_1=zomato_df1.groupby(['location','cuisines']).agg('count')

data=df_1.sort_values(['address'],ascending=False).groupby(['location'],

                as_index=False).apply(lambda x : x.sort_values(by="address",ascending=False).head(3))['address'].reset_index().rename(columns={'address':'count'})
data.tail(10)
#Preprocessing Dish liked

dish_t = zomato_df1.groupby(['address','dish_liked']).dish_liked.nunique().index.to_frame()

tmp_d = pd.DataFrame()

tmp_d = dish_t.dish_liked.str.strip().str.split(',', expand=True)

tmp_d.shape
dish_liked=pd.DataFrame()

dish_liked=pd.concat([tmp_d.iloc[:,0].str.strip(), tmp_d.iloc[:,1].str.strip()]).value_counts()

dish_liked
plt.figure(figsize=(10, 7))

sns.set_style('white')

dishes = dish_liked[:15]

ax = sns.barplot(x= dishes, y = dishes.index, palette='Blues_d')

sns.despine()

plt.title('Top 15 commonly served dishes in Bangalore')

plt.xlabel('Number of restaurants')

plt.ylabel('Name of dishes')

total = len(dish_t)



for p in ax.patches:

        percentage = '{:.1f}%'.format(100 * p.get_width()/total)

        x = p.get_x() + p.get_width() + 0.02

        y = p.get_y() + p.get_height()/2

        ax.annotate(percentage, (x, y),

        ha="left", color='black')

plt.show()

#Utilise matplotlib to scale our goal numbers between the min and max, then assign this scale to our values.



def pre_dish_liked(restaurant_type):

	#Preprocessing Dish liked

	dish_rest_type = zomato_df1.groupby(['address','dish_liked','rest_type']).dish_liked.nunique().index.to_frame()

	tmp_d = pd.DataFrame()

	tmp_d = dish_rest_type[dish_rest_type['rest_type']==restaurant_type].dish_liked.str.strip().str.split(',', expand=True)

	dish_liked=pd.DataFrame()

	dish_liked=pd.concat([tmp_d.iloc[:,0].str.strip(), tmp_d.iloc[:,1].str.strip()]).value_counts()

	df = pd.DataFrame({'nb_people':dish_liked[:10], 'group': dish_liked[:10].index})

	

	norm = matplotlib.colors.Normalize(vmin=min(dish_liked[:10]), vmax=max(dish_liked[:10]))

	colors = [matplotlib.cm.Blues(norm(value)) for value in dish_liked[:10]]

  	

  	



  

	squarify.plot(sizes=df['nb_people'], label=df['group'], alpha=.8, color = colors )

	plt.title("Top 10 dishes served in "+restaurant_type,fontsize=15,fontweight="bold")

 

	plt.axis('off')

	plt.show() 
pre_dish_liked('Quick Bites')

pre_dish_liked('Casual Dining')

pre_dish_liked('Cafe')

pre_dish_liked('Delivery')

pre_dish_liked('Dessert Parlor')
df_1=zomato_df1.groupby(['rest_type','name']).agg('count')

datas=df_1.sort_values(['address'],ascending=False).groupby(['rest_type'],

                as_index=False).apply(lambda x : x.sort_values(by="address",ascending=False).head(5))['address'].reset_index().rename(columns={'address':'count'})

datas
all_ratings = []



for name,ratings in zip(zomato_df['name'],zomato_df['reviews_list']):

    ratings = eval(ratings)

    for score, doc in ratings:

        if score:

            score = score.strip("Rated").strip()

            doc = doc.strip('RATED').strip()

            score = float(score)

            all_ratings.append([name,score, doc])
rating_df=pd.DataFrame(all_ratings,columns=['name','rating','review'])

rating_df['review']=rating_df['review'].apply(lambda x : re.sub('[^a-zA-Z0-9\s]'," ",x))
rating_df.head()
def produce_wordcloud(r):

  print (r+' as a restaurant type')

  plt.figure(figsize=(20,30))

  

  for j, n in enumerate(datas[datas['rest_type'] == r].name):

        plt.subplot(2,5,j+1)

        #print(r, x+j+1)

        stopword_t = set(STOPWORDS)

        stopword_t.update(n.split(" "))

        corpus=rating_df[n == rating_df.name]['review'].values.tolist()

        corpus=' '.join(x  for x in corpus)



        wordcloud = WordCloud(stopwords=stopword_t, max_font_size=None, background_color='black', collocations=False,

                      width=1500, height=1500).generate(corpus)

        plt.imshow(wordcloud)

        plt.title(n)

        plt.axis("off")

  #plt.tight_layout()
produce_wordcloud('Quick Bites')
produce_wordcloud('Casual Dining')
produce_wordcloud('Cafe')
produce_wordcloud('Delivery')
produce_wordcloud('Dessert Parlor')
produce_wordcloud('Casual Dining, Bar')
produce_wordcloud('Bakery')
produce_wordcloud('Beverage Shop')
online_order_t = zomato_df1.groupby(['address','online_order'])

online_orders=online_order_t['online_order'].nunique().index.to_frame()
rest_online_orders = online_orders.online_order.value_counts()

cmap = plt.get_cmap("tab20")

inner_colors = cmap(np.array([0, 1]))

plt.pie(rest_online_orders, labels=rest_online_orders.index, autopct='%1.1f%%', shadow=True, colors=inner_colors)

plt.axis('equal')

plt.show()
booking_t = zomato_df1.groupby(['address','book_table'])

table_booking=booking_t['book_table'].nunique().index.to_frame()
online_booking = table_booking.book_table.value_counts()

cmap = plt.get_cmap("tab20")

inner_colors = cmap(np.array([0, 1]))

plt.pie(online_booking, labels=online_booking.index, autopct='%1.1f%%', shadow=True, colors=inner_colors)

plt.axis('equal')

plt.show()
cost_t = zomato_df1.groupby(['address','approx_cost(for two people)'])

cost=cost_t['approx_cost(for two people)'].nunique().index.to_frame()
cost['approx_cost(for two people)'] = cost['approx_cost(for two people)'].str.replace(',', '').astype(float)
plt.figure(figsize=(6,6))

cost_dist=cost['approx_cost(for two people)'].dropna()

sns.distplot(cost_dist,bins=20,kde_kws={"color": "k", "lw": 3, "label": "KDE"})

plt.show();
cost_book_t = zomato_df1.groupby(['address','book_table','approx_cost(for two people)'])

cost_booking=cost_book_t['book_table','approx_cost(for two people)'].nunique().index.to_frame()
cost_booking['approx_cost(for two people)'] = cost_booking['approx_cost(for two people)'].str.replace(',', '').astype(float)
sns.boxplot(x='book_table',y='approx_cost(for two people)',data=cost_booking)



plt.show()
cost_booking.columns
cost_booking[cost_booking.book_table=="Yes"]['approx_cost(for two people)'].describe()
cost_booking[cost_booking.book_table=="No"]['approx_cost(for two people)'].describe()
online_vote_t = zomato_df1.groupby(['address','online_order','rate'])

online_votes=online_vote_t['online_order','rate'].nunique().index.to_frame()
# Mann-Whitney U test

from scipy.stats import mannwhitneyu



#We will be doing Mann-Whitney U test as our distribution is not normal, hence non-parametric type

data1 = online_votes[online_votes['online_order']=='Yes']['rate'].dropna().apply(lambda x : float(x.split('/')[0]) if (len(x)>3)  else np.nan ).dropna()

data2 = online_votes[online_votes['online_order']=='No']['rate'].dropna().apply(lambda x : float(x.split('/')[0]) if (len(x)>3)  else np.nan ).dropna()

# compare samples

stat, p = mannwhitneyu(data1, data2)

print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret

alpha = 0.05

if p > alpha:

	print('Same distribution (fail to reject H0)')

else:

	print('Different distribution (reject H0)')
plt.figure(figsize=(6,5))

sns.distplot(data1,bins=20,kde_kws={"color": "g", "lw": 3, "label": "Accepting online orders"})

sns.distplot(data2,bins=20,kde_kws={"color": "k", "lw": 3, "label": "Not accepting online orders"})

plt.show()
rating_t = zomato_df1.groupby(['address','rate'])

plt.figure(figsize=(6,5))

rating=rating_t.rate.nunique().index.to_frame()['rate'].dropna().apply(lambda x : float(x.split('/')[0]) if (len(x)>3)  else np.nan ).dropna()

sns.distplot(rating,bins=20,kde_kws={"color": "k", "lw": 3, "label": "KDE"})

plt.show()
rating.describe()
cost_dist_t = zomato_df1.groupby(['address','rate','approx_cost(for two people)','book_table','online_order'])

cost_dist=cost_dist_t['rate','approx_cost(for two people)','book_table','online_order'].nunique().index.to_frame()
cost_dist['approx_cost(for two people)'] = cost_dist['approx_cost(for two people)'].str.replace(',', '').astype(float)

cost_dist['rate']=cost_dist['rate'].apply(lambda x: float(x.split('/')[0]) if len(x)>3 else np.nan ).dropna()
fig, axis = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

sns.scatterplot(x="rate",y='approx_cost(for two people)',hue='book_table',data=cost_dist, ax=axis[0])

sns.scatterplot(x="rate",y='approx_cost(for two people)',hue='online_order',data=cost_dist, ax=axis[1])

plt.show()
custom_restaurant=zomato_df1[['address','rate','approx_cost(for two people)','location','name','rest_type','votes']].dropna().drop_duplicates()
custom_restaurant['rate']=custom_restaurant['rate'].apply(lambda x: float(x.split('/')[0]) if len(x)>3 else 0)

custom_restaurant['approx_cost(for two people)']=custom_restaurant['approx_cost(for two people)'].apply(lambda x: int(x.replace(',','')))
def search_restaurant(location='',rest='',rate=4,no_of_votes=200,min_cost=0, max_cost=500):

    if location!='' and rest!='':

      search_rest=custom_restaurant[(custom_restaurant['approx_cost(for two people)']>=min_cost) & (custom_restaurant['approx_cost(for two people)']<=max_cost) 

                      & (custom_restaurant['location']==location) & (custom_restaurant['rate']>rate) & (custom_restaurant['rest_type']==rest)

                      & (custom_restaurant['votes']>=no_of_votes)]

      pd.options.display.max_colwidth = 500

      return(print(search_rest.loc[:,['name']].reset_index().drop('index', axis=1)))

    else:

      search_rest=custom_restaurant[(custom_restaurant['approx_cost(for two people)']>=min_cost) & (custom_restaurant['approx_cost(for two people)']<=max_cost) 

                       & (custom_restaurant['rate']>rate) & (custom_restaurant['votes']>=no_of_votes)]

      pd.options.display.max_colwidth = 500

      return(print(search_rest.loc[:,['name']].reset_index().drop('index', axis=1)))
search_restaurant('Whitefield',"Casual Dining",4,400,0,1000)