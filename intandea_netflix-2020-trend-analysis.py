import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import geopandas as gpd

from descartes import PolygonPatch

import glob

import matplotlib.cm as cm

import seaborn as sn
# Read all files

path = r'../input/netflix2020' # use your path

all_files = glob.glob(path + "/*.csv")

all_file = all_files.remove('../input/netflix2020/NF_all.csv')

li = []



for filename in all_files:

    df = pd.read_csv(filename, index_col=None, header=0)

    li.append(df)



# Concat All Files

nf = pd.concat(li, axis=0, ignore_index=True)
# Delete error columns

nf.drop(["Continent/the-letter-for-the-king-2020", "EUR"], axis = 1, inplace = True)
#Checking which columns have nan

for col in nf.columns:

    uns = nf[col].unique()

    if nf[col].isnull().values.any():

        print(col, ": existing nan")

    else:

        print(col, ": no nan")
# Replacing NaN values



nf['Continent'] = nf['Continent'].fillna('EUR')



for col in ['ori_country', 'genre']:

    nf[col] = nf[col].fillna('Unknown')

    

for col in ['imdb_rating', 'rt_rating']:

    nf[col] = nf[col].fillna('0%')

    

# Replacing error values in show type

nf.show_type = nf.show_type.replace('Movie -- --', 'Movie')
nf.loc[nf.title.isin(['The World of the Married', 'Hitman: Agent Jun', 'Short', 'Metamorphosis', 'Man of Men']), 'ori_country'] = 'South Korea'

nf.loc[nf.title.isin(['The Great Heist', "Her Mother's Killer"]), 'ori_country'] = 'Colombia'

nf.loc[nf.title.isin(['Once Before', 'Through Night and Day', 'On Vodka, Beers, and Regrets']), 'ori_country'] = 'Philippines'

nf.loc[nf.title.isin(['Scarecrow', 'No Surrender']), 'ori_country'] = 'Egypt'

nf.loc[nf.title.isin(['Em Prova: Amiga do Inimigo', 'Em Prova: Amiga do Inimigo' ]), 'ori_country'] = 'Brazil'

nf.loc[nf.title.isin(['Nihontouitsu Series', 'The Lies She Loved']), 'ori_country'] = 'Japan'

nf.loc[nf.title.isin(['I Love You Two', 'Pee Nak 2', 'Friend Zone', 'Necromancer 2020', 'Lord Bunlue']), 'ori_country'] = 'Thailand'

nf.loc[nf.title.isin(['Exatlon Challenge']), 'ori_country'] = 'Turkey'

nf.loc[nf.title.isin(['KL Gangster 2']), 'ori_country'] = 'Malaysia'

nf.loc[nf.title.isin(['Fix Us']), 'ori_country'] = 'Ghana'
nf.loc[nf.release_date.isin(["Movie", 'Comedy', 'Egypt']), 'release_date'] = '1900-01-01'
nf.loc[nf['release_date'] == "Comedy", 'genre'] = 'Comedy'
nf.release_date = pd.to_datetime(nf.release_date)

nf['rel_yr'] = pd.DatetimeIndex(nf['release_date']).year

nf['rel_mt'] = pd.DatetimeIndex(nf['release_date']).month
nf['imdb_rating'] = nf['imdb_rating'].apply(lambda x: int(x[:-1]))

nf['rt_rating'] = nf['rt_rating'].apply(lambda x: int(x[:-1]))
nf.columns
# Saving for further reference

nf.to_csv("NF_all.csv", index = False)
nf.loc[nf['title'] == "The Platform", 'rt_rating'] = 80
# See how many weeks exist per Country

week_ctry = nf.groupby('country_chart')['week'].nunique()
# See the distribution of total weeks

plt.hist(week_ctry, bins = 24)
# Set a threshold for coloring the map

thr = 20



less_ctry = week_ctry[week_ctry < thr]

suff_ctry = week_ctry[week_ctry >= thr]
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# Check if there are country codes which do not match

countries = nf.country_chart.unique()

check = [a for a in countries if a in sorted(world.iso_a3.unique())]

set(countries) - set(check)
# Change the country code in world



# Hard-coded dictionary for mapping

chg = {'FRA' : 'France', 'GRE': 'Greece', 'HKG' : 'Hong Kong',

          'IRE': 'Ireland', 'NOR':'Norway', 'NZE': 'New Zealand',

       'POR' : 'Portugal', 'SGP' : 'Singapore'

      }



# Copy world and replace iso_a3 values

world_c = world.copy()

for i, (code, country) in enumerate(chg.items()):

    world_c.loc[world_c.name == country, 'iso_a3'] = code

#Creating new column for thresholding by week

world_c['nf_data'] = 'No Data'



for c in less_ctry.index:

    world_c.loc[world_c['iso_a3'] == c, 'nf_data'] = 'Less than 20 Weeks'



for c in suff_ctry.index:

    world_c.loc[world_c['iso_a3'] == c, 'nf_data'] = 'Equal/More than 20 Weeks'
# Plot The Map by Existing Weeks



fig, ax = plt.subplots(1, figsize = (20,15))

world_c.plot(column='nf_data', cmap = cm.get_cmap('Reds_r'), ax=ax, categorical=True,

             legend_kwds={'bbox_to_anchor':(.15, .6),'fontsize':9,'title':'No. Weeks'}, 

             legend = True, edgecolor="black")

ax.axis('off')

ax.set_title('Countries by Number of Existing Weeks in Dataset', fontweight = 'bold')

plt.tight_layout()

def returnVal(x):

    if np.isnan(x):

        return 'No Data'

    else:

        if x >= 0.5 :

            return 'Equal/More than 50%'

        else:

            return 'Less than 50%'
ct_nf_un = nf.groupby('country_chart')['title'].unique()

ct_nf1 = []

for i, shows in ct_nf_un.iteritems():

    shw = ct_nf_un[i]

    isNFori = [nf.loc[nf.title == s, 'is_NF_Ori'].values[0] for s in shows]

    ct_nf1.append({'country':i, 

                   'avg_un_nf': np.mean(isNFori),

                   'cont' : nf.loc[nf.country_chart == i, 'Continent'].values[0]

                  })

    

ct_nf1 = pd.DataFrame(ct_nf1)
world_nf_ori = pd.merge(world_c, ct_nf1.drop(['cont'], axis=1), how='outer', left_on='iso_a3', right_on ='country')

world_nf_ori['nf_ori_maj'] = world_nf_ori.avg_un_nf.apply(lambda x: returnVal(x))



fig, ax = plt.subplots(1, figsize = (20,15))



world_nf_ori.plot(column='nf_ori_maj', cmap = cm.get_cmap('Reds_r'), ax=ax, categorical = True,

             legend_kwds={'bbox_to_anchor':(.15, .6),'fontsize':9,'title':'% of Netflix Original Shows'}, 

             legend = True, edgecolor="black")

ax.axis('off')

ax.set_title('How Many Unique Netflix Original Shows are Watched in Each Country?', fontweight = 'bold')

plt.tight_layout()
# Group by country, then find the most watched genre

ctr_genre = nf.groupby('country_chart')['genre'].agg(lambda x:x.value_counts().index[0])
# Create a new column of most watched genre for plotting

for c in ctr_genre.index:

    world_c.loc[world_c['iso_a3'] == c, 'most_genre'] = ctr_genre[c]

    

world_c.most_genre = world_c.most_genre.fillna('No Data')
# Substitute for most popular is unknown

for c in countries:

    if world_c.loc[world_c.iso_a3 == c, 'most_genre'].values == 'Unknown':

        second_best = nf.loc[nf['country_chart'] == c, 'genre'].value_counts().index[1]

        world_c.loc[world_c.iso_a3 == c, 'most_genre'] = second_best



# This country has only 3 weeks, each of the 6 shows is of different genre

world_c.loc[world_c.iso_a3 == 'IDN', 'most_genre'] = 'Mixed'
world_c.most_genre.value_counts()
for c in world_c.most_genre.unique():

    print(c, list(world_c.loc[world_c.most_genre == c, 'iso_a3']))
# Plot map by most watched genre



fig, ax = plt.subplots(1, figsize = (20,15))



world_c.plot(column='most_genre', cmap = 'RdGy', ax=ax, categorical=True,

             legend_kwds={'bbox_to_anchor':(.2, .8),'fontsize':9, 'title': 'Genre'}, 

             legend = True, edgecolor="black")



ax.axis('off')

ax.set_title('What is The Most Watched Genre in Each Country?', fontweight = 'bold')

plt.tight_layout()
# Calculate total occurence per genre

ctr_genre_vc = ctr_genre.value_counts()



# Plot 

fig, ax = plt.subplots(figsize = (10, 6))

cs=cm.RdGy(np.arange(11)/11.)

ax.pie(ctr_genre_vc, autopct='%1.1f%%', labels = ctr_genre_vc.index, colors = cs) 

ax.set_title('What is The Most Watched Genre in Netflix?', fontweight='bold')
# Separated by Movie and TV Show

mv = nf.loc[nf.show_type.isin(['Movie', 'Documentary', 'Short'])]

tv = nf.loc[nf.show_type.isin(['TV Show', 'Documentary TV'])]
# Group by week, and find most appearing title from existing countries

mv_week_mode = mv.groupby(['week'])[['title']].agg(lambda x:x.value_counts().index[0])



# Adding Data

mv_week_mode['count'] = mv.groupby(['week'])[['title']].agg(lambda x:x.value_counts()[0]).values

mv_week_mode['ncountry'] =  mv.groupby(['week'])['country_chart'].nunique()

mv_week_mode['rate'] = mv_week_mode['count']/ mv_week_mode['ncountry']

mv_week_mode['genre'] = mv_week_mode['title'].apply(lambda x: nf.loc[nf.title == x, 'genre'].values[0])

mv_week_mode['is_nf_ori'] = mv_week_mode['title'].apply(lambda x: nf.loc[nf.title == x, 'is_NF_Ori'].values[0])

mv_week_mode['imdb_rating'] = mv_week_mode['title'].apply(lambda x: nf.loc[nf.title == x, 'imdb_rating'].values[0])

mv_week_mode['rt_rating'] = mv_week_mode['title'].apply(lambda x: nf.loc[nf.title == x, 'rt_rating'].values[0])



# Select weeks when a show topped > 50% of countries

mv_week_top = mv_week_mode.query('rate >= 0.5 & ncountry > 10')
# Set position of bar on X axis

barWidth = 0



week_pop_mv = np.arange(len(mv_week_top))

pop_mv_count = 100 * mv_week_top['rate'].values



# Create Masks for coloring the chart, returning indices when artist-song topped most charts

masks = []



pop_mv_title = mv_week_top.title.unique()

for m in pop_mv_title:

    masks.append(mv_week_top['title'] == m)



# colors = ["#62649C", "#373DFA", "#FCF877", "#ED409C", "#CECFEB", "#AB226B", "#62FCAA",  "#915776"]

# colors = ["#0A0530", "#4C5413", "#BECF40", "#7D1A40", "#1A1730", "#02F296", "#2E3019", "#A30844"]

colors = ["#470309","#61040B","#7C050D","#960610", "#B00711", "#CB0813", "#e50914", "#DA633F"]

colors = ["#67001F", "#B1182B", "#D6604D", "#F3A481", "#FDDBC7", "#FEEDE3", "#B9B9B9", "#5D5D5D"]



# Make the plot

f, ax = plt.subplots(figsize=(8, 6))



for i, mask in enumerate(masks):

    plt.bar(week_pop_mv[mask.values], pop_mv_count[mask.values], color=colors[i], edgecolor='white', \

            label = pop_mv_title[i])



# title_genre = [x + " (" + str(mv_week_top.loc[mv_week_top.title == x, 'genre'].values[0]) + ")" for x in pop_mv]

# Add xticks on the middle of the group bars

plt.xlabel('Week', fontweight='bold')

plt.ylabel('Percentage of Total Countries Topped at That Week (%)', fontweight='bold')

plt.xticks([r + barWidth for r in range(len(mv_week_top))], mv_week_top.index)

plt.title('What Movies Have Been #1 in a Week for More Than 50% Countries?', fontweight='bold')



ax.legend(handles=ax.lines[::len(mv_week_top)+1], labels=pop_mv_title, fontsize=8, title='Title',

         bbox_to_anchor=(1.3, 0.8))

plt.show()
# width of the bars

barWidth = 0.3



mv_top_un = mv_week_top.drop_duplicates(subset=["title"])

# The x position of bars

r1 = np.arange(len(mv_top_un))

r2 = [x + barWidth for x in r1]



# Create blue bars

plt.bar(r1, mv_top_un.imdb_rating, width = barWidth, color = '#F3A481', label='IMDB')

 

# Create cyan bars

plt.bar(r2, mv_top_un.rt_rating, width = barWidth, color = '#B1182B',label='Rotten Tomatoes')

 

# general layout

plt.xticks([r + 0.15 for r in range(len(mv_top_un))], mv_top_un.title, rotation = 60)

plt.ylabel('Score (%)', fontweight = 'bold')

plt.xlabel('Movie Title', fontweight = 'bold')

plt.title('What are The Ratings of The Most Watched Movies in Netflix?', fontweight = 'bold')

plt.legend(bbox_to_anchor=(1.05, 0.75))

 

# Show graphic

plt.show()
mv_week_top
# Group by week, and find most appearing title from existing countries

tv_week_mode = tv.groupby(['week'])[['title']].agg(lambda x:x.value_counts().index[0])



# Adding Data

tv_week_mode['count'] = tv.groupby(['week'])[['title']].agg(lambda x:x.value_counts()[0]).values

tv_week_mode['ncountry'] = tv.groupby(['week'])['country_chart'].nunique()

tv_week_mode['rate'] = tv_week_mode['count']/ tv_week_mode['ncountry']

tv_week_mode['genre'] = tv_week_mode['title'].apply(lambda x: nf.loc[nf.title == x, 'genre'].values[0])

tv_week_mode['is_nf_ori'] = tv_week_mode['title'].apply(lambda x: nf.loc[nf.title == x, 'is_NF_Ori'].values[0])

tv_week_mode['imdb_rating'] = tv_week_mode['title'].apply(lambda x: nf.loc[nf.title == x, 'imdb_rating'].values[0])

tv_week_mode['rt_rating'] = tv_week_mode['title'].apply(lambda x: nf.loc[nf.title == x, 'rt_rating'].values[0])



# Select weeks when a show topped > 50% of countries

tv_week_top = tv_week_mode.query('rate >= 0.5 & ncountry > 10')
tv_week_top
# Set position of bar on X axis

barWidth = 0



week_pop_tv = np.arange(len(tv_week_top))

pop_tv_count = 100 * tv_week_top['rate'].values



# Create Masks for coloring the chart, returning indices when artist-song topped most charts

masks_tv = []



pop_tv_title = tv_week_top.title.unique()

for t in pop_tv_title:

    masks_tv.append(tv_week_top['title'] == t)



# colors = ["#0A0530", "#4C5413", "#BECF40", "#7D1A40", "#1A1730", "#02F296", "#2E3019", "#A30844"]

# colors = ["#2E3019", "#4C5413", "#BECF40", "#A30844", "#7D1A40", "#02F296",  "#0A0530",]

# colors = ["#F9476F", "#F02443","#E50914","#BE0712", "#960610", "#6F040C", "#470309"]

# colors = ["#006D77", "#83C5BE", "#EDF6F9", "#FFDDD2", "#E29578", "#795663", "#1E1E24"]



colors = ["#67001F", "#B1182B", "#D6604D", "#F3A481", "#FDDBC7", "#FEEDE3", "#B9B9B9", "#5D5D5D"]



# Make the plot

f, ax = plt.subplots(figsize=(8, 6))



for i, mask in enumerate(masks_tv):

    plt.bar(week_pop_tv[mask.values], pop_tv_count[mask.values], color=colors[i],  \

            label = pop_tv_title[i])



# Add xticks on the middle of the group bars

plt.xlabel('Week', fontweight='bold')

plt.ylabel('Percentage of Total Countries Topped at That Week (%)', fontweight='bold')

plt.xticks([r + barWidth for r in range(len(tv_week_top))], tv_week_top.index)

plt.title('What TV Shows Have Been #1 in The Same Week for More Than 50% Countries?', fontweight='bold')



ax.legend(handles=ax.lines[::len(tv_week_top)+1], labels=pop_tv_title, bbox_to_anchor=(1.4, 0.8))

plt.show()

# width of the bars

barWidth = 0.3

 

tv_top_un = tv_week_top.drop_duplicates(subset=["title"])

# The x position of bars

r1 = np.arange(len(tv_top_un))

r2 = [x + barWidth for x in r1]

 

# Create blue bars

plt.bar(r1, tv_top_un.imdb_rating, width = barWidth, color = '#F3A481', label='IMDB')

 

# Create cyan bars

plt.bar(r2, tv_top_un.rt_rating, width = barWidth, color = '#B1182B', label='Rotten Tomatoes')

 

# general layout

plt.xticks([r + 0.15 for r in range(len(tv_top_un))], tv_top_un.title, rotation = 60)

plt.ylabel('Score (%)', fontweight = 'bold')

plt.xlabel('Movie Title', fontweight = 'bold')

plt.title('What are The Ratings of The Most Watched TV Shows in Netflix?', fontweight = 'bold')

plt.legend(bbox_to_anchor=(1.05, 0.75))

 

# Show graphic

plt.show()
drakor = tv.loc[tv['ori_country'] == 'South Korea']

dk = drakor[['week', 'country_chart', 'title']]

dk = dk.groupby(['week', 'country_chart'])[['title']].agg(lambda x: x).unstack('week').fillna('#1 was not K-Drama')

dk.columns  = dk.columns.droplevel()

dku = np.append(drakor.title.unique(),'#1 was not K-Drama')
value_to_int = {j:i for i,j in enumerate(dku)} 

n = len(value_to_int)     

# discrete colormap (n samples from a given cmap)

fig,ax = plt.subplots(1, figsize=(13,8))

cmap = sn.color_palette("RdGy", n) 

ax = sn.heatmap(dk.replace(value_to_int), cmap=cmap) 

# modify colorbar:

colorbar = ax.collections[0].colorbar 

r = colorbar.vmax - colorbar.vmin 

colorbar.set_ticks([colorbar.vmin + r / n * (0.5 + i) for i in range(n)])

colorbar.set_ticklabels(list(value_to_int.keys()))       

plt.title('Which Korean Dramas were #1 in Netflix?', fontweight='bold')

plt.ylabel('Country', fontweight='bold')

plt.xlabel('Week', fontweight='bold')

plt.yticks(rotation = 0)

plt.show()
nf_unique = nf.drop_duplicates(subset = ["title"])

mv_un = nf_unique.loc[nf_unique.show_type.isin(['Movie', 'Documentary', 'Short'])]

tv_un = nf_unique.loc[nf_unique.show_type.isin(['TV Show', 'Documentary TV'])]
mv_5 = pd.DataFrame(mv_un.ori_country.value_counts()[:5])

mv_5.at['Others', 'ori_country'] = sum(mv_un.ori_country.value_counts()) - sum(mv_5['ori_country'].values)



fig, ax = plt.subplots(figsize = (10, 6))

cs=cm.RdGy(np.arange(6)/6.)

ax.pie(mv_5, labels=mv_5.index, autopct='%1.1f%%', colors = cs) 

ax.set_title('Where did #1 Movies in Netflix Come From?', fontweight='bold')
tv_5 = pd.DataFrame(tv_un.ori_country.value_counts()[:5])

tv_5.at['Others', 'ori_country'] = sum(tv_un.ori_country.value_counts()) - sum(tv_5['ori_country'].values)



fig, ax = plt.subplots(figsize = (10, 6))

cs=cm.RdGy(np.arange(6)/6.)

ax.pie(tv_5, labels = tv_5.index, autopct='%1.1f%%', colors=cs) 

ax.set_title('Where did #1 TV Shows in Netflix Come From?', fontweight='bold')
mvu = nf.loc[nf.show_type.isin(['Movie', 'Documentary', 'Short'])].drop_duplicates(subset = ['show_link'])

tvu = nf.loc[nf.show_type.isin(['TV Show', 'Documentary TV'])].drop_duplicates(subset = ['show_link'])



# tvu.query('rel_yr == 2020')