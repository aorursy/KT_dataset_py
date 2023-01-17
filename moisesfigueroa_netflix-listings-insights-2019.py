import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.patheffects as path_effects
from matplotlib.patches import ConnectionPatch
import numpy as np
data = pd.read_csv('../input/netflix-shows/netflix_titles.csv')
data.head(2)
data.tail(2)
data.info()
data.isna().sum()
data.nunique()
data_processed= data.copy()
data_processed.loc[data_processed.country.isna(), 'country'] = 'TBD'
data_processed.loc[data_processed.cast.isna(), 'cast'] = 'TBD'
data_processed.loc[data_processed.director.isna(), 'director'] = 'TBD'
data_processed.loc[data_processed.rating.isna(), 'rating'] = 'TBD'
data_processed.loc[data_processed.date_added.isna(), 'date_added'] = 'January 1,0000'
data_processed.info()
data_processed.cast = data_processed.cast.apply(lambda x: x.split(', '))
data_processed.country = data_processed.country.apply(lambda x: x.split(', '))
data_processed.director = data_processed.director.apply(lambda x: x.split(', '))
data_processed.listed_in = data_processed.listed_in.apply(lambda x: x.split(', '))
data_processed.sample(5)
data_processed.loc[data_processed.rating == 'TV-Y7-FV', 'rating'] = "Children"
data_processed.loc[data_processed.rating == 'TV-Y7', 'rating'] = "Children"
data_processed.loc[data_processed.rating == 'TV-G', 'rating'] = "Children"
data_processed.loc[data_processed.rating == 'TV-Y', 'rating'] = "Children"
data_processed.loc[data_processed.rating == 'TV-PG', 'rating'] = "Older Children"
data_processed.loc[data_processed.rating == 'TV-14', 'rating'] = "Older Children"
data_processed.loc[data_processed.rating == 'PG-13', 'rating'] = "Older Children"
data_processed.loc[data_processed.rating == 'PG', 'rating'] = "Older Children"
data_processed.loc[data_processed.rating == 'G', 'rating'] = "Children"
data_processed.loc[data_processed.rating == 'TV-MA', 'rating'] = "Adult"
data_processed.loc[data_processed.rating == 'R', 'rating'] = "Adult"
data_processed.loc[data_processed.rating == 'NC-17', 'rating'] = "Adult"
data_processed.loc[data_processed.rating == 'NR', 'rating'] = "Not Rated/No Data"
data_processed.loc[data_processed.rating == 'UR', 'rating'] = "Not Rated/No Data"
data_processed.loc[data_processed.rating == 'TBD', 'rating'] = "Not Rated/No Data"
data_processed.rename(columns={'rating': 'content'},inplace=True)
data_processed.columns
data_processed.info()
data_processed.drop(columns='show_id', inplace=True)
data_processed.type = data_processed.type.astype('category')
data_processed.date_added = data_processed.date_added.astype('datetime64')
data_processed.release_year = data_processed.release_year.astype('category')
data_processed.content = data_processed.content.astype('category')
data_processed.info()
eastern_Countries = ['India','Japan', 'South Korea', 'China', 'Hong Kong', 'Turkey', 'Taiwan']

list_of_countries = data_processed.country.tolist()
list_of_countries_flat = [x for s in list_of_countries for x in s if x != 'TBD' and x not in eastern_Countries]
Counter(list_of_countries_flat).most_common(10)

graph = pd.DataFrame(Counter(list_of_countries_flat).most_common(10), columns=['Country','Count'])
fig, ax1 = plt.subplots()
fig.set_size_inches(12,8)

dict_rc = rc={'axes.labelcolor':'#E50914','figure.facecolor': 'red', 'axes.facecolor': 'red', 
              'text.color': 'white','ytick.color': 'white','xtick.color': 'white','axes.labelcolor': 'white', 
              'axes.spines.left': True,'axes.spines.bottom': False,'axes.spines.right': False,'axes.spines.top': False,
              'grid.linestyle': '-', 'grid.color': 'white'}

text = fig.text(0.5, .95, 'Top 10 Western Countries - Most Content on NETFLIX', color='white',
                          ha='center', va='center', size=35, weight='bold')
text.set_path_effects([
                          path_effects.PathPatchEffect(offset=(2, -2.5),facecolor='black', edgecolor='black'),
                          path_effects.PathPatchEffect(edgecolor='black',facecolor='white', linewidth=2)
                      ])

sns.set(font_scale=2, rc=dict_rc)
ax1 = sns.barplot(data=graph, y='Country',x='Count', order=graph.Country ,edgecolor='black',linewidth=3, palette='gray_r')  
ax1.set(xlabel='Content Count', ylabel='')
for k in ax1.patches:
    x = k.get_x()
    y = k.get_y()
    width_of_bar = k.get_width()
    height_of_bar = k.get_height()
    ax1.annotate(int(width_of_bar), (width_of_bar + 10, y + height_of_bar / 2), fontsize=15, va='center',weight='bold')
    
plt.show()
def get_labels_data(country = 'ALL'):
    if country != 'ALL':
        content = data_processed[data_processed.country.apply(lambda x: country in x)]
    else:
        content = data_processed
    
    
    movie_content = content.loc[content.type == 'Movie','content'].value_counts()
    tv_content = content.loc[content.type == 'TV Show', 'content'].value_counts()
    
    movie_tv_counts = content.type.value_counts()
    movie_tv_label = ['Movie', 'TV Show']
    
    movie_rating_data = movie_content.values.tolist()
    movie_rating_label = movie_content.index.tolist()
    
    tv_rating_data = tv_content.values.tolist()
    tv_rating_label = tv_content.index.tolist()
    
    return (movie_rating_label, movie_rating_data,
            movie_tv_label, movie_tv_counts, 
            tv_rating_label, tv_rating_data)
(movie_rating_label, movie_rating_data,
movie_tv_label, movie_tv_counts,
tv_rating_label, tv_rating_data) = get_labels_data()

(usa_movie_rating_label, usa_movie_rating_data,
 usa_movie_tv_label, usa_movie_tv_counts,
 usa_tv_rating_label, usa_tv_rating_data) = get_labels_data('United States')

(Spain_movie_rating_label, Spain_movie_rating_data,
 Spain_movie_tv_label, Spain_movie_tv_counts,
 Spain_tv_rating_label, Spain_tv_rating_data) = get_labels_data('Spain')

(UK_movie_rating_label, UK_movie_rating_data,
 UK_movie_tv_label, UK_movie_tv_counts,
 UK_tv_rating_label, UK_tv_rating_data) = get_labels_data('United Kingdom')

(Canada_movie_rating_label, Canada_movie_rating_data,
 Canada_movie_tv_label, Canada_movie_tv_counts,
 Canada_tv_rating_label, Canada_tv_rating_data) = get_labels_data('Canada')

(France_movie_rating_label, France_movie_rating_data,
 France_movie_tv_label, France_movie_tv_counts,
 France_tv_rating_label, France_tv_rating_data) = get_labels_data('France')
fig, ax = plt.subplots(6, 3)
fig.set_facecolor('red')
fig.set_size_inches(12,21)

text = fig.text(.5, .92, 'Breakdown of Types of Content by Intended Audience', color='white',
                          ha='center', va='center', size=32, weight='bold')

text.set_path_effects([path_effects.PathPatchEffect(offset=(2, -2.5),facecolor='black', edgecolor='black'),
                    path_effects.PathPatchEffect(edgecolor='black',facecolor='white', linewidth=2)])

text = fig.text(.5, .88, 'Movie Audience     Content Type     TV Audience', color='white',
                          ha='center', va='center', size=30, weight='bold')

text.set_path_effects([path_effects.PathPatchEffect(offset=(2, -2.5),facecolor='black', edgecolor='black'),
                    path_effects.PathPatchEffect(edgecolor='black',facecolor='white', linewidth=2)])

circle_rad = 1
font_size_labels = 8

dict_labels = { 'All': { 'ax1': [movie_rating_label, tv_rating_data, 280],
                    'ax2': [movie_tv_label, movie_tv_counts, 70], 
                    'ax3': [tv_rating_label, tv_rating_data, 260]},
    
                'USA': { 'ax4': [usa_movie_rating_label, usa_movie_rating_data,100], 
                'ax5': [usa_movie_tv_label, usa_movie_tv_counts, 40], 
                'ax6': [usa_tv_rating_label, usa_tv_rating_data, 280]},
    
                'UK' : {'ax10': [UK_movie_rating_label, UK_movie_rating_data, 100], 
                'ax11': [UK_movie_tv_label, UK_movie_tv_counts, 120],
                'ax12': [UK_tv_rating_label, UK_tv_rating_data, 280]},
    
                'CANADA' : {'ax13': [Canada_movie_rating_label, Canada_movie_rating_data, 110],
                'ax14': [Canada_movie_tv_label, Canada_movie_tv_counts, 50], 
                'ax15': [Canada_tv_rating_label, Canada_tv_rating_data, 280]},
    
                'FRANCE' : {'ax16': [France_movie_rating_label, France_movie_rating_data, 100], 
                'ax17': [France_movie_tv_label, France_movie_tv_counts, 90],
                'ax18': [France_tv_rating_label, France_tv_rating_data, 280]},
               
                'SPAIN': {'ax7': [Spain_movie_rating_label, Spain_movie_rating_data, 100],
                'ax8': [Spain_movie_tv_label, Spain_movie_tv_counts, 40], 
                'ax9': [Spain_tv_rating_label, Spain_tv_rating_data, 280]}}

bbox_props = dict(fc="red", ec="red", lw=5)
kw = dict(arrowprops=dict(arrowstyle="-", color='black', lw=1),
          bbox=bbox_props, zorder=0, va="center")

for x,z in zip(range(0, 6), dict_labels.keys()):
    text = ax[x,0].text(-2.5, 0, z, color='white',
                        ha='center', va='center', size=28, weight='bold',rotation=90)

    text.set_path_effects([path_effects.PathPatchEffect(offset=(2, -2.5),facecolor='black', edgecolor='black'),
                        path_effects.PathPatchEffect(edgecolor='black',facecolor='white', linewidth=2)])
    
    for y,w in zip(range(0, 3),dict_labels[z].keys()):
        patches,texts,autopct = ax[x,y].pie(
                                dict_labels[z][w][1],
                                startangle=dict_labels[z][w][2],
                                autopct='%1.1f%%',
                                shadow=True, 
                                colors = ['white'],
                                pctdistance=.7,
                                labeldistance=1.1, 
                                wedgeprops={'width': .65, 'antialiased': True, 'edgecolor': 'black', 'linewidth':3},
                                textprops={'color':'black','fontsize':font_size_labels, 'weight': 'bold'},
                                radius = circle_rad)
        plt.setp(autopct, color='black', fontsize=10, weight='bold', va='center')
        ax[x,y].set_aspect('equal')
        
        for i, p in enumerate(patches):
            ang = (p.theta2 - p.theta1)/2. + p.theta1
            y2 = np.sin(np.deg2rad(ang))
            x2 = np.cos(np.deg2rad(ang))
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x2))]
            connectionstyle = "angle,angleA=0,angleB={}".format(ang)
            kw["arrowprops"].update({"connectionstyle": connectionstyle})
            ax[x,y].annotate(dict_labels[z][w][0][i], xy=(x2, y2), xytext=(1.35*np.sign(x2), 1.4*y2),
                        horizontalalignment=horizontalalignment,fontsize='15', weight='bold',**kw)
        


plt.subplots_adjust(hspace=0, wspace=.8)

plt.show()
def top_5_actors_country(country):
    country_cast_list_of_lists_Movie = data_processed.loc[(data_processed.country.apply(lambda x: country in x)) & (data_processed.type == 'Movie'),'cast']
    flattened_list_Movie_Actors = [x for i in country_cast_list_of_lists_Movie for x in i if x != 'TBD']
    
    country_cast_list_of_lists_TV = data_processed.loc[(data_processed.country.apply(lambda x: country in x)) & (data_processed.type == 'TV Show'),'cast']
    flattened_list_TV_Actors = [x for i in country_cast_list_of_lists_TV for x in i if x != 'TBD']
    
    
    
    return [pd.DataFrame(Counter(flattened_list_Movie_Actors).most_common(5), columns= ['Actor', 'Count']),
           pd.DataFrame(Counter(flattened_list_TV_Actors).most_common(5), columns= ['Actor', 'Count'])]
top_5 = { 'USA': top_5_actors_country('United States'), 'UK': top_5_actors_country('United Kingdom'),  
         'Canada': top_5_actors_country('Canada'),'France' : top_5_actors_country('France'),'Spain': top_5_actors_country('Spain')}

fig, ax = plt.subplots(5,2,sharex=True)
fig.set_size_inches(12,14)


dict_rc = rc={'axes.labelcolor':'#E50914','figure.facecolor': 'red', 'axes.facecolor': 'red', 
              'text.color': 'white','ytick.color': 'white','xtick.color': 'white','axes.labelcolor': 'white', 
              'axes.spines.left': True,'axes.spines.bottom': False,'axes.spines.right': False,'axes.spines.top': False,
              'grid.linestyle': '-', 'grid.color': 'white'}

sns.set(font_scale=1.3, rc=dict_rc)

text = fig.text(0.43, .97, 'Top 5 Actors For Top 5 Countries with the Most Content', color='white',
                              ha='center', va='center', size=30, weight='bold')
text.set_path_effects([
                              path_effects.PathPatchEffect(offset=(2, -2.5),facecolor='black', edgecolor='black'),
                              path_effects.PathPatchEffect(edgecolor='black',facecolor='white', linewidth=2)
                          ])


text = fig.text(0.5, .93, 'Movies                                            TV Shows', color='white',
                              ha='center', va='center', size=25, weight='bold')
text.set_path_effects([
                        path_effects.PathPatchEffect(offset=(2, -2.5),facecolor='black', edgecolor='black'),
                        path_effects.PathPatchEffect(edgecolor='black',facecolor='white', linewidth=2)
                     ])
        

for x,i in zip(top_5.keys(), range(0,len(top_5))):
    for z,w in zip(top_5[x],range(0,2)):
        

        sns.barplot(data=z, y='Actor',x='Count', order=z.Actor ,edgecolor='black',linewidth=2, palette='gray_r',ax=ax[i,w])  
        ax[i,w].set(xlabel='Content Count', ylabel='',title=x)
        ax[i,w].tick_params(labelbottom=True)

        for k in ax[i, w].patches:
            x2 = k.get_x()
            y2 = k.get_y()
            width_of_bar = k.get_width()
            height_of_bar = k.get_height()
            ax[i,w].annotate(int(width_of_bar), (width_of_bar+.1, y2 + height_of_bar / 2), fontsize=10, va='center',weight='bold')

        

plt.subplots_adjust(wspace=.9,hspace=.8)
plt.show()
df_duration_genre = data_processed.loc[data_processed.type == 'Movie',['duration', 'listed_in']]
df_duration_genre.duration = df_duration_genre.duration.apply(lambda x: x.replace('min',''))

df_duration_genre = df_duration_genre.explode('listed_in').reset_index(drop=True)
df_duration_genre.duration = df_duration_genre.duration.astype('int64')

df_duration_genre.drop(df_duration_genre.loc[df_duration_genre.listed_in == 'Movies'].index, inplace=True)

order= df_duration_genre.groupby('listed_in').describe().sort_values(('duration', '75%'), ascending=False).index.tolist()

df_duration_genre.groupby('listed_in').describe().sort_values(('duration', '75%'), ascending=False)
fig, ax1 = plt.subplots()
fig.set_size_inches(12,15)
dict_rc = rc={'axes.labelcolor':'#E50914','figure.facecolor': 'red', 'axes.facecolor': 'red', 
              'text.color': 'white','ytick.color': 'white','xtick.color': 'white','axes.labelcolor': 'white', 
              'axes.spines.left': False,'axes.spines.bottom': False,'axes.spines.right': False,'axes.spines.top': False,
              'grid.linestyle': '-', 'grid.color': 'white'}

sns.set(font_scale=2, rc=dict_rc)

text = fig.text(0.35, .93, 'Movie Durations by Genre', color='white',
                              ha='center', va='center', size=40, weight='bold')
text.set_path_effects([
                              path_effects.PathPatchEffect(offset=(2, -2.5),facecolor='black', edgecolor='black'),
                              path_effects.PathPatchEffect(edgecolor='black',facecolor='white', linewidth=2)
                          ])


ax1 = sns.swarmplot(data = df_duration_genre, y='listed_in', x = 'duration', order=order,zorder=.5, alpha=.9, color='white')
ax2 = sns.boxplot(data = df_duration_genre, y='listed_in', x = 'duration', color='black', linewidth=3,order=order,showfliers=False,boxprops={'facecolor': 'None'})

ax1.set(ylabel='',xlabel= 'Duration (mins)')
ax2.set(ylabel='',xlabel= '')

plt.show()
df_duration_genre = data_processed.loc[data_processed.type == 'TV Show',['duration', 'listed_in']]
df_duration_genre.duration = df_duration_genre.duration.apply(lambda x: x.replace('Seasons',''))
df_duration_genre.duration = df_duration_genre.duration.apply(lambda x: x.replace('Season',''))



df_duration_genre = df_duration_genre.explode('listed_in').reset_index(drop=True)
df_duration_genre.duration = df_duration_genre.duration.astype('int64')



df_duration_genre.drop(df_duration_genre.loc[df_duration_genre.listed_in == 'TV Shows'].index, inplace=True)
fig, ax1 = plt.subplots()
fig.set_size_inches(12,8)
dict_rc = rc={'axes.labelcolor':'#E50914','figure.facecolor': 'red', 'axes.facecolor': 'red', 
              'text.color': 'white','ytick.color': 'white','xtick.color': 'white','axes.labelcolor': 'white', 
              'axes.spines.left': False,'axes.spines.bottom': False,'axes.spines.right': False,'axes.spines.top': False,
              'grid.linestyle': '-', 'grid.color': 'white'}

sns.set(font_scale=1.6, rc=dict_rc)

text = fig.text(0.5, .95, 'Number of Seasons - TV Shows', color='white',
                              ha='center', va='center', size=40, weight='bold')
text.set_path_effects([
                              path_effects.PathPatchEffect(offset=(2, -2.5),facecolor='black', edgecolor='black'),
                              path_effects.PathPatchEffect(edgecolor='black',facecolor='white', linewidth=2)
                          ])


ax1 = sns.countplot(y=df_duration_genre.duration, palette='Greys_r')
ax1.set(ylabel='Season #',xlabel= 'Number of shows')

#ax.set(ylabel='',xlabel= 'Season')
#ax2.set(ylabel='',xlabel= '')

plt.show()
df_duration_genre = data_processed.loc[data_processed.type == 'TV Show',['duration', 'listed_in']]
df_duration_genre.duration = df_duration_genre.duration.apply(lambda x: x.replace('Seasons',''))
df_duration_genre.duration = df_duration_genre.duration.apply(lambda x: x.replace('Season',''))



df_duration_genre = df_duration_genre.explode('listed_in').reset_index(drop=True)
df_duration_genre.duration = df_duration_genre.duration.astype('int64')



df_duration_genre.drop(df_duration_genre.loc[df_duration_genre.listed_in == 'TV Shows'].index, inplace=True)


df_duration_genre.groupby('listed_in').duration.value_counts(normalize=True).unstack().sort_values(list(range(15,0,-1)), ascending=False)

index = df_duration_genre.groupby('listed_in').duration.value_counts(normalize=True).unstack().sort_values(list(range(15,0,-1)), ascending=False).head(5).index.tolist()
top_5_genres_TV = df_duration_genre.loc[df_duration_genre.listed_in.apply(lambda x: any(g in x for g in index))]
top_5_genres_TV = top_5_genres_TV.groupby('listed_in').duration.value_counts(normalize=True).unstack().reset_index().melt(id_vars = 'listed_in')
fig, ax1 = plt.subplots()
fig.set_size_inches(12,8)
dict_rc = rc={'axes.labelcolor':'#E50914','figure.facecolor': 'red', 'axes.facecolor': 'red', 
              'text.color': 'white','ytick.color': 'white','xtick.color': 'white','axes.labelcolor': 'white', 
              'axes.spines.left': False,'axes.spines.bottom': False,'axes.spines.right': False,'axes.spines.top': False,
              'grid.linestyle': '-', 'grid.color': 'white'}

sns.set(font_scale=1.6, rc=dict_rc)

text = fig.text(0.5, .95, 'Number of Seasons - TV Shows', color='white',
                              ha='center', va='center', size=40, weight='bold')
text.set_path_effects([
                              path_effects.PathPatchEffect(offset=(2, -2.5),facecolor='black', edgecolor='black'),
                              path_effects.PathPatchEffect(edgecolor='black',facecolor='white', linewidth=2)
                          ])


ax1 = sns.lineplot(data = top_5_genres_TV, x ='duration', y='value',hue='listed_in', palette='Greys_r', lw=3)

ax1.set(ylabel='Season #',xlabel= 'Number of shows')

plt.xticks(range(1,15))
#ax.set(ylabel='',xlabel= 'Season')
#ax2.set(ylabel='',xlabel= '')

plt.show()
