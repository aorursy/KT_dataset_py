movies = pd.read_csv('../input/tmdb_5000_movies.csv')

credits = pd.read_csv('../input/tmdb_5000_credits.csv')



movies.head()
credits.head()
(credits['title']==movies['title']).describe()
del credits['title']

del credits['movie_id']

movie_df = pd.concat([movies, credits], axis=1)

movie_df.head()
newCols = ['title','release_date','popularity','vote_average','vote_count',

           'budget','revenue','genres','keywords','cast','crew','tagline', 'runtime', 'production_companies', 

           'production_countries', 'status']



df2 = movie_df[newCols]

df2.head()
df2.describe().round()
my_imputer = Imputer()



temp=df2

X2 = my_imputer.fit_transform(df2[['runtime']])

df2['runtime'] = X2

df2.describe().round()
#df2['vote_classes'] = pd.cut(df2['vote_average'],10, labels=["1", "2","3","4","5","6","7","8","9","10"])

df2['vote_classes'] = pd.cut(df2['vote_average'],4, labels=["low", "medium-low","medium-high","high"])
df2['log_budget'] = np.log(df2['budget'])

df2['log_popularity'] = np.log(df2['popularity'])

df2['log_vote_average'] = np.log(df2['vote_average'])

df2['log_vote_count'] = np.log(df2['vote_count'])

df2['log_revenue']= np.log(df2['revenue'])

df2['log_runtime']= np.log(df2['runtime'])

df3=df2[df2.columns[-5:]]



#df3.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

df3=df3[df3.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]

df3=df3.dropna(axis=1)

#df3[~df3.isin([np.nan, np.inf, -np.inf]).any(1)]

from pandas.plotting import scatter_matrix

scatter_matrix(df3,alpha=0.2, figsize=(20, 20), diagonal='kde')
Early_df = df2[df2.columns[0:16]]

Early_df.head()
mean_per_keyword.sort_values('mean_vote_average', ascending=False).head()
mean_per_keyword.sort_values('mean_budget', ascending=False).head()
mean_per_keyword.sort_values('mean_revenue', ascending=False).head()
fig = plt.figure(1, figsize=(18,13))

trunc_occurences = new_keyword_occurences[0:50]

# LOWER PANEL: HISTOGRAMS

ax2 = fig.add_subplot(2,1,2)

y_axis = [i[1] for i in trunc_occurences]

x_axis = [k for k,i in enumerate(trunc_occurences)]

x_label = [i[0] for i in trunc_occurences]

plt.xticks(rotation=85, fontsize = 15)

plt.yticks(fontsize = 15)

plt.xticks(x_axis, x_label)

plt.ylabel("Nb. of occurences", fontsize = 18, labelpad = 10)

ax2.bar(x_axis, y_axis, align = 'center', color='g')

#_______________________

plt.title("Keywords popularity",bbox={'facecolor':'k', 'pad':5},color='w',fontsize = 25)

plt.show()
Df1 = pd.DataFrame(trunc_occurences)

Df2 = mean_per_keyword

result = Df1.merge(Df2, left_on=0, right_on=0, how='inner')
result = result.rename(columns ={0:'keyword', 1:'occurences'})
result.sort_values('mean_vote_average', ascending= False)
result['mean_vote_average'].mean()
import matplotlib.pyplot as plt



ax = result.plot.bar(x = 'keyword', y='mean_vote_average', title="mean vote average",

                     figsize=(15,4), legend=True, fontsize=12, color='green', label = "mean vote average")

ax.set_ylim(5, 8)

ax.axhline(y=result['mean_vote_average'].mean(),c="blue",linewidth=0.5, label='mean')

ax.legend()

plt.show()

import matplotlib.pyplot as plt



ax = result.plot.bar(x = 'keyword', y='mean_budget', title="mean budget",

                     figsize=(15,4), legend=True, fontsize=12, color='green', label="mean budget")

ax.axhline(y=result['mean_budget'].mean(),c="blue",linewidth=0.5, label='mean')

ax.legend()

plt.show()

result.sort_values('mean_budget').head()
ax = result.plot.bar(x = 'keyword', y='mean_revenue', title="mean revenue",

                     figsize=(15,4), legend=True, fontsize=12, color='green', label="mean revenue")

ax.axhline(y=result['mean_revenue'].mean(),c="blue",linewidth=0.5, label='mean')

ax.legend()

plt.show()
result['profit'] = result['mean_revenue'] - result['mean_budget']

result.head()
ax = result.plot.bar(x = 'keyword', y='profit', title="profit",

                     figsize=(15,4), legend=True, fontsize=12, color='green', label="profit")

ax.axhline(y=result['profit'].mean(),c="blue",linewidth=0.5, label='mean')

ax.legend()

plt.show()
df.head()
df3 = df # We store a copy of the dataframe for later use
columns = ['homepage', 'plot_keywords', 'language', 'overview', 'popularity', 'tagline',

           'original_title', 'num_voted_users', 'country', 'spoken_languages', 'duration',

          'production_companies', 'production_countries', 'status']



df = df.drop(columns, axis=1)
liste_genres = set()

for s in df['genres'].str.split('|'):

    liste_genres = set().union(s, liste_genres)

liste_genres = list(liste_genres)

liste_genres.remove('')
df_reduced = df[['actor_1_name', 'vote_average',

                 'title_year', 'movie_title', 'gross', 'budget']].reset_index(drop = True)

for genre in liste_genres:

    df_reduced[genre] = df['genres'].str.contains(genre).apply(lambda x:1 if x else 0)



df_reduced2 = df[['actor_2_name', 'vote_average',

                 'title_year', 'movie_title', 'gross', 'budget']].reset_index(drop = True)

for genre in liste_genres:

    df_reduced2[genre] = df['genres'].str.contains(genre).apply(lambda x:1 if x else 0)



df_reduced3 = df[['actor_3_name', 'vote_average',

                 'title_year', 'movie_title', 'gross', 'budget']].reset_index(drop = True)

for genre in liste_genres:

    df_reduced3[genre] = df['genres'].str.contains(genre).apply(lambda x:1 if x else 0)

df_reduced = df_reduced.rename(columns={'actor_1_name': 'actor'})

df_reduced2 = df_reduced2.rename(columns={'actor_2_name': 'actor'})

df_reduced3 = df_reduced3.rename(columns={'actor_3_name': 'actor'})



total = [df_reduced, df_reduced2, df_reduced3]

df_total = pd.concat(total)

df_total.head()
df_actors = df_total.groupby('actor').mean()

df_actors.loc[:, 'favored_genre'] = df_actors[liste_genres].idxmax(axis = 1)

df_actors.drop(liste_genres, axis = 1, inplace = True)

df_actors = df_actors.reset_index()
df_appearance = df_total[['actor', 'title_year']].groupby('actor').count()

df_appearance = df_appearance.reset_index(drop = True)

selection = df_appearance['title_year'] > 9

selection = selection.reset_index(drop = True)

most_prolific = df_actors[selection]
most_prolific.sort_values('vote_average', ascending=False).head()
most_prolific.sort_values('gross', ascending=False).head()
most_prolific.sort_values('budget', ascending=False).head()
genre_count = []

for genre in liste_genres:

    genre_count.append([genre, df_reduced[genre].values.sum()])

genre_count.sort(key = lambda x:x[1], reverse = True)

labels, sizes = zip(*genre_count)

labels_selected = [n if v > sum(sizes) * 0.01 else '' for n, v in genre_count]

reduced_genre_list = labels[:19]

trace=[]

for genre in reduced_genre_list:

    trace.append({'type':'scatter',

                  'mode':'markers',

                  'y':most_prolific.loc[most_prolific['favored_genre']==genre,'gross'],

                  'x':most_prolific.loc[most_prolific['favored_genre']==genre,'budget'],

                  'name':genre,

                  'text': most_prolific.loc[most_prolific['favored_genre']==genre,'actor'],

                  'marker':{'size':10,'opacity':0.7,

                            'line':{'width':1.25,'color':'black'}}})

layout={'title':'Actors favored genres',

       'xaxis':{'title':'mean year of activity'},

       'yaxis':{'title':'mean score'}}

fig=Figure(data=trace,layout=layout)

pyo.iplot(fig)
selection = df_appearance['title_year'] > 20

most_prolific = df_actors[selection]

most_prolific
class Trace():

    #____________________

    def __init__(self, color):

        self.mode = 'markers'

        self.name = 'default'

        self.title = 'default title'

        self.marker = dict(color=color, size=110,

                           line=dict(color='white'), opacity=0.7)

        self.r = []

        self.t = []

    #______________________________

    def set_color(self, color):

        self.marker = dict(color = color, size=110,

                           line=dict(color='white'), opacity=0.7)

    #____________________________

    def set_name(self, name):

        self.name = name

    #____________________________

    def set_title(self, title):

        self.na = title

    #___________________________

    def set_actor(self, actor):

        self.actor = actor

    

    #__________________________

    def set_values(self, r, t):

        self.r = np.array(r)

        self.t = np.array(t)
names =['Morgan Freeman']

df2 = df_reduced[df_reduced['actor'] == 'Morgan Freeman']

total_count  = 0

years = []

imdb_score = []

genre = []

titles = []

actor = []

for s in liste_genres:

    icount = df2[s].sum()

    #__________________________________________________________________

    # Here, we set the limit to 3 because of a bug in plotly's package

    if icount > 3: 

        total_count += 1

        genre.append(s)

        actor.append(list(df2[df2[s] ==1 ]['actor']))

        years.append(list(df2[df2[s] == 1]['title_year']))

        imdb_score.append(list(df2[df2[s] == 1]['vote_average'])) 

        titles.append(list(df2[df2[s] == 1]['movie_title']))

max_y = max([max(s) for s in years])

min_y = min([min(s) for s in years])

year_range = max_y - min_y



years_normed = []

for i in range(total_count):

    years_normed.append( [360/total_count*((an-min_y)/year_range+i) for an in years[i]])
color = ('royalblue', 'grey', 'wheat', 'c', 'firebrick', 'seagreen', 'lightskyblue',

          'lightcoral', 'yellowgreen', 'gold', 'tomato', 'violet', 'aquamarine', 'chartreuse', 'red')
trace = [Trace(color[i]) for i in range(total_count)]

tr    = []

for i in range(total_count):

    trace[i].set_name(genre[i])

    trace[i].set_title(titles[i])

    trace[i].set_values(np.array(imdb_score[i]),

                        np.array(years_normed[i]))

    tr.append(go.Scatter(r      = trace[i].r,

                         t      = trace[i].t,

                         mode   = trace[i].mode,

                         name   = trace[i].name,

                         marker = trace[i].marker,

#                         text   = ['default title' for j in range(len(trace[i].r))], 

                         hoverinfo = 'all'

                        ))        

layout = go.Layout(

    title='Morgan Freeman movies',

    font=dict(

        size=15

    ),

    plot_bgcolor='rgb(223, 223, 223)',

    angularaxis=dict(        

        tickcolor='rgb(253,253,253)'

    ),

    hovermode='Closest',

)

fig = go.Figure(data = tr, layout=layout)

pyo.iplot(fig)
#Directors are counted and compared

df = df3



def create_comparison_database(name, value, x, no_films):

    

    comparison_df = df3.groupby(name, as_index=False)

    

    if x == 'mean':

        comparison_df = comparison_df.mean()

    elif x == 'median':

        comparison_df = comparison_df.median()

    elif x == 'sum':

        comparison_df = comparison_df.sum() 

    

    # Create database with either name of directors or actors, the value being compared i.e. 'revenue',

    # and number of films they're listed with. Then sort by value being compared.

    name_count_key = df[name].value_counts().to_dict()

    comparison_df['films'] = comparison_df[name].map(name_count_key)

    comparison_df.sort_values(value, ascending=False, inplace=True)

    comparison_df[name] = comparison_df[name].map(str) + " (" + comparison_df['films'].astype(str) + ")"

   # create a Series with the name as the index so it can be plotted to a subgrid

    comp_series = comparison_df[comparison_df['films'] >= no_films][[name, value]][10::-1].set_index(name).ix[:,0]

    

    return comp_series
fig = plt.figure(figsize=(18,6))



# Director_name

plt.subplot2grid((2,3),(0,0), rowspan = 2)

create_comparison_database('director_name','gross','sum', 4).plot(kind='barh', color='#006600')

plt.legend().set_visible(False)

plt.title("Total revenue for Directors with 4+ Films")

plt.ylabel("Director (no. films)")

plt.xlabel("Revenue")



plt.subplot2grid((2,3),(0,1), rowspan = 2)

create_comparison_database('director_name','gross','mean', 4).plot(kind='barh', color='#ffff00')

plt.legend().set_visible(False)

plt.title('Average revenue for Directors with 4+ Films')

plt.ylabel("Director (no. films)")

plt.xlabel("Revenue")



plt.tight_layout()
fig = plt.figure(figsize=(18,6))



# Director_name

plt.subplot2grid((2,3),(0,0), rowspan = 2)

create_comparison_database('director_name','budget','mean', 4).plot(kind='barh', color='#006600')

plt.legend().set_visible(False)

plt.title("Average budget for Directors with 4+ Filmss")

plt.ylabel("Director (no. films)")

plt.xlabel("Budget (in billons)")



plt.subplot2grid((2,3),(0,1), rowspan = 2)

create_comparison_database('director_name','vote_average','mean', 4).plot(kind='barh', color='#ffff00')

plt.legend().set_visible(False)

plt.title('Mean IMDB Score for Directors with 4+ Films')

plt.ylabel("Director (no. films)")

plt.xlabel("IMDB Score")

plt.xlim(0,10)



plt.tight_layout()
fig = plt.figure(figsize=(18,6))



# Director_name

plt.subplot2grid((2,3),(0,0), rowspan = 2)

create_comparison_database('director_name','budget','mean', 10).plot(kind='barh', color='#006600')

plt.legend().set_visible(False)

plt.title("Average budget for Directors with 15+ Filmss")

plt.ylabel("Director (no. films)")

plt.xlabel("Budget")



plt.subplot2grid((2,3),(0,1), rowspan = 2)

create_comparison_database('director_name','vote_average','mean', 10).plot(kind='barh', color='#ffff00')

plt.legend().set_visible(False)

plt.title('Mean IMDB Score for Directors with 15+ Films')

plt.ylabel("Director (no. films)")

plt.xlabel("IMDB Score")

plt.xlim(0,10)



plt.tight_layout()
df2.head()
df2 = Early_df
df2['log_budget'] = np.log(df2['budget'])

df2['log_popularity'] = np.log(df2['popularity'])

df2['log_vote_average'] = np.log(df2['vote_average'])

df2['log_vote_count'] = np.log(df2['vote_count'])

df2['log_revenue']= np.log(df2['revenue'])

df2['log_runtime']= np.log(df2['runtime'])

df3=df2[df2.columns[-6:]]



df3=df3[df3.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]

df3=df3.dropna(axis=1)

df3.head()
num_list = ['budget','popularity','revenue','runtime','vote_average','vote_count']

movie_num = df2[num_list]

movie_num.head()
df3.head()
movie_num['revenue'] .plot(kind='hist')
df3['log_revenue'].plot(kind='hist')
f, ax = plt.subplots(figsize=(12,10))

plt.title('Pearson Correlation of Movie Features')

sns.heatmap(movie_num.astype(float).corr(), linewidths=0.25, vmax=1.0, square=True,

           cmap="YlGnBu", linecolor='black', annot=True)
num_list = ['budget','popularity','revenue','runtime','vote_average','vote_count']

movie_num = df2[num_list]

movie_num.head()
training_list = ['budget','popularity','revenue','runtime','vote_count']

training = movie_num[training_list]

target = movie_num['vote_average']
X = training.values

y = target.values
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(

    X, y, test_size=0.33, random_state=42)
from sklearn import linear_model

# Create linear regression object

regr = linear_model.LinearRegression()



# Train the model using the training sets

regr.fit(X_train, y_train)



# Make predictions using the testing set

y_pred_lr = regr.predict(X_test)
f = plt.figure(figsize=(10,5))

plt.scatter(X_test[:,1], y_test, s=50,label="Real vote_average");

plt.scatter(X_test[:,1], y_pred_lr,s=100, c='r',label="Predicted vote_average");

plt.ylabel("vote_average");

plt.legend(loc=2);
from sklearn.ensemble import RandomForestRegressor

# Create linear regression object

rf = RandomForestRegressor(1)



# Train the model using the training sets

rf.fit(X_train, y_train)



# Make predictions using the testing set

y_pred_rf = rf.predict(X_test)
f = plt.figure(figsize=(10,5))

plt.scatter(X_test[:,1], y_test, s=50,label="Real vote_average");

plt.scatter(X_test[:,1], y_pred_rf,s=100, c='r',label="Predited vote_average");

plt.ylabel("vote_average");

plt.legend(loc=2);
from sklearn.metrics import mean_squared_error



error_lr = mean_squared_error(y_test,y_pred_lr)

error_rf = mean_squared_error(y_test,y_pred_rf)



print(error_lr)

print(error_rf)
f = plt.figure(figsize=(10,5))

plt.bar(range(2),[error_lr,error_rf])

plt.xlabel("Classifiers");

plt.ylabel("Mean Squared Error of the vote_average");

plt.xticks(range(2),['Linear Regression','Random Forest'])

plt.legend(loc=2);