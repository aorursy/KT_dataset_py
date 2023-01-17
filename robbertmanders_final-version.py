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
df = df3
def create_comparison_database(name, value, x, no_films):

    

    comparison_df = df3.groupby(name, as_index=False)

    

    if x == 'mean':

        comparison_df = comparison_df.mean()

    elif x == 'median':

        comparison_df = comparison_df.median()

    elif x == 'sum':

        comparison_df = comparison_df.sum() 

    

    # Create database with either name of directors or actors, the value being compared i.e. 'gross',

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

plt.title("Total Gross for Directors with 4+ Films")

plt.ylabel("Director (no. films)")

plt.xlabel("Gross (in billons)")



plt.subplot2grid((2,3),(0,1), rowspan = 2)

create_comparison_database('director_name','gross','mean', 4).plot(kind='barh', color='#ffff00')

plt.legend().set_visible(False)

plt.title('Average revenue for Directors with 4+ Films')

plt.ylabel("Director (no. films)")

plt.xlabel("Gross (in billons)")



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

plt.xlabel("Budget (in billons)")



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