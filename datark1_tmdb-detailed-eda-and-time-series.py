import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os  # operating system library

import matplotlib.pyplot as plt  # basic visualisation library

import seaborn as sns  # advanced visualisations

import ast  # Abstact Syntax Trees

import itertools  # Iterating tools

import re  # Regular Expressions
train = pd.read_csv('../input/train.csv')

train.head(4)
shape = train.shape

print("rows: {}\ncolumns: {}".format(shape[0], shape[1]))
# calculate percentage of missing values

pct_nans = round(train.isnull().sum()/shape[0]*100,1).to_frame().sort_values(by=[0], ascending=False)

# create a bar chart

plt.figure(figsize=(20,8))

sns.barplot(x=pct_nans.index, y=pct_nans[0])

plt.axhline(10, ls="--")

plt.xticks(rotation=90, fontsize=13)

plt.title("Percentage of missing values", fontsize=13)

plt.ylabel("Missing values [%]", fontsize=13)

plt.show()
def clean_dates(row):

    '''

    This function cleans release_date row. 

    '''

    text = row["release_date"]

    yr = re.findall(r"\d+/\d+/(\d+)",text)



    if int(yr[0]) >= 18:

        return(text[:-2] + "19" + yr[0])

    else:

        return(text[:-2] + "20" + yr[0])

    
train["release_date"] = train.apply(clean_dates, axis=1)    # applying cleaning function

train["release_date"] = pd.to_datetime(train["release_date"])    # converting release_date column to datetime type
check = train[train["release_date"].dt.year==2017]

check[["id","title","release_date","runtime","budget","revenue"]].head()
first_date = train["release_date"].min()

last_date = train["release_date"].max()



first_movie = train[train["release_date"]==first_date]

first_movie[["id","title","release_date","runtime","budget","revenue","poster_path"]]
url = "https://image.tmdb.org/t/p/original"+first_movie["poster_path"].values[0]

url
last_movie = train[train["release_date"]==last_date]

last_movie[["id","title","release_date","runtime","budget","revenue","poster_path"]]
url = "https://image.tmdb.org/t/p/original"+last_movie["poster_path"].values[0]

url
train.loc[:,"release_year"] = train.loc[:,"release_date"].dt.year

train.loc[:,"release_month"] = train.loc[:,"release_date"].dt.month

movies_2017 = train[train["release_year"]==2017]

movies_2017[["id","title","release_date","runtime","budget","revenue"]].head(10)
movies_2018 = train[train["release_year"]==2018]

movies_2018[["id","title","release_date","runtime","budget","revenue"]].head(10)
TS = train.loc[:,["original_title","release_date","budget","runtime","revenue","release_year","release_month"]]

#TS = train.copy()

TS.dropna()

TS.head()
plt.figure(figsize=(25,10))

sns.countplot(x="release_year", data=TS)

plt.xticks(rotation=70, fontsize=12)

plt.xlabel("Release Year", size=15)

plt.ylabel("Count", size=15)

plt.show()
plt.figure(figsize=(16,8))

ax = sns.boxplot(x="release_year", y="revenue", data = train)

ax.set_title("Revenue of  movies between 1969-2017")

plt.xticks(rotation=70)

plt.text(2,1.35e9,"Max revenue: {:,.0f} $ ('{}')".format(train["revenue"].max(), train.loc[TS["revenue"].idxmax(),"original_title"]), fontweight="bold")

plt.text(2,1.4e9,"Mean revenue: {:,.0f} $".format(train["revenue"].mean()), fontweight="bold")

plt.xlabel("Release Year", size=15)

plt.ylabel("Revenue", size=15)

plt.show()
zero_budget = train[train["budget"]==0]

print(len(zero_budget))

zero_budget[["title","release_date","budget"]].head(10)
train.loc[train['id'] == 7,'budget'] = 60000 # Control Room 

train.loc[train['id'] == 8,'budget'] = 50000000 # Muppet Treasure Island

train.loc[train['id'] == 17,'budget'] = 12000000 # The Invisible Woman

train.loc[train['id'] == 11,'budget'] = 10000000 # Revenge of the Nerds II:

train.loc[train['id'] == 31,'budget'] = 8000000 # Caché

train.loc[train['id'] == 38,'budget'] = 4000000 # Final: The Rapture

train.loc[train['id'] == 48,'budget'] = 5000000 # Wilson

train.loc[train['id'] == 52,'budget'] = 800000 # The Last Flight of Noah's Ark

train.loc[train['id'] == 53,'budget'] = 100000000 # For Keeps

train.loc[train['id'] == 55,'budget'] = 200000000 # Son in Law

train.loc[train['id'] == 56,'budget'] = 50000000 # Queen to Play

train.loc[train['id'] == 62,'budget'] = 46000000 # Fallen

train.loc[train['id'] == 67,'budget'] = 4000000 # A Touch of Sin

train.loc[train['id'] == 73,'budget'] = 19000000 # Czech Dream

train.loc[train['id'] == 89,'budget'] = 30000000 # Sommersby

train.loc[train['id'] == 91,'budget'] = 75000000 # His Secret Life

train.loc[train['id'] == 93,'budget'] = 8500 # Hunger

train.loc[train['id'] == 97,'budget'] = 300 # Target

train.loc[train['id'] == 102,'budget'] = 14000000 # Going by the Book

train.loc[train['id'] == 103,'budget'] = 19000000 # Won't Back Down

train.loc[train['id'] == 116,'budget'] = 2400000 # Back to 1942

train.loc[train['id'] == 117,'budget'] = 60000000 # Wild Hogs

train.loc[train['id'] == 118,'budget'] = 2000000 # Boxing Helena

train.loc[train['id'] == 126,'budget'] = 9000000 # Corvette Summer

train.loc[train['id'] == 141,'budget'] = 12000000 # Girl with a Pearl Earring

train.loc[train['id'] == 146,'budget'] = 14000000 # Police Academy 5

train.loc[train['id'] == 148,'budget'] = 18000000 # Beethoven
train.loc[:,"release_year"] = train.loc[:,"release_date"].dt.year

train.loc[:,"release_month"] = train.loc[:,"release_date"].dt.month
years = train.loc[:,"release_year"].unique()



for year in years:

    year_mean = train[(train["release_year"]==year) & (train["budget"]!=0)]["budget"].mean()

    if year_mean != np.nan:

        train[train["release_year"]==year]  = train[train["release_year"]==year].replace({"budget": 0}, year_mean)
pd.set_option('display.float_format', lambda x: '%.0f' % x)

train.iloc[zero_budget.index,:][["title","budget"]].head(10)
plt.figure(figsize=(16,8))

ax = sns.boxplot(x="release_year", y="budget", data = TS)

ax.set_title("Budgets of  movies between 1969-2017")

plt.xticks(rotation=70)

plt.text(2,3.4e8,"Mean budget: {:,.0f} $".format(train["budget"].mean()), fontweight="bold")

plt.text(2,3.3e8,"Max budget: {:,.0f} $ ('{}')".format(train["budget"].max(), train.loc[TS["budget"].idxmax(),"original_title"]), fontweight="bold")

plt.show()
b_zeros = TS[TS.budget==0]

b_zeros.head()
plt.figure(figsize=(16,8))

ax = sns.boxplot(x="release_year", y="runtime", data = train)

ax.set_title("Budgets of  movies between 1969-2017")

plt.xticks(rotation=70)

plt.text(2,300,"Mean runtime: {:.0f} minutes".format(train["runtime"].mean()), fontweight="bold")

plt.text(2,280,"Max runtime: {:.0f} minutes ('{}')".format(TS["runtime"].max(), train.loc[TS["runtime"].idxmax(),"original_title"]), fontweight="bold")

plt.show()
r_zeros = train[train.runtime==0][["id","title"]]

r_zeros.head()
TS_clean_runtime = train.drop(train[train.runtime==0].index)

plt.figure(figsize=(16,8))

ax = sns.boxplot(x="release_year", y="runtime", data = train)

ax.set_title("Budgets of  movies between 1969-2017")

plt.xticks(rotation=70)

plt.text(2,300,"Mean runtime: {:.0f} minutes".format(train["runtime"].mean()), fontweight="bold")

plt.text(2,280,"Max runtime: {:.0f} minutes ('{}')".format(TS_clean_runtime["runtime"].max(), TS_clean_runtime.loc[train["runtime"].idxmax(),"original_title"]),

         fontweight="bold")

plt.show()
fig = sns.PairGrid(train[['runtime', 'budget', 'popularity','revenue']].dropna())

# define top, bottom and diagonal plots

fig.map_upper(plt.scatter, color='purple')

fig.map_lower(sns.kdeplot, cmap='cool_d')

fig.map_diag(sns.distplot, bins=30);
train.plot(x="runtime",y="budget", kind="scatter",figsize=(12,8))

runtime_mean = np.mean(train.runtime)

plt.axvline(x=runtime_mean, c="red", linestyle="--")

plt.text(runtime_mean-10, 3.5e8,"mean: {0:0.1f}".format(runtime_mean), color="red", rotation=90)

plt.ylim([0,4e8])

plt.show()
plt.figure(figsize=(12,7))

ax = sns.distplot(train["runtime"].dropna(),hist_kws={"rwidth":0.75,"alpha": 0.6, "color": "g"})

plt.axvline(x=runtime_mean, c="red", linestyle="--")

plt.text(runtime_mean+5, 0.025,"mean: {0:0.1f}".format(runtime_mean), color="red", rotation=90)

plt.show()
train.plot(x="popularity",y="budget", kind="scatter",figsize=(12,7))

pop_mean = np.mean(train.popularity)

plt.axvline(x=pop_mean, c="red", linestyle="--")

plt.text(pop_mean-10, 3.5e8,"mean: {0:0.1f}".format(pop_mean), color="red", rotation=90)

plt.ylim([0,4e8])

plt.show()
pop = train[train.popularity<50]

pop.plot(x="popularity",y="budget", kind="scatter",figsize=(12,7))

plt.axvline(x=pop_mean, c="red", linestyle="--")

plt.text(pop_mean-1, 3.5e8,"mean: {0:0.1f}".format(pop_mean), color="red", rotation=90)

plt.ylim([0,4e8])

plt.show()
top3 = train.popularity.nlargest(3)

idx3 = top3.index.tolist()

train.iloc[idx3,[7,9,13]]
top3 = train.budget.nlargest(3)

idx3 = top3.index.tolist()

train.iloc[idx3,[7,2,13]]
pop.plot(x="popularity",y="revenue", kind="scatter",figsize=(12,8))

plt.show()
top3 = train.revenue.nlargest(3)

idx3 = top3.index.tolist()

train.iloc[idx3,[7,13,22]]
sns.clustermap(TS.corr(), annot=True, linewidths=.5, fmt= '.2f')

plt.show()
train.original_language.nunique()
plt.figure(figsize=(15,5))

sns.countplot(train['original_language'].sort_values())

plt.show()
non_english = train['original_language'][train['original_language'] != "en"].value_counts()

plt.figure(figsize=(15,5))

sns.barplot(x=non_english.index, y=non_english.values)

plt.title("Number of non-english movies per language")

plt.show()
prod_c = train["production_companies"].to_frame()



prod_c.dropna(inplace=True)



# convert the strings into lists

prod_c["production_companies"] = prod_c["production_companies"].apply(lambda x: ast.literal_eval(x))



prod_c_clean = pd.DataFrame(list(itertools.chain(*prod_c["production_companies"].values.tolist())))

prod_c_clean.head()
tmp1 = prod_c_clean["name"].value_counts()

TOP20 = tmp1[:20].to_frame()

TOP20
plt.figure(figsize=(16,6))

ax = sns.barplot(x=TOP20.index, y="name", data=TOP20)

plt.title("Number of movies produced by companies")

plt.xticks(rotation=90)

plt.show()
genres = train.loc[:,["id","original_title","genres", "release_date","release_year"]]

genres.loc[:,"genres"] = genres.loc[:,"genres"].fillna("None")

genres.head(5)
def extract_genres(row):

    if row == "None":

        return "None"

    else:

        results = re.findall(r"'name': '(\w+\s?\w+)'", row)

        return results
genres["genres"] = genres["genres"].apply(extract_genres)

genres["genres"].head(10)
unique_genres = genres["genres"].apply(pd.Series).stack().unique()

print("Number of genres: {}".format(len(unique_genres)))

print("Genres: {}".format(unique_genres))
genres_dummies = pd.get_dummies(genres["genres"].apply(pd.Series).stack()).sum(level=0)

genres_dummies.head()
train_genres = pd.concat([train, genres_dummies],axis=1, sort=False)

train_genres.head(5)
genres_overall = train_genres[unique_genres].sum().sort_values(ascending=False)

plt.figure(figsize=(15,5))

ax = sns.barplot(x=genres_overall.index, y=genres_overall.values)

plt.xticks(rotation=90)

plt.title("Popularity of genres overall")

plt.ylabel("count")

plt.show()
genres_by_years = train_genres.groupby("release_year")[unique_genres].sum()

other_genres = ["TV Movie", "None", "Foreign", "Western", "Documentary", "War","Music","History", "Animation","Mystery"]

genres_by_years["Others"] = genres_by_years.loc[:,other_genres].sum(axis=1)

genres_by_years.drop(other_genres, axis=1, inplace=True)

genres_by_years.head()
plt.figure(figsize=(17,10))

sns.lineplot(data=genres_by_years[:-1], dashes=False)

plt.title("Popularity of genres over years")

plt.show()
genres_by_years_5yrs = genres_by_years.rolling(6).sum()

plt.figure(figsize=(17,10))

sns.lineplot(data=genres_by_years_5yrs[:-1], dashes=False, palette ="Paired")

plt.title("Popularity of genres over years (smoothed)")

plt.show()
cast = train.loc[:,["id","original_title","cast","release_date"]]

cast.loc[:,"cast"] = train.loc[:,"cast"].fillna("None")

cast.head(10)
def extract_cast(row):

    if row == "None":

        return "None"

    else:

        results = re.findall(r"\'name\': \'(\w+\s\w+-?\w+)'", row)

        return results
cast["cast"] = cast["cast"].apply(extract_cast)

cast.head()
# number of unique actors

unique_actors = cast["cast"].apply(pd.Series).stack().unique()

print("Number of unique actors: {}".format(len(unique_actors)))
actors = cast["cast"].apply(pd.Series).stack().value_counts()

actors.head(20)
act_mov = []

mov = range(1,27)

for i in mov:

    actors_movies = actors[actors.values >i]

    act_mov.append(len(actors_movies))
plt.figure(figsize=(16,8))

sns.barplot(x=list(mov), y=act_mov)

plt.xlabel("Number of movies played in")

plt.show()
actors_top30 = actors[:30]



cast = cast["cast"].apply(pd.Series).stack()



def masking(actor):

    if actor in actors_top30:

        return actor

    else:

        return "Other"

    

cast = cast.apply(masking)

cast_dummies = pd.get_dummies(cast.apply(pd.Series).stack()).sum(level=0)

cast_dummies["Other"][cast_dummies["Other"]>1] = 1

cast_dummies.head()