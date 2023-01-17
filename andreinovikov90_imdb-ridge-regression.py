# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/imdb-extensive-dataset/IMDb movies.csv")

rating = pd.read_csv("/kaggle/input/imdb-extensive-dataset/IMDb ratings.csv")

actors = pd.read_csv("/kaggle/input/imdb-extensive-dataset/IMDb names.csv")

titles = pd.read_csv("/kaggle/input/imdb-extensive-dataset/IMDb title_principals.csv")



print(actors.columns)

print(titles.columns)

print(rating.columns)

print(data.columns)
features = ['imdb_title_id', 'title', 'original_title', 'year', 'date_published',

       'genre', 'duration', 'country', 'language', 'director', 'writer',

       'production_company', 'actors', 'description', 'avg_vote', 'votes',

       'budget', 'usa_gross_income', 'worlwide_gross_income', 'metascore',

       'reviews_from_users', 'reviews_from_critics']

for feature in features:

    print(feature, data[feature].unique())
def get_currecy(x):

    if len(str(x).split())>1:

        return str(x).split()[0]

    else:

        return "$"



def get_amount(x):

    if len(str(x).split())>1:

        return float(str(x).split()[1])

    else:

        return float(x)



data["budget currecy"] = data["budget"].apply(get_currecy)

data["budget amount"] = data["budget"].apply(get_amount)

del data["budget"]

data["worlwide_gross_income currecy"] = data["worlwide_gross_income"].apply(get_currecy)

data["worlwide_gross_income amount"] = data["worlwide_gross_income"].apply(get_amount)

del data["worlwide_gross_income"]

data["usa_gross_income currecy"] = data["usa_gross_income"].apply(get_currecy)

data["usa_gross_income amount"] = data["usa_gross_income"].apply(get_amount)

del data["usa_gross_income"]







money = ["budget currecy", "budget amount", 

         "usa_gross_income currecy", "usa_gross_income amount", 

         "worlwide_gross_income currecy", "worlwide_gross_income amount"]



new_data = data[money][(data["budget currecy"] == "$") & 

                       (data["usa_gross_income currecy"] == "$") & 

                       (data["worlwide_gross_income currecy"] == "$")].dropna()



new_data["usa_income/budget ratio"] = new_data["usa_gross_income amount"].apply(float)/new_data["budget amount"].apply(float)

new_data["world_income/budget ratio"] = new_data["worlwide_gross_income amount"].apply(float)/new_data["budget amount"].apply(float)



data = data.merge(new_data)

col_names = data.columns

for name in col_names:

    if "currecy" in name:

        data = data[data[name]=="$"]

        del data[name]



data["duration"] = data["duration"].apply(float)
data["world_income/budget ratio"].hist()
print(data[data["world_income/budget ratio"]>500])
data = data[data["world_income/budget ratio"]<100]
data["world_income/budget ratio"].hist()
from math import log



data["log(usa_income/budget ratio)"] = data["usa_income/budget ratio"].apply(log)

data["log(usa_income/budget ratio)"].hist()

data["log(usa_income/budget ratio)"].describe()
data["log(world_income/budget ratio)"] = data["world_income/budget ratio"].apply(log)

data["log(world_income/budget ratio)"].hist()

data["log(world_income/budget ratio)"].describe()
import numpy as np

from statsmodels.distributions.empirical_distribution import ECDF

import matplotlib.pyplot as plt



ecdf = ECDF(data["world_income/budget ratio"])

plt.plot(ecdf.x,ecdf.y, label = "Эмпирическая функция распределения")

plt.plot(ecdf.x*0+1,ecdf.y, color="black")

plt.plot(ecdf.x,ecdf.y*0+0.42, color="black")

plt.show()
dynamics_mean = data.groupby("year").mean()

dynamics_std = data.groupby("year").std()



from matplotlib import pyplot as plt



plt.plot(dynamics_mean["world_income/budget ratio"], label="средняя эффективность")

plt.plot(dynamics_mean["log(usa_income/budget ratio)"]*0, color="black")

plt.legend()

plt.show()

plt.plot(dynamics_mean["world_income/budget ratio"]*0, color="black")

plt.plot(dynamics_std["world_income/budget ratio"], label="ср.кв. откл. эффективности")

plt.legend()

plt.show()

plt.plot(dynamics_mean["log(world_income/budget ratio)"], label="средняя логарифмическая эффективность")

plt.plot(dynamics_mean["log(usa_income/budget ratio)"]*0, color="black")

plt.legend()

plt.show()

plt.plot(dynamics_mean["world_income/budget ratio"]*0, color="black")

plt.plot(dynamics_std["log(world_income/budget ratio)"], label="ср.кв. откл. логарифмической эффективности")

plt.legend()

plt.show()

dynamics_mean = data.groupby("year").mean()

dynamics_std = data.groupby("year").std()



from matplotlib import pyplot as plt



plt.scatter(data["year"], data["log(world_income/budget ratio)"])

plt.plot(dynamics_mean["log(usa_income/budget ratio)"]*0, color="black")

plt.legend()

plt.show()

plt.scatter(data["year"], data["world_income/budget ratio"])

plt.plot(dynamics_mean["log(usa_income/budget ratio)"]*0+1, color="black")

plt.legend()

plt.show()
from matplotlib import pyplot as plt





plt.scatter(data["year"], data["log(world_income/budget ratio)"])

plt.plot(dynamics_mean["log(usa_income/budget ratio)"]*0, color="black")

plt.legend()

plt.show()
data.corr()
def collect_set(df_series, col_name):

    answer = set()

    for value in df_series[col_name].apply(lambda x: str(x).split(",")):

        for x in value:

            x = x.strip()

            if not (x in answer):

                answer.add(x)

    return answer



genres = collect_set(data, "genre")

countries = collect_set(data, "country")

languages = collect_set(data, "language")



def splitter(x):

    x = str(x).split(",")

    mt = []

    for word in x:

        mt.append(word.strip())

    return mt



print("Genres")

for genre in genres:

    length = len(data[data["genre"].apply(lambda x: int(genre in splitter(x)))==1])

    if length>100:

        print(genre, length)

print()

print("Countries")

for country in countries:

    length = len(data[data["country"].apply(lambda x: int(country in splitter(x)))==1])

    if length>100:

        print(country, length)

print()

print("Languages")

for language in languages:

    length = len(data[data["language"].apply(lambda x: int(language in splitter(x)))==1])

    if length>100:

        print(language, length)

Genres = ["Thriller", "Sci-Fi", "Adventure", "Crime", "Romance", "Comedy", "Horror", "Biography", "Sport",

          "Fantasy", "Animation", "Family", "Action", "Music", "Mystery", "Drama", "War", "History"]

Rus = ["Триллер", "Научная фантастика", "Приключения", "Криминал", "Романтический", "Комедия", 

       "Ужасы", "Биографический", "Спортивный", "Фэнтези", "Анимация", "Семейный", "Боевик", 

       "Музыкальный", "Мистический", "Драма", "Военный", "Исторический"]



Rus_Genres=dict()

for i in range(len(Genres)):

    Rus_Genres[Genres[i]] = Rus[i]

print(Rus_Genres)



for genre in Genres:

    data = data[data["year"]>=1980]

    loc_data = data[data["genre"].apply(lambda x: genre in splitter(x))]

    print(genre, "World", loc_data["log(world_income/budget ratio)"].mean(), loc_data["log(world_income/budget ratio)"].std())

    #print(genre, "US", loc_data["log(usa_income/budget ratio)"].mean(), loc_data["log(usa_income/budget ratio)"].std())

    dynamics_mean = loc_data.groupby("year").mean()

    dynamics_std = loc_data.groupby("year").std()

    plt.title = genre

    plt.plot(dynamics_mean["log(world_income/budget ratio)"], label=Rus_Genres[genre])

    #plt.plot(dynamics_mean["log(usa_income/budget ratio)"], label="log(US Income/Budget)")

    plt.plot(0*dynamics_mean["log(usa_income/budget ratio)"], color="black")

    plt.legend()

    plt.show()

    plt.scatter(loc_data["year"], loc_data["log(world_income/budget ratio)"])

    plt.plot(0*dynamics_mean["log(usa_income/budget ratio)"], color="black")

    plt.show()

    ##print(genre, collect_set(loc_data, "genre"))

    #print(genre, collect_set(loc_data, "country"))

    #print(genre, collect_set(loc_data, "language"))
Languages = ["Spanish", "Russian", "Arabic", "Mandarin", "English", "French", "Italian", "Japanese", "German"]

Rus = ["Испанский", "Русский", "Арабский", "Китайский", 

       "Английский", "Французский", "Итальянский", "Японский", "Немецкий"]

Rus_Languages=dict()

for i in range(len(Languages)):

    Rus_Genres[Languages[i]] = Rus[i]

print(Rus_Genres)





for language in Languages:

    loc_data = data[data["language"].apply(lambda x: language in splitter(x))]

    #if (loc_data1["world_income/budget ratio"].mean()<standard_deviation):

    #    loc_data1 = loc_data1[loc_data1["world_income/budget ratio"]<standard_deviation]

    #    standard_deviation = loc_data1["world_income/budget ratio"].std()

    #if (loc_data2["usa_income/budget ratio"].mean()<standard_deviation):

    #    loc_data1 = loc_data2[loc_data2["usa_income/budget ratio"]<standard_deviation]

    #    standard_deviation = loc_data2["usa_income/budget ratio"].std()

    dynamics_mean = loc_data.groupby("year").mean()

    dynamics_std = loc_data.groupby("year").std()

    print(language, "World", loc_data["log(world_income/budget ratio)"].mean(), loc_data["log(world_income/budget ratio)"].std())

    plt.ylim(top=3)

    plt.ylim(bottom=-3)

    plt.plot(dynamics_mean["log(world_income/budget ratio)"], label=Rus_Genres[language])

    plt.plot(dynamics_mean["log(world_income/budget ratio)"]*0, color="black")

    plt.legend()

    plt.show()

Countries = ["Spain", "UK", "France", "USA", "Australia", "Mexico", "China", "Japan", "Germany", "Hong Kong",

             "Canada", "Italy"]

Rus = ["Испания", "Великобритания", "Франция", "США",  "Австралия",

       "Мексика", "Китай", "Япония", "Германия", "Гонконг", "Канада", "Италия"]

Rus_Countries=dict()

for i in range(len(Countries)):

    Rus_Countries[Countries[i]] = Rus[i]

print(Rus_Countries)



for country in Countries:

    loc_data = data[data["country"].apply(lambda x: country in splitter(x))]

    dynamics_mean = loc_data.groupby("year").mean()

    dynamics_std = loc_data.groupby("year").std()

    print(country, "World", loc_data["log(world_income/budget ratio)"].mean(), loc_data["log(world_income/budget ratio)"].std())

    plt.plot(dynamics_mean["log(world_income/budget ratio)"], label=Rus_Countries[country])

    plt.plot(0*dynamics_mean["log(usa_income/budget ratio)"])

    plt.legend()

    plt.show()

Directors = collect_set(data, "director")

directors_df = pd.DataFrame()

for director in Directors:

    loc_data = data[data["director"].apply(lambda x: director in splitter(x))]

    if len(loc_data)>3:

        loc_data.loc[:,("Director_for_grouping")] = director

        directors_df = pd.concat([directors_df, loc_data])

del directors_df["imdb_title_id"]
from math import floor

from scipy.special import softmax



def count_rating_max(x):

    answer = np.dot(np.array(x), softmax(x))

    return answer



def count_rating(x):

    answer = np.mean(x)

    return answer



def count_rating_min(x):

    answer = np.dot(np.array(x), softmax(-np.array(x)))

    return answer





new_directors_df = directors_df.groupby("Director_for_grouping")

new_directors_df = new_directors_df.agg(lambda x: list(x))

new_directors_df = new_directors_df.reset_index()

new_directors_df["rating mean"] = new_directors_df["log(world_income/budget ratio)"].apply(count_rating)

new_directors_df["rating mean"].hist()

for i in range(len(new_directors_df)):

    X = new_directors_df["year"].loc[i]

    Y = [new_directors_df["rating mean"].loc[i]]*len(X)

    plt.scatter(X,Y)

plt.plot([0]*len(data["year"]))

plt.xlim(left=1980)

plt.xlim(right=2020)

plt.ylim(bottom=-5)

plt.ylim(top=5)

plt.show()
features = pd.DataFrame()

Producer = collect_set(data, "production_company")

for production_company in Producer:

    prod = pd.DataFrame(data["production_company"].apply(lambda x: production_company in splitter(x)))

    prod[production_company] = prod["production_company"] 

    del prod["production_company"] 

    #loc_data = data[prod]

    features = pd.concat([features, prod], axis=1)
Writers = collect_set(data, "writer")

for writer in Writers:

    prod = pd.DataFrame(data["writer"].apply(lambda x: writer in splitter(x)))

    prod[writer] = prod["writer"] 

    del prod["writer"] 

    #loc_data = data[prod]

    features = pd.concat([features, prod], axis=1)
Directors = collect_set(data, "director")

for director in Directors:

    direc = pd.DataFrame(data["director"].apply(lambda x: director in splitter(x)))

    direc[director] = direc["director"] 

    del direc["director"] 

    #loc_data = data[prod]

    features = pd.concat([features, direc], axis=1)
features.to_csv("boolean.csv")
features = features.applymap(lambda x: int(x))
features.to_csv("boolean_maped.csv")
target = data["log(world_income/budget ratio)"]
from math import sqrt



def criteria(x):

    if x==1:

        return True

    else:

        return x/sqrt(1-x*x)*sqrt(l-2)>1.96



backup = data.copy()

l = len(data)

del data["log(usa_income/budget ratio)"]

del data["usa_income/budget ratio"]

del data["usa_gross_income amount"]

del data["reviews_from_users"]

del data["reviews_from_critics"]

del data["metascore"]

del data["votes"]

del data["avg_vote"]

cor = data.corr()

cor
features = pd.concat([features, data["budget amount"]], axis=1)

features = pd.concat([features, data["duration"]], axis=1)
#del features["budget amount"]

del features["duration"]
from sklearn.linear_model import Ridge



from sklearn.model_selection import train_test_split



model = Ridge()

m = []

for i in range(10):

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

    model.fit(X_train, y_train)

    print(0.2, model.score(X_test, y_test))

    m.append(model.score(X_test, y_test))

print(np.mean(m))

m = []

for i in range(10):

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.1)

    model.fit(X_train, y_train)

    print(0.1, model.score(X_test, y_test))    

    m.append(model.score(X_test, y_test))

print(np.mean(m))

m=[]

for i in range(10):

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.05)

    model.fit(X_train, y_train)

    print(0.05, model.score(X_test, y_test))

    m.append(model.score(X_test, y_test))

print(np.mean(m))
dataframe = features.corrwith(target).apply(criteria)

for i in dataframe[dataframe].index:

    print(i, dataframe.loc[i])
from math import exp



output_df = pd.DataFrame(features.columns)

output_df["factor"] = output_df[0]

del output_df[0]

output_df = pd.concat([output_df, pd.DataFrame(model.coef_)], axis=1)

output_df["coef"] = output_df[0]

del output_df[0]

output_df["a's"] = output_df["coef"].apply(exp)

print(output_df)
len(data)
new_features = features[dataframe[dataframe].index]

new_features = pd.concat([new_features, data["budget amount"]], axis=1)
from sklearn.linear_model import Ridge



from sklearn.model_selection import train_test_split



model = Ridge()

m = []

for i in range(10):

    X_train, X_test, y_train, y_test = train_test_split(new_features, target, test_size=0.2)

    model.fit(X_train, y_train)

    print(0.2, model.score(X_test, y_test))

    m.append(model.score(X_test, y_test))

print(np.mean(m))

m = []

for i in range(10):

    X_train, X_test, y_train, y_test = train_test_split(new_features, target, test_size=0.1)

    model.fit(X_train, y_train)

    print(0.1, model.score(X_test, y_test))    

    m.append(model.score(X_test, y_test))

print(np.mean(m))

m=[]

for i in range(10):

    X_train, X_test, y_train, y_test = train_test_split(new_features, target, test_size=0.05)

    model.fit(X_train, y_train)

    print(0.05, model.score(X_test, y_test))

    m.append(model.score(X_test, y_test))

print(np.mean(m))
from math import exp



output_df = pd.DataFrame(new_features.columns)

output_df["factor"] = output_df[0]

del output_df[0]

output_df = pd.concat([output_df, pd.DataFrame(model.coef_)], axis=1)

output_df["coef"] = output_df[0]

del output_df[0]

output_df["a's"] = output_df["coef"].apply(exp)

print(output_df)
for i in output_df.sort_values(by="coef").index:

    print(i, output_df["factor"].loc[i], output_df["a's"].loc[i])
Actors = collect_set(data, "actors")
l = len(Actors)

i = 0

for actors in Actors:

    act = pd.DataFrame(data["actors"].apply(lambda x: actors in splitter(x)))

    act[actors] = act["actors"] 

    del act["actors"] 

    act = act.corrwith(target).apply(criteria).loc[actors]

    if act:

        print(actors, ";", act, ";", i, l)

    i+=1
influencial_actors = pd.read_csv("/kaggle/input/inf-actors/IMDB_Influencial_actors.csv", sep=";", header=None)

print(influencial_actors)
features_2 = pd.DataFrame()

i=0

for actor in list(influencial_actors[0]):

    actors = pd.DataFrame(data["actors"].apply(lambda x: actor in splitter(x)))

    actors[actor] = actors["actors"]

    del actors["actors"]

    features_2 = pd.concat([features_2, actors], axis=1)

    print(i, end="; ")

    i += 1

    if i%20==0:

        print()

print(features_2)

    
features_2.to_csv("boolean2.csv")
features_2.applymap(lambda x: int(x)).to_csv("boolean2_maped.csv")
features_2 = pd.read_csv("boolean2_maped.csv")
from sklearn.linear_model import Ridge



from sklearn.model_selection import train_test_split



model = Ridge()

m = []

for i in range(10):

    X_train, X_test, y_train, y_test = train_test_split(features_2, target, test_size=0.2)

    model.fit(X_train, y_train)

    print(0.2, model.score(X_test, y_test))

    m.append(model.score(X_test, y_test))

print(np.mean(m))

m = []

for i in range(10):

    X_train, X_test, y_train, y_test = train_test_split(features_2, target, test_size=0.1)

    model.fit(X_train, y_train)

    print(0.1, model.score(X_test, y_test))    

    m.append(model.score(X_test, y_test))

print(np.mean(m))

m=[]

for i in range(10):

    X_train, X_test, y_train, y_test = train_test_split(features_2, target, test_size=0.05)

    model.fit(X_train, y_train)

    print(0.05, model.score(X_test, y_test))

    m.append(model.score(X_test, y_test))

print(np.mean(m))
features_genre = pd.DataFrame()

for genre in Genres:

    features_genre[genre] = data["genre"].apply(lambda x: genre in splitter(x))

features_genre = features_genre.applymap(lambda x: int(x))
features_3 = pd.concat([features, features_genre], axis=1)

from sklearn.linear_model import Ridge



from sklearn.model_selection import train_test_split



model = Ridge()

m = []

for i in range(10):

    X_train, X_test, y_train, y_test = train_test_split(features_3, target, test_size=0.2)

    model.fit(X_train, y_train)

    print(0.2, model.score(X_test, y_test))

    m.append(model.score(X_test, y_test))

print(np.mean(m))

m = []

for i in range(10):

    X_train, X_test, y_train, y_test = train_test_split(features_3, target, test_size=0.1)

    model.fit(X_train, y_train)

    print(0.1, model.score(X_test, y_test))    

    m.append(model.score(X_test, y_test))

print(np.mean(m))

m=[]

for i in range(10):

    X_train, X_test, y_train, y_test = train_test_split(features_3, target, test_size=0.05)

    model.fit(X_train, y_train)

    print(0.05, model.score(X_test, y_test))

    m.append(model.score(X_test, y_test))

print(np.mean(m))
from math import exp



output_df = pd.DataFrame(features_3.columns)

output_df["factor"] = output_df[0]

del output_df[0]

output_df = pd.concat([output_df, pd.DataFrame(model.coef_)], axis=1)

output_df["coef"] = output_df[0]

del output_df[0]

output_df["a's"] = output_df["coef"].apply(exp)

print(output_df)
print(output_df.loc[12186:12204])
for genre in Genres:

    print(genre, (target-target.mean() - 0.163059*features_3[genre]).apply(lambda x: x**2).sum()/target.apply(lambda x: x**2).sum())

from nltk.corpus import stopwords #we must reduce the stopwords from the text



data["description"] = data["description"].apply(lambda x: str(x).lower())



print("1 OK!")



stopWords = set(stopwords.words('english')) 



for sep in [".", ",", "!", ")", "(", "&", "*", "_", "-", "+", "=", "\'", "\"", "|", ":", ";", "/"]:

    data["description"] = data["description"].apply(lambda x: x.replace(sep, " "))



print("2 OK!")

    

def trim(x):

    while "  " in x:

        x = x.replace("  ", " ")

        print(x)

    return x



print("3 OK!")



data["description"] = data["description"].apply(trim)



print("4 OK!")



for w in stopWords:

    data["description"] = data["description"].apply(lambda x: x.replace(" "+w+" ", " "))



print("5 OK!")

    

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()



print("6 OK!")



tfidf_train = vectorizer.fit_transform(data["description"])



print("7 OK!")



from sklearn.decomposition import PCA

pca = PCA(n_components=500)

lower_dim_tfidf_train = pca.fit_transform(tfidf_train.todense())
lower_dim_tfidf_train
train_df = lower_dim_tfidf_train



from sklearn.model_selection import train_test_split



train_features = train_df.copy()



X_train, X_test, y_train, y_test = train_test_split(train_features, target, test_size=0.2)



#and find precision and recall, not only the accuracy.

from sklearn.metrics import precision_recall_fscore_support



from sklearn.model_selection import train_test_split



model = Ridge()



X_train, X_test, y_train, y_test = train_test_split(train_features, target, test_size=0.2)

model.fit(X_train, y_train)

print(0.2, model.score(X_test, y_test))

print(model.coef_)



X_train, X_test, y_train, y_test = train_test_split(train_features, target, test_size=0.1)

model.fit(X_train, y_train)

print(0.1, model.score(X_test, y_test))

print(model.coef_)



X_train, X_test, y_train, y_test = train_test_split(train_features, target, test_size=0.05)

model.fit(X_train, y_train)

print(0.05, model.score(X_test, y_test))

print(model.coef_)