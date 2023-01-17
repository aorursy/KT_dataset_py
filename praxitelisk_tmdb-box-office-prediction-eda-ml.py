import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))





import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')



import gc



from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import warnings

warnings.filterwarnings("ignore")
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

sub_df = pd.read_csv('../input/sample_submission.csv')
train_df.head()
train_df.columns
train_df.isna().sum()
test_df.isna().sum()
train_df.columns
f, ax = plt.subplots(3, figsize=(12,7))



sns.set(rc={'figure.figsize':(12,8)})

sns.boxplot(x=train_df.revenue, ax = ax[0])

ax[0].set_title("revenue Boxplot")

sns.distplot(a=train_df.revenue, kde = False, ax = ax[1])

ax[1].set_title("revenue Histogram")

sns.distplot(a=np.log1p(train_df.revenue), kde = False, ax = ax[2])

ax[2].set_title("Log1p transformed revenue Histogram")

f.tight_layout()



train_df["log_revenue"] = np.log1p(train_df.revenue)





wordcloud = WordCloud().generate(train_df.title.to_string())



sns.set(rc={'figure.figsize':(12,8)})



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
train_df["title"] = train_df["title"].fillna("")

test_df["title"] = test_df["title"].fillna("")



train_df["title_len"] = train_df["title"].apply(len)

test_df["title_len"] = test_df["title"].apply(len)



f, ax = plt.subplots(3, figsize=(12,7))

sns.set(rc={'figure.figsize':(12,8)})

sns.boxplot(x=train_df.title_len, ax = ax[0])

ax[0].set_title("titles' length Boxplot")

sns.distplot(a=train_df.title_len, kde = False, ax = ax[1])

ax[1].set_title("titles' length Histogram")

sns.distplot(a=np.log1p(train_df.title_len), kde = False, ax = ax[2])

ax[2].set_title("Log1p transformed titles' length Histogram")

f.tight_layout()



train_df["log_title_len"] = np.log1p(train_df.title_len)

test_df["log_title_len"] = np.log1p(test_df.title_len)
wordcloud = WordCloud().generate(train_df.overview.to_string())



sns.set(rc={'figure.figsize':(12,8)})



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
train_df["overview"] = train_df["overview"].fillna("")

test_df["overview"] = test_df["overview"].fillna("")



train_df["overview_len"] = train_df["overview"].apply(len)

test_df["overview_len"] = test_df["overview"].apply(len)



f, ax = plt.subplots(3, figsize=(12,7))

sns.set(rc={'figure.figsize':(12,8)})

sns.boxplot(x=train_df.overview_len, ax = ax[0])

ax[0].set_title("overview' length Boxplot")

sns.distplot(a=train_df.overview_len, kde = False, ax = ax[1])

ax[1].set_title("overview' length Histogram")

sns.distplot(a=np.log1p(train_df.overview_len), kde = False, ax = ax[2])

ax[2].set_title("Log1p transformed overview' length Histogram")

f.tight_layout()



train_df["log_overview_len"] = np.log1p(train_df.overview_len)

test_df["log_overview_len"] = np.log1p(test_df.overview_len)
wordcloud = WordCloud().generate(train_df.tagline.to_string())



sns.set(rc={'figure.figsize':(12,8)})



# Display the generated image:

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
train_df["tagline"] = train_df["tagline"].fillna("")

test_df["tagline"] = test_df["tagline"].fillna("")



train_df["tagline_len"] = train_df["tagline"].apply(len)

test_df["tagline_len"] = test_df["tagline"].apply(len)



f, ax = plt.subplots(3, figsize=(12,7))

sns.set(rc={'figure.figsize':(12,8)})

sns.boxplot(x=train_df.tagline_len, ax = ax[0])

ax[0].set_title("tagline_len' length Boxplot")

sns.distplot(a=train_df.tagline_len, kde = False, ax = ax[1])

ax[1].set_title("tagline_len' length Histogram")

sns.distplot(a=np.log1p(train_df.tagline_len), kde = False, ax = ax[2])

ax[2].set_title("Log1p transformed tagline_len' length Histogram")

f.tight_layout()



train_df["log_tagline_len"] = np.log1p(train_df.tagline_len)

test_df["log_tagline_len"] = np.log1p(test_df.tagline_len)
f, ax = plt.subplots(3, figsize=(12,7))



sns.set(rc={'figure.figsize':(12,8)})

sns.boxplot(x=train_df.budget, ax = ax[0])

ax[0].set_title("budget Boxplot")

sns.distplot(a=train_df.budget, kde = False, ax = ax[1])

ax[1].set_title("budget Histogram")

sns.distplot(a=np.log1p(train_df.budget), kde = False, ax = ax[2])

ax[2].set_title("Log1p transformed budget Histogram")

f.tight_layout()



train_df["log_budget"] = np.log1p(train_df.budget)

test_df["log_budget"] = np.log1p(test_df.budget)
def genres_preprocessing(elem):

    string = str(elem)

    str1 = string.replace(']','').replace('[','').replace('{','').replace('}','').replace('\'','').replace(' ','').replace("name", "").replace("id", "").replace(":", "")

    ll = str1.split(",")[1::2]

    return ll



train_df["genres_processed"] = train_df.genres.apply(lambda elem: genres_preprocessing(elem))

test_df["genres_processed"] = test_df.genres.apply(lambda elem: genres_preprocessing(elem))



genres_dict = dict()



for genre in train_df["genres_processed"]:

    for elem in genre:

        if elem not in genres_dict:

            genres_dict[elem] = 1

        else:

            genres_dict[elem] += 1





sns.set(rc={'figure.figsize':(12,8)})

genres_df = pd.DataFrame.from_dict(genres_dict, orient='index')

genres_df.columns = ["number_of_movies"]

genres_df = genres_df.sort_values(by="number_of_movies", ascending=False)

genres_df.plot.bar()

plt.title("Number of films per genre")
sns.set(rc={'figure.figsize':(9,8)})

train_df['num_genres'] = train_df['genres_processed'].apply(lambda x: len(x) if x != {} else 0)

test_df['num_genres'] = test_df['genres_processed'].apply(lambda x: len(x) if x != {} else 0)



train_df['num_genres'].value_counts().plot.bar()

plt.title("Number of films with more than 1 genre")
genres_df.index.values

for g in genres_df.index.values:

    train_df['isGenre_' + g] = train_df['genres_processed'].apply(lambda x: 1 if g in x else 0)

    test_df['isGenre_' + g] = test_df['genres_processed'].apply(lambda x: 1 if g in x else 0)
train_df.columns
train_df.original_language.value_counts()[:10].plot.bar()

plt.title("Number of films per language")


train_df["is_english_language"] = train_df.original_language.apply(lambda x: 1 if x == "en" else 0)

test_df["is_english_language"] = test_df.original_language.apply(lambda x: 1 if x == "en" else 0)



train_df.is_english_language = train_df.is_english_language.fillna(1)

test_df.is_english_language = test_df.is_english_language.fillna(1)



sns.set(rc={'figure.figsize':(12,8)})

ax = sns.countplot(x="is_english_language", data=train_df)
def production_companies_preprocessing(elem):

    string = str(elem)

    str1 = string.replace(']','').replace('[','').replace('{','').replace('}','').replace(' ','').replace("name", "").replace("id", "").replace(":", "").replace("\'", "")

    ll = str1.split(",")[0::2]

    return ll



train_df["production_companies"] = train_df.production_companies.fillna('NoProductionCompany')

train_df["production_companies"] = test_df.production_companies.fillna('NoProductionCompany')



train_df["production_companies_processed"] = train_df.production_companies.apply(lambda elem: production_companies_preprocessing(elem))

test_df["production_companies_processed"] = test_df.production_companies.apply(lambda elem: production_companies_preprocessing(elem))







production_companies_dict = dict()



for production_company in train_df["production_companies_processed"]:

    for elem in production_company:

        if elem not in production_companies_dict:

            production_companies_dict[elem] = 1

        else:

            production_companies_dict[elem] += 1





sns.set(rc={'figure.figsize':(12,8)})

production_companies_df = pd.DataFrame.from_dict(production_companies_dict, orient='index')

production_companies_df.columns = ["number_of_movies"]

production_companies_df = production_companies_df.sort_values(by="number_of_movies", ascending=False)

production_companies_df.head(10).plot.bar()

plt.title("Number of films per production company")
train_df["num_of_production_companies"] = train_df.production_companies_processed.apply(len)

test_df["num_of_production_companies"] = test_df.production_companies_processed.apply(len)



train_df["num_of_production_companies"].value_counts().plot.bar()

plt.title("Number of multiple production companies per movie")
for g in production_companies_df.index.values:

    train_df['isProductionCompany_' + g] = train_df['production_companies_processed'].apply(lambda x: 1 if g in x else 0)

    test_df['isProductionCompany_' + g] = test_df['production_companies_processed'].apply(lambda x: 1 if g in x else 0)
def production_countries_preprocessing(elem):

    string = str(elem)

    str1 = string.replace(']','').replace('[','').replace('{','').replace('}','').replace(' ','').replace("name", "").replace("iso_3166_1", "").replace(":", "").replace("\'", "")

    ll = str1.split(",")[0::2]

    return ll



train_df["production_countries_processed"] = train_df.production_countries.fillna("NaN").apply(lambda elem: production_countries_preprocessing(elem))

test_df["production_countries_processed"] = test_df.production_countries.fillna("NaN").apply(lambda elem: production_countries_preprocessing(elem))





production_countries_dict = dict()



for production_country in train_df["production_countries_processed"]:

    for elem in production_country:

        if elem not in production_countries_dict:

            production_countries_dict[elem] = 1

        else:

            production_countries_dict[elem] += 1







production_countries_df = pd.DataFrame.from_dict(production_countries_dict, orient='index')

production_countries_df.columns = ["number_of_movies"]

production_countries_df = production_countries_df.sort_values(by="number_of_movies", ascending=False)

production_countries_df.head(10).plot.bar()

plt.title("Number of films per production country")
for c in production_countries_df.index.values:

    train_df['isProductionCountry_' + c] = train_df['production_countries_processed'].apply(lambda x: 1 if c in x else 0)

    test_df['isProductionCountry_' + c] = test_df['production_countries_processed'].apply(lambda x: 1 if c in x else 0)
f, ax = plt.subplots(3, figsize=(12,7))

sns.boxplot(x=train_df.popularity, ax = ax[0])

ax[0].set_title("Popularity Boxplot")

sns.distplot(a=train_df.popularity, kde = False, ax = ax[1])

ax[1].set_title("Popularity Histogram")

sns.distplot(a=np.log1p(train_df.popularity), kde = False, ax = ax[2])

ax[2].set_title("Log1p transformed Popularity Histogram")

f.tight_layout()



train_df["log_popularity"] = np.log1p(train_df.popularity)

test_df["log_popularity"] = np.log1p(test_df.popularity)
train_df["runtime"] = train_df["runtime"].fillna(train_df["runtime"].mode()[0])

test_df["runtime"] = test_df["runtime"].fillna(test_df["runtime"].mode()[0])



f, ax = plt.subplots(4, figsize=(12,7))



train_df.runtime = train_df.runtime.fillna(train_df.runtime.mode())



sns.boxplot(x=train_df.runtime, ax = ax[0])

ax[0].set_title("Runtime Boxplot")

sns.distplot(a=train_df.runtime, kde = False, ax = ax[1])

ax[1].set_title("Runtime Histogram")

sns.distplot(a=train_df.runtime/360, kde = False, ax = ax[2])

ax[2].set_title("Runtime in Hours Histogram")

sns.distplot(a=np.log1p(train_df.runtime), kde = False, ax = ax[3])

ax[3].set_title("Log1p transformed Runtime Histogram")

f.tight_layout()



train_df["runtime_in_hours"] = train_df.runtime/360

test_df["runtime_in_hours"] = test_df.runtime/360



train_df["log_runtime"] = np.log1p(train_df.runtime)

test_df["log_runtime"] = np.log1p(test_df.runtime)
from datetime import datetime



# fill possible NA values with the statistical mode

train_df["release_date"] = train_df["release_date"].fillna(train_df["release_date"].mode()[0])

test_df["release_date"] = test_df["release_date"].fillna(test_df["release_date"].mode()[0])





train_df['temp'] = train_df.release_date.apply(lambda x: datetime.strptime(x, '%m/%d/%y'))



train_df["month"] = train_df.temp.apply(lambda x: x.month)

train_df["season"] = train_df["month"]%4

train_df["year"] = train_df.temp.apply(lambda x: x.year)

train_df["day_of_week"] = train_df.temp.apply(lambda x: x.weekday()+1)

train_df["week_of_year"] = train_df.temp.apply(lambda x: x.isocalendar()[1])



train_df = train_df.drop(['temp'], axis=1)





test_df['temp'] = test_df.release_date.apply(lambda x: datetime.strptime(x, '%m/%d/%y'))



test_df["month"] = test_df.temp.apply(lambda x: x.month)

test_df["season"] = test_df["month"]%4

test_df["year"] = test_df.temp.apply(lambda x: x.year)

test_df["day_of_week"] = test_df.temp.apply(lambda x: x.weekday()+1)

test_df["week_of_year"] = test_df.temp.apply(lambda x: x.isocalendar()[1])



test_df = test_df.drop(['temp'], axis=1)







train_df["day_of_week"] = train_df["day_of_week"].fillna(train_df["day_of_week"].mode()[0])

test_df["day_of_week"] = test_df["day_of_week"].fillna(test_df["day_of_week"].mode()[0])



train_df["year"] = train_df["year"].fillna(train_df["year"].mode()[0])

test_df["year"] = test_df["year"].fillna(test_df["year"].mode()[0])



train_df["month"] = train_df["month"].fillna(train_df["month"].mode()[0])

test_df["month"] = test_df["month"].fillna(test_df["month"].mode()[0])



train_df["week_of_year"] = train_df["week_of_year"].fillna(train_df["week_of_year"].mode()[0])

test_df["week_of_year"] = test_df["week_of_year"].fillna(test_df["week_of_year"].mode()[0])



train_df["season"] = train_df["season"].fillna(train_df["season"].mode()[0])

test_df["season"] = test_df["season"].fillna(test_df["season"].mode()[0])



train_df[["release_date", "month", "year", "day_of_week", "week_of_year", "season"]].head()
sns.set(rc={'figure.figsize':(12,8)})

train_df.month.value_counts().plot.bar()

plt.title('Number of films per month')
sns.set(rc={'figure.figsize':(16,8)})

train_df.week_of_year.value_counts().plot.bar()

plt.title('Number of films per week_of_year')
sns.set(rc={'figure.figsize':(16,8)})

train_df.season.value_counts().plot.bar()

plt.title('Number of films per season')
sns.set(rc={'figure.figsize':(12,8)})

train_df.day_of_week.value_counts().plot.bar()

plt.title('Number of films per day_of_week')
sns.set(rc={'figure.figsize':(20,8)})

train_df.year.value_counts().plot.bar()

plt.title('Number of films per year')
import re



actors_dict = {}

size_of_actors = len(train_df) - train_df.cast.isna().sum()



for element in train_df[["revenue", "cast"]].values:

    if type(element[1]) == type(str()):

        

        result = re.findall('name\': \'\w+\s*\w*', element[1])

        result = [x.replace("name\': \'", "") for x in result]



        for actor in result:

            if actor not in actors_dict:

                actors_dict[actor] = element[0]

            else:

                actors_dict[actor] += element[0]

                

for actor in actors_dict:

    actors_dict[actor] = actors_dict[actor]/size_of_actors

    





actors_df = pd.DataFrame.from_dict(actors_dict, orient='index', columns=["mean_movies_revenue"])

actors_df.sort_values(by="mean_movies_revenue", ascending=False).head(20).plot.bar()
def find_top_actor_from_cast(top_actor, element):

    

    result = []

    if type(element) == type(str()):



        result = re.findall('name\': \'\w+\s*\w*', element)

        result = [x.replace("name\': \'", "") for x in result]

        

    if top_actor in result:

        return 1

    else:

        return 0



for top_actor in actors_df.sort_values(by="mean_movies_revenue", ascending=False).head(10).index.values:

    train_df["has_top_actor_"+ top_actor] = train_df.cast.apply(lambda element: find_top_actor_from_cast(top_actor, element))

    test_df["has_top_actor_"+ top_actor] = test_df.cast.apply(lambda element: find_top_actor_from_cast(top_actor, element))
import re



keywords_dict = {}

size_of_keywords = len(train_df) - train_df.Keywords.isna().sum()



for element in train_df[["revenue", "Keywords"]].values:

    if type(element[1]) == type(str()):

        

        result = re.findall('name\': \'\w+\s*\w*', element[1])

        result = [x.replace("name\': \'", "") for x in result]



        for key in result:

            if key not in keywords_dict:

                keywords_dict[key] = element[0]

            else:

                keywords_dict[key] += element[0]

                

for key in keywords_dict:

    keywords_dict[key] = keywords_dict[key]/size_of_keywords

    

keywords_df = pd.DataFrame.from_dict(keywords_dict, orient='index', columns=["mean_movies_revenue"])

keywords_df.sort_values(by="mean_movies_revenue", ascending=False).head(10).plot.bar()
def find_top_keywords_from_cast(top_keyword, element):

    

    result = []

    if type(element) == type(str()):



        result = re.findall('name\': \'\w+\s*\w*', element)

        result = [x.replace("name\': \'", "") for x in result]

        

    if top_keyword in result:

        return 1

    else:

        return 0



for top_keyword in keywords_df.sort_values(by="mean_movies_revenue", ascending=False).head(10).index.values:

    train_df["has_top_keyword_"+ top_keyword] = train_df.Keywords.apply(lambda element: find_top_keywords_from_cast(top_keyword, element))

    test_df["has_top_keyword_"+ top_keyword] = test_df.Keywords.apply(lambda element: find_top_keywords_from_cast(top_keyword, element))
train_df["num_of_cast"] = train_df["cast"].str.count("name")

test_df["num_of_cast"] = test_df["cast"].str.count("name")



f, ax = plt.subplots(3, figsize=(12,7))



train_df.num_of_cast = train_df.num_of_cast.fillna(0)

test_df.num_of_cast = test_df.num_of_cast.fillna(0)



sns.boxplot(x=train_df.num_of_cast, ax = ax[0])

ax[0].set_title("num_of_cast Boxplot")

sns.distplot(a=train_df.num_of_cast, kde = False, ax = ax[1])

ax[1].set_title("num_of_cast Histogram")

sns.distplot(a=np.log1p(train_df.num_of_cast), kde = False, ax = ax[2])

ax[2].set_title("Log1p transformed num_of_cast Histogram")

f.tight_layout()





train_df["log_num_of_cast"] = np.log1p(train_df.num_of_cast)

test_df["log_num_of_cast"] = np.log1p(test_df.num_of_cast)
train_df["num_of_male_cast"] = train_df["cast"].str.count("'gender': 2")

test_df["num_of_male_cast"] = test_df["cast"].str.count("'gender': 2")



f, ax = plt.subplots(3, figsize=(12,7))



train_df.num_of_male_cast = train_df.num_of_male_cast.fillna(0)

test_df.num_of_male_cast = test_df.num_of_male_cast.fillna(0)



sns.boxplot(x=train_df.num_of_male_cast, ax = ax[0])

ax[0].set_title("num_of_male_cast Boxplot")

sns.distplot(a=train_df.num_of_male_cast, kde = False, ax = ax[1])

ax[1].set_title("num_of_male_cast Histogram")

sns.distplot(a=np.log1p(train_df.num_of_male_cast), kde = False, ax = ax[2])

ax[2].set_title("Log1p transformed num_of_male_cast Histogram")

f.tight_layout()





train_df["log_num_of_male_cast"] = np.log1p(train_df.num_of_male_cast)

test_df["log_num_of_male_cast"] = np.log1p(test_df.num_of_male_cast)
train_df["num_of_female_cast"] = train_df["cast"].str.count("'gender': 1")

test_df["num_of_female_cast"] = test_df["cast"].str.count("'gender': 1")



f, ax = plt.subplots(3, figsize=(12,7))



train_df.num_of_female_cast = train_df.num_of_female_cast.fillna(0)

test_df.num_of_female_cast = test_df.num_of_female_cast.fillna(0)



sns.boxplot(x=train_df.num_of_female_cast, ax = ax[0])

ax[0].set_title("num_of_female_cast Boxplot")

sns.distplot(a=train_df.num_of_female_cast, kde = False, ax = ax[1])

ax[1].set_title("num_of_female_cast Histogram")

sns.distplot(a=np.log1p(train_df.num_of_female_cast), kde = False, ax = ax[2])

ax[2].set_title("Log1p transformed num_of_female_cast Histogram")

f.tight_layout()





train_df["log_num_of_female_cast"] = np.log1p(train_df.num_of_female_cast)

test_df["log_num_of_female_cast"] = np.log1p(test_df.num_of_female_cast)
train_df["num_of_crew"] = train_df["crew"].str.count("'job")

test_df["num_of_crew"] = test_df["crew"].str.count("'job")



f, ax = plt.subplots(3, figsize=(12,7))



train_df.num_of_crew = train_df.num_of_crew.fillna(0)

test_df.num_of_crew = test_df.num_of_crew.fillna(0)



sns.boxplot(x=train_df.num_of_crew, ax = ax[0])

ax[0].set_title("num_of_crew Boxplot")

sns.distplot(a=train_df.num_of_crew, kde = False, ax = ax[1])

ax[1].set_title("num_of_crew Histogram")

sns.distplot(a=np.log1p(train_df.num_of_crew), kde = False, ax = ax[2])

ax[2].set_title("Log1p transformed num_of_crew Histogram")

f.tight_layout()





train_df["log_num_of_crew"] = np.log1p(train_df.num_of_crew)

test_df["log_num_of_crew"] = np.log1p(test_df.num_of_crew)
train_df["num_of_male_crew"] = train_df["crew"].str.count("'gender': 2")

test_df["num_of_male_crew"] = test_df["crew"].str.count("'gender': 2")



f, ax = plt.subplots(3, figsize=(12,7))



train_df.num_of_male_crew = train_df.num_of_male_crew.fillna(0)

test_df.num_of_male_crew = test_df.num_of_crew.fillna(0)



sns.boxplot(x=train_df.num_of_male_crew, ax = ax[0])

ax[0].set_title("num_of_male_crew Boxplot")

sns.distplot(a=train_df.num_of_male_crew, kde = False, ax = ax[1])

ax[1].set_title("num_of_male_crew Histogram")

sns.distplot(a=np.log1p(train_df.num_of_male_crew), kde = False, ax = ax[2])

ax[2].set_title("Log1p transformed num_of_male_crew Histogram")

f.tight_layout()





train_df["log_num_of_male_crew"] = np.log1p(train_df.num_of_male_crew)

test_df["log_num_of_male_crew"] = np.log1p(test_df.num_of_male_crew)
train_df["num_of_female_crew"] = train_df["crew"].str.count("'gender': 1")

test_df["num_of_female_crew"] = test_df["crew"].str.count("'gender': 1")



f, ax = plt.subplots(3, figsize=(12,7))



train_df.num_of_female_crew = train_df.num_of_female_crew.fillna(0)

test_df.num_of_female_crew = test_df.num_of_female_crew.fillna(0)



sns.boxplot(x=train_df.num_of_female_crew, ax = ax[0])

ax[0].set_title("num_of_female_crew Boxplot")

sns.distplot(a=train_df.num_of_female_crew, kde = False, ax = ax[1])

ax[1].set_title("num_of_female_crew Histogram")

sns.distplot(a=np.log1p(train_df.num_of_female_crew), kde = False, ax = ax[2])

ax[2].set_title("Log1p transformed num_of_female_crew Histogram")

f.tight_layout()





train_df["log_num_of_female_crew"] = np.log1p(train_df.num_of_female_crew)

test_df["log_num_of_female_crew"] = np.log1p(test_df.num_of_female_crew)
import re



directors_dict = {}

size_of_crew = len(train_df) - train_df.crew.isna().sum()



for element in train_df[["revenue", "crew"]].values:

    if type(element[1]) == type(str()):

        

        result = re.findall('Director\', \'name\': \'\w+\s*\w*', element[1])

        result = [x.replace("Director\', \'name\': \'", "") for x in result]



        for key in result:

            if key not in directors_dict:

                directors_dict[key] = element[0]

            else:

                directors_dict[key] += element[0]

                

for key in directors_dict:

    directors_dict[key] = directors_dict[key]/size_of_crew

    

directors_df = pd.DataFrame.from_dict(directors_dict, orient='index', columns=["mean_movies_revenue"])

directors_df.sort_values(by="mean_movies_revenue", ascending=False).head(10).plot.bar()
def find_top_directors_from_crew(top_director, element):

    

    result = []

    if type(element) == type(str()):



        result = re.findall('Director\', \'name\': \'\w+\s*\w*', element)

        result = [x.replace("Director\', \'name\': \'", "") for x in result]

        

    if top_director in result:

        return 1

    else:

        return 0



for top_director in directors_df.sort_values(by="mean_movies_revenue", ascending=False).head(10).index.values:

    train_df["has_top_director_"+ top_director] = train_df.crew.apply(lambda element: find_top_directors_from_crew(top_director, element))

    test_df["has_top_director_"+ top_director] = test_df.crew.apply(lambda element: find_top_directors_from_crew(top_director, element))
import re



producers_dict = {}

size_of_crew = len(train_df) - train_df.crew.isna().sum()



for element in train_df[["revenue", "crew"]].values:

    if type(element[1]) == type(str()):

        

        result = re.findall('Producer\', \'name\': \'\w+\s*\w*', element[1])

        result = [x.replace("Producer\', \'name\': \'", "") for x in result]



        for key in result:

            if key not in producers_dict:

                producers_dict[key] = element[0]

            else:

                producers_dict[key] += element[0]

                

for key in producers_dict:

    producers_dict[key] = producers_dict[key]/size_of_crew

    

producers_df = pd.DataFrame.from_dict(producers_dict, orient='index', columns=["mean_movies_revenue"])

producers_df.sort_values(by="mean_movies_revenue", ascending=False).head(10).plot.bar()
def find_top_producers_from_crew(top_producer, element):

    

    result = []

    if type(element) == type(str()):



        result = re.findall('Director\', \'name\': \'\w+\s*\w*', element)

        result = [x.replace("Director\', \'name\': \'", "") for x in result]

        

    if top_producer in result:

        return 1

    else:

        return 0



for top_producer in producers_df.sort_values(by="mean_movies_revenue", ascending=False).head(10).index.values:

    train_df["has_top_producer_"+ top_producer] = train_df.crew.apply(lambda element: find_top_producers_from_crew(top_producer, element))

    test_df["has_top_producer_"+ top_producer] = test_df.crew.apply(lambda element: find_top_producers_from_crew(top_producer, element))
train_df["num_of_directors"] = train_df["crew"].str.count("Directing")

test_df["num_of_directors"] = test_df["crew"].str.count("Directing")



f, ax = plt.subplots(3, figsize=(12,7))



train_df.num_of_directors = train_df.num_of_directors.fillna(0)

test_df.num_of_directors = test_df.num_of_directors.fillna(0)



sns.boxplot(x=train_df.num_of_directors, ax = ax[0])

ax[0].set_title("num_of_directors Boxplot")

sns.distplot(a=train_df.num_of_directors, kde = False, ax = ax[1])

ax[1].set_title("num_of_directors Histogram")

sns.distplot(a=np.log1p(train_df.num_of_directors), kde = False, ax = ax[2])

ax[2].set_title("Log1p transformed num_of_directors Histogram")

f.tight_layout()





train_df["log_num_of_directors"] = np.log1p(train_df.num_of_directors)

test_df["log_num_of_directors"] = np.log1p(test_df.num_of_directors)
train_df["num_of_producers"] = train_df["crew"].str.count("Production")

test_df["num_of_producers"] = test_df["crew"].str.count("Production")



f, ax = plt.subplots(3, figsize=(12,7))



train_df.num_of_producers = train_df.num_of_producers.fillna(0)

test_df.num_of_producers = test_df.num_of_producers.fillna(0)



sns.boxplot(x=train_df.num_of_producers, ax = ax[0])

ax[0].set_title("num_of_producers Boxplot")

sns.distplot(a=train_df.num_of_producers, kde = False, ax = ax[1])

ax[1].set_title("num_of_producers Histogram")

sns.distplot(a=np.log1p(train_df.num_of_producers), kde = False, ax = ax[2])

ax[2].set_title("Log1p transformed num_of_producers Histogram")

f.tight_layout()





train_df["log_num_of_producers"] = np.log1p(train_df.num_of_producers)

test_df["log_num_of_producers"] = np.log1p(test_df.num_of_producers)
train_df["num_of_writers"] = train_df["crew"].str.count("Writing")

test_df["num_of_writers"] = test_df["crew"].str.count("Writing")



f, ax = plt.subplots(3, figsize=(12,7))



train_df.num_of_writers = train_df.num_of_writers.fillna(0)

test_df.num_of_writers = test_df.num_of_writers.fillna(0)



sns.boxplot(x=train_df.num_of_writers, ax = ax[0])

ax[0].set_title("num_of_writers Boxplot")

sns.distplot(a=train_df.num_of_writers, kde = False, ax = ax[1])

ax[1].set_title("num_of_writers Histogram")

sns.distplot(a=np.log1p(train_df.num_of_writers), kde = False, ax = ax[2])

ax[2].set_title("Log1p transformed num_of_writers Histogram")

f.tight_layout()





train_df["log_num_of_writers"] = np.log1p(train_df.num_of_writers)

test_df["log_num_of_writers"] = np.log1p(test_df.num_of_writers)
train_df["num_of_editors"] = train_df["crew"].str.count("Editing")

test_df["num_of_editors"] = test_df["crew"].str.count("Editing")



f, ax = plt.subplots(3, figsize=(12,7))



train_df.num_of_editors = train_df.num_of_editors.fillna(0)

test_df.num_of_editors = test_df.num_of_editors.fillna(0)



sns.boxplot(x=train_df.num_of_editors, ax = ax[0])

ax[0].set_title("num_of_editors Boxplot")

sns.distplot(a=train_df.num_of_editors, kde = False, ax = ax[1])

ax[1].set_title("num_of_editors Histogram")

sns.distplot(a=np.log1p(train_df.num_of_editors), kde = False, ax = ax[2])

ax[2].set_title("Log1p transformed num_of_editors Histogram")

f.tight_layout()





train_df["log_num_of_editors"] = np.log1p(train_df.num_of_editors)

test_df["log_num_of_editors"] = np.log1p(test_df.num_of_editors)
train_df["num_of_art_crew"] = train_df["crew"].str.count("Art")

test_df["num_of_art_crew"] = test_df["crew"].str.count("Art")



f, ax = plt.subplots(3, figsize=(12,7))



train_df.num_of_art_crew = train_df.num_of_art_crew.fillna(0)

test_df.num_of_art_crew = test_df.num_of_art_crew.fillna(0)



sns.boxplot(x=train_df.num_of_art_crew, ax = ax[0])

ax[0].set_title("num_of_art_crew Boxplot")

sns.distplot(a=train_df.num_of_art_crew, kde = False, ax = ax[1])

ax[1].set_title("num_of_art_crew Histogram")

sns.distplot(a=np.log1p(train_df.num_of_art_crew), kde = False, ax = ax[2])

ax[2].set_title("Log1p transformed num_of_art_crew Histogram")

f.tight_layout()





train_df["log_num_of_art_crew"] = np.log1p(train_df.num_of_art_crew)

test_df["log_num_of_art_crew"] = np.log1p(test_df.num_of_art_crew)
train_df["num_of_sound_crew"] = train_df["crew"].str.count("Sound")

test_df["num_of_sound_crew"] = test_df["crew"].str.count("Sound")



f, ax = plt.subplots(3, figsize=(12,7))



train_df.num_of_sound_crew = train_df.num_of_sound_crew.fillna(0)

test_df.num_of_sound_crew = test_df.num_of_sound_crew.fillna(0)



sns.boxplot(x=train_df.num_of_sound_crew, ax = ax[0])

ax[0].set_title("num_of_sound_crew Boxplot")

sns.distplot(a=train_df.num_of_sound_crew, kde = False, ax = ax[1])

ax[1].set_title("num_of_sound_crew Histogram")

sns.distplot(a=np.log1p(train_df.num_of_sound_crew), kde = False, ax = ax[2])

ax[2].set_title("Log1p transformed num_of_sound_crew Histogram")

f.tight_layout()





train_df["log_num_of_sound_crew"] = np.log1p(train_df.num_of_sound_crew)

test_df["log_num_of_sound_crew"] = np.log1p(test_df.num_of_sound_crew)
train_df["num_of_costume_crew"] = train_df["crew"].str.count("Costume & Make-Up")

test_df["num_of_costume_crew"] = test_df["crew"].str.count("Costume & Make-Up")



f, ax = plt.subplots(3, figsize=(12,7))



train_df.num_of_costume_crew = train_df.num_of_costume_crew.fillna(0)

test_df.num_of_costume_crew = test_df.num_of_costume_crew.fillna(0)



sns.boxplot(x=train_df.num_of_costume_crew, ax = ax[0])

ax[0].set_title("num_of_costume_crew Boxplot")

sns.distplot(a=train_df.num_of_costume_crew, kde = False, ax = ax[1])

ax[1].set_title("num_of_costume_crew Histogram")

sns.distplot(a=np.log1p(train_df.num_of_costume_crew), kde = False, ax = ax[2])

ax[2].set_title("Log1p transformed num_of_costume_crew Histogram")

f.tight_layout()





train_df["log_num_of_costume_crew"] = np.log1p(train_df.num_of_costume_crew)

test_df["log_num_of_costume_crew"] = np.log1p(test_df.num_of_costume_crew)
train_df["num_of_camera_crew"] = train_df["crew"].str.count("\'department\': \'Camera\'")

test_df["num_of_camera_crew"] = test_df["crew"].str.count("\'department\': \'Camera\'")



f, ax = plt.subplots(3, figsize=(12,7))



train_df.num_of_camera_crew = train_df.num_of_camera_crew.fillna(0)

test_df.num_of_camera_crew = test_df.num_of_camera_crew.fillna(0)



sns.boxplot(x=train_df.num_of_camera_crew, ax = ax[0])

ax[0].set_title("num_of_camera_crew Boxplot")

sns.distplot(a=train_df.num_of_camera_crew, kde = False, ax = ax[1])

ax[1].set_title("num_of_camera_crew Histogram")

sns.distplot(a=np.log1p(train_df.num_of_camera_crew), kde = False, ax = ax[2])

ax[2].set_title("Log1p transformed num_of_camera_crew Histogram")

f.tight_layout()





train_df["log_num_of_camera_crew"] = np.log1p(train_df.num_of_camera_crew)

test_df["log_num_of_camera_crew"] = np.log1p(test_df.num_of_camera_crew)
train_df["num_of_visual_effects_crew"] = train_df["crew"].str.count("\'department\': \'Visual Effects\'")

test_df["num_of_visual_effects_crew"] = test_df["crew"].str.count("\'department\': \'Visual Effects\'")



f, ax = plt.subplots(3, figsize=(12,7))



train_df.num_of_visual_effects_crew = train_df.num_of_visual_effects_crew.fillna(0)

test_df.num_of_visual_effects_crew = test_df.num_of_visual_effects_crew.fillna(0)



sns.boxplot(x=train_df.num_of_visual_effects_crew, ax = ax[0])

ax[0].set_title("num_of_visual_effects_crew Boxplot")

sns.distplot(a=train_df.num_of_visual_effects_crew, kde = False, ax = ax[1])

ax[1].set_title("num_of_visual_effects_crew Histogram")

sns.distplot(a=np.log1p(train_df.num_of_visual_effects_crew), kde = False, ax = ax[2])

ax[2].set_title("Log1p transformed num_of_visual_effects_crew Histogram")

f.tight_layout()





train_df["log_num_of_visual_effects_crew"] = np.log1p(train_df.num_of_visual_effects_crew)

test_df["log_num_of_visual_effects_crew"] = np.log1p(test_df.num_of_visual_effects_crew)
train_df["num_of_lighting_crew"] = train_df["crew"].str.count("\'department\': \'Lighting\'")

test_df["num_of_lighting_crew"] = test_df["crew"].str.count("\'department\': \'Lighting\'")



f, ax = plt.subplots(3, figsize=(12,7))



train_df.num_of_lighting_crew = train_df.num_of_lighting_crew.fillna(0)

test_df.num_of_lighting_crew = test_df.num_of_lighting_crew.fillna(0)



sns.boxplot(x=train_df.num_of_lighting_crew, ax = ax[0])

ax[0].set_title("num_of_lighting_crew Boxplot")

sns.distplot(a=train_df.num_of_lighting_crew, kde = False, ax = ax[1])

ax[1].set_title("num_of_lighting_crew Histogram")

sns.distplot(a=np.log1p(train_df.num_of_lighting_crew), kde = False, ax = ax[2])

ax[2].set_title("Log1p transformed num_of_lighting_crew Histogram")

f.tight_layout()





train_df["log_num_of_lighting_crew"] = np.log1p(train_df.num_of_lighting_crew)

test_df["log_num_of_lighting_crew"] = np.log1p(test_df.num_of_lighting_crew)
train_df["num_of_other_crew"] = train_df["crew"].str.count("\'department\': \'Crew\'")

test_df["num_of_other_crew"] = test_df["crew"].str.count("\'department\': \'Crew\'")



f, ax = plt.subplots(3, figsize=(12,7))



train_df.num_of_other_crew = train_df.num_of_other_crew.fillna(0)

test_df.num_of_other_crew = test_df.num_of_other_crew.fillna(0)



sns.boxplot(x=train_df.num_of_other_crew, ax = ax[0])

ax[0].set_title("num_of_other_crew Boxplot")

sns.distplot(a=train_df.num_of_other_crew, kde = False, ax = ax[1])

ax[1].set_title("num_of_other_crew Histogram")

sns.distplot(a=np.log1p(train_df.num_of_other_crew), kde = False, ax = ax[2])

ax[2].set_title("Log1p transformed num_of_other_crew Histogram")

f.tight_layout()





train_df["log_num_of_other_crew"] = np.log1p(train_df.num_of_other_crew)

test_df["log_num_of_other_crew"] = np.log1p(test_df.num_of_other_crew)
train_df["num_of_production_countries"] = train_df.production_countries_processed.apply(len)

test_df["num_of_production_countries"] = test_df.production_countries_processed.apply(len)



f, ax = plt.subplots(3, figsize=(12,7))



train_df.num_of_production_countries = train_df.num_of_production_countries.fillna(0)

test_df.num_of_production_countries = test_df.num_of_production_countries.fillna(0)



sns.boxplot(x=train_df.num_of_production_countries, ax = ax[0])

ax[0].set_title("num_of_production_countries Boxplot")

sns.distplot(a=train_df.num_of_production_countries, kde = False, ax = ax[1])

ax[1].set_title("num_of_production_countries Histogram")

sns.distplot(a=np.log1p(train_df.num_of_production_countries), kde = False, ax = ax[2])

ax[2].set_title("Log1p transformed num_of_production_countries Histogram")

f.tight_layout()





train_df["log_num_of_production_countries"] = np.log1p(train_df.num_of_production_countries)

test_df["log_num_of_production_countries"] = np.log1p(test_df.num_of_production_countries)
train_df["num_of_genres"] = train_df.genres_processed.apply(len)

test_df["num_of_genres"] = test_df.genres_processed.apply(len)



f, ax = plt.subplots(3, figsize=(12,7))



train_df.num_of_genres = train_df.num_of_genres.fillna(0)

test_df.num_of_genres = test_df.num_of_genres.fillna(0)



sns.boxplot(x=train_df.num_of_genres, ax = ax[0])

ax[0].set_title("num_of_genres Boxplot")

sns.distplot(a=train_df.num_of_genres, kde = False, ax = ax[1])

ax[1].set_title("num_of_genres Histogram")

sns.distplot(a=np.log1p(train_df.num_of_genres), kde = False, ax = ax[2])

ax[2].set_title("Log1p transformed num_of_genres Histogram")

f.tight_layout()





train_df["log_num_of_genres"] = np.log1p(train_df.num_of_genres)

test_df["log_num_of_genres"] = np.log1p(test_df.num_of_genres)
sns.set(rc={'figure.figsize':(20,27)})



# Compute the correlation matrix

corr = train_df[["revenue", "budget", "popularity", "runtime", "num_of_cast", "num_of_male_cast",

                 "num_of_female_cast",

                 "num_genres", "num_of_production_countries", "day_of_week", "month", "year", "week_of_year", "season",

                 "title_len", "overview_len", "tagline_len",

                 "num_of_directors", "num_of_producers", "num_of_editors", "num_of_art_crew", "num_of_sound_crew",

                 "num_of_costume_crew", "num_of_camera_crew", "num_of_visual_effects_crew", "num_of_lighting_crew",

                 "num_of_other_crew"]].corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



sns.heatmap(corr, mask=mask, 

            annot=True, 

            fmt=".2f", 

            cmap='coolwarm')



plt.title("Correlation between numerical features")
sns.set(rc={'figure.figsize':(18,20)})



# Compute the correlation matrix

corr = train_df[["log_revenue", "log_budget", "log_popularity", "log_runtime",

                 "log_num_of_cast", "log_num_of_male_cast",

                 "log_num_of_female_cast", "num_genres", "num_of_production_countries",

                "day_of_week", "month", "year", "week_of_year", "season",

                "log_title_len", "log_overview_len", "log_tagline_len",

                "log_num_of_directors", "log_num_of_producers", "log_num_of_editors", "log_num_of_art_crew", "log_num_of_sound_crew",

                       "log_num_of_costume_crew", "log_num_of_camera_crew", "log_num_of_visual_effects_crew", "log_num_of_lighting_crew",

                        "log_num_of_other_crew"]].corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



sns.heatmap(corr, mask=mask, 

            annot=True, 

            fmt=".2f", 

            cmap='coolwarm')



plt.title("Correlation between log1p transformed numerical features")
train_df['has_collection'] = [0 if pd.isnull(x) else 1 for x in train_df['belongs_to_collection']]

test_df['has_collection'] = [0 if pd.isnull(x) else 1 for x in test_df['belongs_to_collection']]

print(train_df['has_collection'].value_counts())



sns.set(rc={'figure.figsize':(12, 8)})

sns.boxplot(x='has_collection', y='revenue', data=train_df)

plt.title('Revenue for film with and without being in a collection')
train_df['has_homepage'] = [0 if pd.isnull(x) else 1 for x in train_df['homepage']]

test_df['has_homepage'] = [0 if pd.isnull(x) else 1 for x in test_df['homepage']]

print(train_df['has_homepage'].value_counts())



sns.set(rc={'figure.figsize':(12, 8)})

sns.boxplot(x='has_homepage', y='revenue', data=train_df)

plt.title('Revenue for film with and without homepage')
train_df['has_tag'] = [0 if len(x) == 0 else 1 for x in train_df['tagline']]

test_df['has_tag'] = [0 if len(x) == 0 else 1 for x in test_df['tagline']]

print(train_df['has_tag'].value_counts())



sns.set(rc={'figure.figsize':(12, 8)})

sns.boxplot(x='has_tag', y='revenue', data=train_df)

plt.title('Revenue for film with and without tagline')
sns.set(rc={'figure.figsize':(12, 8)})

sns.boxplot(x='num_of_genres', y='revenue', data=train_df)

plt.title('Revenues for films with multiple genres')
for i, g in enumerate(genres_df.index.values):

    genres_df.loc[g, "median_salary"] = train_df[train_df['isGenre_' + g]==1].revenue.median()



genres_df.sort_values(by=["number_of_movies", "median_salary"], ascending=False).head(10)
genres_df.sort_values(by=["median_salary"], ascending=False).median_salary.plot.bar()

plt.title("Sorted movie genres by median revenue")
for i, p in enumerate(genres_df.sort_values(by=["number_of_movies", "median_salary"], ascending=False).head(10).index.values):

    train_df['isTopGenre_' + p] = train_df['genres_processed'].apply(lambda x: 1 if p in x else 0)

    train_df['isTopGenre_Other'] = train_df['genres_processed'].apply(lambda x: 1 if p not in x else 0)

    test_df['isTopGenre_' + p] = test_df['genres_processed'].apply(lambda x: 1 if p in x else 0)

    test_df['isTopGenre_Other'] = test_df['genres_processed'].apply(lambda x: 1 if p not in x else 0)
f, ax = plt.subplots(len(genres_df.index.values), 1, figsize=(15,160))



for i, g in enumerate(genres_df.index.values):

    sns.boxplot(x=train_df['isGenre_' + g], y='revenue', ax=ax[i], data=train_df)

    ax[i].set_title('isGenre_' + g +" and revenue boxplot")

f.tight_layout()
sns.set(rc={'figure.figsize':(12, 8)})

sns.boxplot(x='original_language', y='revenue', data=train_df)

plt.title('Revenue for a movie and its and original_language')
sns.set(rc={'figure.figsize':(12, 8)})

sns.boxplot(x='is_english_language', y='revenue', data=train_df)

plt.title('Revenue for a movie in contrast with english and non-english language')
sns.set(rc={'figure.figsize':(12, 8)})

sns.boxplot(x='num_of_production_countries', y='revenue', data=train_df)

plt.title('number of production countries for a movie and revenue')
f, ax = plt.subplots(len(production_countries_df.index.values), 1, figsize=(15,350))



for i, c in enumerate(production_countries_df.index.values):

    sns.boxplot(x=train_df['isProductionCountry_' + c], y='revenue', ax=ax[i], data=train_df)

    ax[i].set_title('isProductionCountry_' + c +" and revenue boxplot")

f.tight_layout()
for i, c in enumerate(production_countries_df.index.values):

    production_countries_df.loc[c, "median_salary"] = train_df[train_df['isProductionCountry_' + c]==1].revenue.median()



production_countries_df.sort_values(by=["number_of_movies", "median_salary"], ascending=False).head(10)
for i, p in enumerate(production_countries_df.sort_values(by=["number_of_movies", "median_salary"], ascending=False).head(10).index.values):

    train_df['isTopProductionCountry_' + p] = train_df['production_countries_processed'].apply(lambda x: 1 if p in x else 0)

    test_df['isTopProductionCountry_' + p] = test_df['production_countries_processed'].apply(lambda x: 1 if p in x else 0)
sns.set(rc={'figure.figsize':(12, 8)})

sns.boxplot(x='day_of_week', y='revenue', data=train_df)

plt.title('day_of_week when the movie release and revenue')
sns.set(rc={'figure.figsize':(12, 8)})

sns.boxplot(x='week_of_year', y='revenue', data=train_df)

plt.title('day_of_week when the movie release and revenue')
sns.set(rc={'figure.figsize':(12, 8)})

sns.boxplot(x='month', y='revenue', data=train_df)

plt.title('month when the movie release and revenue')
sns.set(rc={'figure.figsize':(12, 8)})

sns.boxplot(x='season', y='revenue', data=train_df)

plt.title('season when the movie release and revenue')
sns.set(rc={'figure.figsize':(20, 8)})

g = sns.boxplot(x='year', y='revenue', data=train_df)

plt.xticks(rotation=90)

plt.title('Year when the movie release and revenue')
sns.set(rc={'figure.figsize':(12, 8)})

sns.boxplot(x='num_of_production_companies', y='revenue', data=train_df)

plt.title('number of production companies for a movie and revenue')
f, ax = plt.subplots(len(production_companies_df.head(5).index.values), 1, figsize=(15,20))



for i, p in enumerate(production_companies_df.head(5).index.values):

    sns.boxplot(x=train_df['isProductionCompany_' + p], y='revenue', ax=ax[i], data=train_df)

    ax[i].set_title('isProductionCompany_' + p +" and revenue boxplot")

f.tight_layout()
for i, p in enumerate(production_companies_df.index.values):

    production_companies_df.loc[p, "median_salary"] = train_df[train_df['isProductionCompany_' + p]==1].revenue.median()



production_companies_df.sort_values(by=["number_of_movies", "median_salary"], ascending=False).head(5)
for i, p in enumerate(production_companies_df.sort_values(by=["number_of_movies", "median_salary"], ascending=False).head(10).index.values):

    train_df['isTopProductionCompany_' + p] = train_df['production_companies_processed'].apply(lambda x: 1 if p in x else 0)

    #train_df['isTopProductionCompany_Other'] = train_df['production_companies_processed'].apply(lambda x: 1 if p not in x else 0)

    test_df['isTopProductionCompany_' + p] = test_df['production_companies_processed'].apply(lambda x: 1 if p in x else 0)

    #test_df['isTopProductionCompany_Other'] = test_df['production_companies_processed'].apply(lambda x: 1 if p not in x else 0)
train_df.columns
columns_for_training = ["log_budget", "log_popularity", "log_runtime", "day_of_week", "year", "month", "week_of_year", "season",

                        "num_genres", "num_of_production_countries", "log_num_of_cast", "log_num_of_male_cast", "log_num_of_female_cast", "has_collection", 

                        "has_homepage", "has_tag", "is_english_language",

                       "log_num_of_crew", "log_num_of_male_crew", "log_num_of_female_crew",

                       "log_title_len", "log_overview_len", "log_tagline_len",

                       "log_num_of_directors", "log_num_of_producers", "log_num_of_editors", "log_num_of_art_crew", "log_num_of_sound_crew",

                       "log_num_of_costume_crew", "log_num_of_camera_crew", "log_num_of_visual_effects_crew", "log_num_of_lighting_crew",

                        "log_num_of_other_crew"]





# adding isTopGenre_ columns for features before ML modeling

columns_for_training.extend(train_df.select(lambda col: col.startswith('isTopGenre_'), axis=1).columns.values)



# adding isTopProductionCompany_ columns for features before ML modeling

columns_for_training.extend(train_df.select(lambda col: col.startswith('isTopProductionCompany_'), axis=1).columns.values)



# adding isTopProductionCountry_ columns for features before ML modeling

columns_for_training.extend(train_df.select(lambda col: col.startswith('isTopProductionCountry_'), axis=1).columns.values)



# adding has_top_actor_ columns for features before ML modeling

columns_for_training.extend(train_df.select(lambda col: col.startswith('has_top_actor_'), axis=1).columns.values)



# adding has_top_keyword_ columns for features before ML modeling

columns_for_training.extend(train_df.select(lambda col: col.startswith('has_top_keyword_'), axis=1).columns.values)



# adding has_top_director_ columns for features before ML modeling

columns_for_training.extend(train_df.select(lambda col: col.startswith('has_top_director_'), axis=1).columns.values)



# adding has_top_producer_ columns for features before ML modeling

columns_for_training.extend(train_df.select(lambda col: col.startswith('has_top_producer_'), axis=1).columns.values)



columns_for_training
train_df[columns_for_training].head(4)
train_df[columns_for_training].isna().sum().sum()
len(columns_for_training)
### a small snippet code for t-test significance between 2 groups, I may use it in the future:





#from scipy import stats



#columns_to_test = train_df.select(lambda col: col.startswith('isProductionCompany_'), axis=1).columns.values



#def check_catagorical_to_revenue_statistical_difference(train_df):

    

#    for col in columns_to_test:



#        a = train_df[train_df[col]==0].revenue

#        b = train_df[train_df[col]==1].revenue

#        t2, p2 = stats.ttest_ind(a,b)

#        if p2<0.05:

#            print(col , " is important for prediction with p-value:", p2)

        

#check_catagorical_to_revenue_statistical_difference(train_df)

    
y = train_df['log_revenue']

X = train_df[columns_for_training]

kfold_splits = 5
from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import mean_squared_error

import scikitplot as skplt

import time

import random



import xgboost as xgb



# create a 70/30 stratified split of the data 

xtrain, xvalid, ytrain, yvalid = train_test_split(X, y, random_state=42, test_size=0.3)





predictions_test_xgb = np.zeros(len(test_df))

num_fold = 0

num_of_splits = kfold_splits

oof_rmse = 0



folds = KFold(n_splits=num_of_splits, shuffle=False, random_state = 42)



for train_index, valid_index in folds.split(xtrain, ytrain):

    xtrain_stra, xvalid_stra = xtrain.iloc[train_index,:], xtrain.iloc[valid_index,:]

    ytrain_stra, yvalid_stra = ytrain.iloc[train_index], ytrain.iloc[valid_index]



    print()

    print("Fold:", num_fold)

    print()



    clf_stra_xgb = xgb.XGBRegressor(n_estimators=10000, seed=42, nthread=-1)



    clf_stra_xgb.fit(xtrain_stra, ytrain_stra, eval_set=[(xtrain_stra, ytrain_stra), (xvalid_stra, yvalid_stra)], 

                early_stopping_rounds=1000, eval_metric='rmse', verbose=100)



    predictions_valid = clf_stra_xgb.predict(xvalid)

    rmse_valid = np.sqrt(mean_squared_error(yvalid, predictions_valid))

    

    print("Fold",num_fold,"xvalid rmse:",rmse_valid)

    num_fold = num_fold + 1

    

    oof_rmse += rmse_valid



    predictions_test_xgb += clf_stra_xgb.predict(test_df[xtrain.columns])/num_of_splits





predictions_test_xgb = np.expm1(predictions_test_xgb)

print()

print(predictions_test_xgb)

print("OOF Out-of-fold rmse:", oof_rmse/num_of_splits)



f, ax = plt.subplots(2, figsize=(12,7))





sns.set(rc={'figure.figsize':(9,86)})

sns.distplot(train_df.revenue, ax=ax[0])

ax[0].set_title("Train Set Revenue Histogram")

sns.distplot(predictions_test_xgb, ax=ax[1])

ax[1].set_title("Test Set Revenue Prediction Histogram")

f.tight_layout()



xgb.plot_importance(clf_stra_xgb)
def bayesian_tuning(xtrain, ytrain):

    

    from skopt import BayesSearchCV

    import xgboost as xgb

    

    

    # Classifier

    bayes_cv_tuner = BayesSearchCV(

        estimator = xgb.XGBRegressor(

            nthread = -1,

            objective = 'reg:linear',

            verbosity=1,

            random_state=42

        ),

        search_spaces = {

            'learning_rate': (0.01, 1.0, 'log-uniform'),

            'min_child_weight': (0, 10),

            'n_estimators': (50, 300),

            'max_depth': (2, 12),

            'gamma': (1e-3, 1, 'log-uniform'),

            'subsample': (0.01, 1.0, 'uniform'),

            'colsample_bytree': (0.01, 1.0, 'uniform'),

            'colsample_bylevel': (0.01, 1.0, 'uniform'),

            'reg_lambda': (1e-1, 10, 'log-uniform'),

            'reg_alpha': (1e-2, 1.0, 'log-uniform')

        },

        cv = KFold(

            n_splits=kfold_splits,

            shuffle=True,

            random_state=42

        ),

        scoring = 'neg_mean_squared_error',

        n_jobs = 2,

        n_iter = 12,

        verbose=0,

        refit = True,

        random_state = 42

    )



    def status_print(optim_result):

        """Status callback durring bayesian hyperparameter search"""



        # Get all the models tested so far in DataFrame format

        all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    



        # Get current parameters and the best parameters    

        best_params = pd.Series(bayes_cv_tuner.best_params_)

        print('Model #{}\nBest score: {}\nBest params: {}\n'.format(

            len(all_models),

            np.round(bayes_cv_tuner.best_score_, 4),

            bayes_cv_tuner.best_params_

        ))

        

    result = bayes_cv_tuner.fit(xtrain, ytrain, callback = status_print)

    return result

    

# Fit the model

#xtrain, ytrain = prepare_for_tuning(X, y, type_of_training=type_of_training)

result = bayesian_tuning(xtrain, ytrain)
result.best_params_
from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import mean_squared_error



import xgboost as xgb



# create a 70/30 stratified split of the data 

xtrain, xvalid, ytrain, yvalid = train_test_split(X, y, random_state=42, test_size=0.3)



predictions_test_xgb_tuned = np.zeros(len(test_df))

num_fold = 0

oof_rmse = 0

num_of_splits = kfold_splits



folds = KFold(n_splits=num_of_splits, shuffle=False, random_state = 42)



for train_index, valid_index in folds.split(xtrain, ytrain):

    xtrain_stra, xvalid_stra = xtrain.iloc[train_index,:], xtrain.iloc[valid_index,:]

    ytrain_stra, yvalid_stra = ytrain.iloc[train_index], ytrain.iloc[valid_index]



    print()

    print("Fold:", num_fold)

    print()

    

    clf_stra_tuned_xgb = xgb.XGBRegressor(colsample_bytree = result.best_params_["colsample_bytree"],

                                          colsample_bylevel = result.best_params_["colsample_bylevel"],

                                    gamma=result.best_params_["gamma"],                 

                                    learning_rate=result.best_params_["learning_rate"],

                                    max_depth=result.best_params_["max_depth"],

                                    min_child_weight=result.best_params_["min_child_weight"],

                                    n_estimators=10000,

                                    reg_alpha=result.best_params_["reg_alpha"],

                                    reg_lambda=result.best_params_["reg_lambda"],

                                    subsample=result.best_params_["subsample"],

                                    seed=42,

                                    nthread = -1)



    clf_stra_tuned_xgb.fit(xtrain_stra, ytrain_stra, eval_set=[(xtrain_stra, ytrain_stra), (xvalid_stra, yvalid_stra)], 

                early_stopping_rounds=1000, eval_metric='rmse', verbose=100)



    predictions_valid = clf_stra_tuned_xgb.predict(xvalid)

    rmse_valid = np.sqrt(mean_squared_error(yvalid, predictions_valid))

    print("Fold",num_fold,"xvalid rmse:",rmse_valid)

    num_fold = num_fold + 1

    

    oof_rmse += rmse_valid



    predictions_test_xgb_tuned += clf_stra_tuned_xgb.predict(test_df[xtrain.columns])/num_of_splits

    

print()

predictions_test_xgb_tuned = np.expm1(predictions_test_xgb_tuned)

print(predictions_test_xgb_tuned)

print("OOF Out-of-fold rmse:", oof_rmse/num_of_splits)



f, ax = plt.subplots(2, figsize=(12,7))





sns.set(rc={'figure.figsize':(9,86)})

sns.distplot(train_df.revenue, ax=ax[0])

ax[0].set_title("Train Set Revenue Histogram")

sns.distplot(predictions_test_xgb_tuned, ax=ax[1])

ax[1].set_title("Test Set Revenue Prediction Histogram")

f.tight_layout()



xgb.plot_importance(clf_stra_tuned_xgb)
from sklearn.ensemble import ExtraTreesRegressor



from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import mean_squared_error

import scikitplot as skplt



# create a 70/30 stratified split of the data 

xtrain, xvalid, ytrain, yvalid = train_test_split(X, y, random_state=42, test_size=0.3)



predictions_extra_trees_test = np.zeros(len(test_df))

num_fold = 0

num_of_splits = kfold_splits

oof_rmse = 0



folds = KFold(n_splits=num_of_splits, shuffle=False, random_state = 42)



for train_index, valid_index in folds.split(xtrain, ytrain):

    xtrain_stra, xvalid_stra = xtrain.iloc[train_index,:], xtrain.iloc[valid_index,:]

    ytrain_stra, yvalid_stra = ytrain.iloc[train_index], ytrain.iloc[valid_index]



    print()

    print("Fold:", num_fold)

    print()



    clf_extra_trees = ExtraTreesRegressor(n_estimators=100, random_state=42)



    clf_extra_trees.fit(xtrain_stra, ytrain_stra)



    predictions_valid = clf_extra_trees.predict(xvalid)

    rmse_valid = np.sqrt(mean_squared_error(yvalid, predictions_valid))

    print("Fold" ,num_fold, "xvalid rmse:", rmse_valid)

    num_fold = num_fold + 1

    oof_rmse += rmse_valid



    predictions_extra_trees_test += clf_extra_trees.predict(test_df[xtrain.columns])/num_of_splits





predictions_extra_trees_test = np.expm1(predictions_extra_trees_test)

print()

print(predictions_extra_trees_test)

print()

print("OOF Out-of-fold rmse:", oof_rmse/num_of_splits)



f, ax = plt.subplots(2, figsize=(12,7))





sns.set(rc={'figure.figsize':(9,14)})

sns.distplot(train_df.revenue, ax=ax[0])

ax[0].set_title("Train Set Revenue Histogram")

sns.distplot(predictions_extra_trees_test, ax=ax[1])

ax[1].set_title("Test Set Revenue Prediction Histogram")

f.tight_layout()
def bayesian_tuning_extra_trees(xtrain, ytrain):

    

    from skopt import BayesSearchCV

    from sklearn.ensemble import ExtraTreesRegressor

    

    

    # Classifier

    bayes_cv_tuner = BayesSearchCV(

        estimator = ExtraTreesRegressor(

            random_state=42

        ),

        search_spaces = {

            'n_estimators': (10, 500),

            'max_depth': (1, 12),

            'min_samples_split': (2, 20),

            'min_samples_leaf': (1, 20)

        },

        cv = KFold(

            n_splits=kfold_splits,

            shuffle=True,

            random_state=42

        ),

        scoring = 'neg_mean_squared_error',

        n_jobs = 2,

        n_iter = 12,   

        verbose = 0,

        refit = True,

        random_state = 42

    )



    def status_print(optim_result):

        """Status callback durring bayesian hyperparameter search"""



        # Get all the models tested so far in DataFrame format

        all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    



        # Get current parameters and the best parameters    

        best_params = pd.Series(bayes_cv_tuner.best_params_)

        print('Model #{}\nBest score: {}\nBest params: {}\n'.format(

            len(all_models),

            np.round(bayes_cv_tuner.best_score_, 4),

            bayes_cv_tuner.best_params_

        ))

        

    result_extra_trees = bayes_cv_tuner.fit(xtrain, ytrain, callback = status_print)

    return result_extra_trees

    

# Fit the model

#xtrain, ytrain = prepare_for_tuning(X, y, type_of_training=type_of_training)

result_extra_trees = bayesian_tuning_extra_trees(xtrain, ytrain)
result_extra_trees.best_params_
from sklearn.ensemble import ExtraTreesRegressor



from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import mean_squared_error

import scikitplot as skplt



# create a 70/30 stratified split of the data 

xtrain, xvalid, ytrain, yvalid = train_test_split(X, y, random_state=42, test_size=0.3)



predictions_extra_trees_tuned_test = np.zeros(len(test_df))

num_fold = 0

num_of_splits = kfold_splits

oof_rmse = 0



folds = KFold(n_splits=num_of_splits, shuffle=False, random_state = 42)



for train_index, valid_index in folds.split(xtrain, ytrain):

    xtrain_stra, xvalid_stra = xtrain.iloc[train_index,:], xtrain.iloc[valid_index,:]

    ytrain_stra, yvalid_stra = ytrain.iloc[train_index], ytrain.iloc[valid_index]



    print()

    print("Fold:", num_fold)

    print()



    clf_extra_trees_tuned = ExtraTreesRegressor(random_state=42, 

                                                max_depth = result_extra_trees.best_params_['max_depth'], 

                                                min_samples_leaf = result_extra_trees.best_params_['min_samples_leaf'], 

                                                min_samples_split = result_extra_trees.best_params_['min_samples_split'], 

                                                n_estimators = result_extra_trees.best_params_['n_estimators'])



    clf_extra_trees_tuned.fit(xtrain_stra, ytrain_stra)



    predictions_valid = clf_extra_trees_tuned.predict(xvalid)

    rmse_valid = np.sqrt(mean_squared_error(yvalid, predictions_valid))

    print("Fold" ,num_fold, "xvalid rmse:", rmse_valid)

    num_fold = num_fold + 1

    oof_rmse += rmse_valid



    predictions_extra_trees_tuned_test += clf_extra_trees_tuned.predict(test_df[xtrain.columns])/num_of_splits





predictions_extra_trees_tuned_test = np.expm1(predictions_extra_trees_tuned_test)

print()

print(predictions_extra_trees_tuned_test)

print()

print("OOF Out-of-fold rmse:", oof_rmse/num_of_splits)



f, ax = plt.subplots(2, figsize=(12,7))





sns.set(rc={'figure.figsize':(9,14)})

sns.distplot(train_df.revenue, ax=ax[0])

ax[0].set_title("Train Set Revenue Histogram")

sns.distplot(predictions_extra_trees_tuned_test, ax=ax[1])

ax[1].set_title("Test Set Revenue Prediction Histogram")

f.tight_layout()
from sklearn.ensemble import RandomForestRegressor



from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import mean_squared_error

import scikitplot as skplt



# create a 70/30 stratified split of the data 

xtrain, xvalid, ytrain, yvalid = train_test_split(X, y, random_state=42, test_size=0.3)



predictions_random_forest_test = np.zeros(len(test_df))

num_fold = 0

num_of_splits = kfold_splits

oof_rmse = 0



folds = KFold(n_splits=num_of_splits, shuffle=False, random_state = 42)



for train_index, valid_index in folds.split(xtrain, ytrain):

    xtrain_stra, xvalid_stra = xtrain.iloc[train_index,:], xtrain.iloc[valid_index,:]

    ytrain_stra, yvalid_stra = ytrain.iloc[train_index], ytrain.iloc[valid_index]



    print()

    print("Fold:", num_fold)

    print()

    

    clf_random_forest = RandomForestRegressor(random_state=42, n_estimators = 100)



    clf_random_forest.fit(xtrain_stra, ytrain_stra)



    predictions_valid = clf_random_forest.predict(xvalid)

    rmse_valid = np.sqrt(mean_squared_error(yvalid, predictions_valid))

    print("Fold" ,num_fold, "xvalid rmse:", rmse_valid)

    num_fold = num_fold + 1

    oof_rmse += rmse_valid



    predictions_random_forest_test += clf_random_forest.predict(test_df[xtrain.columns])/num_of_splits





predictions_random_forest_test = np.expm1(predictions_random_forest_test)

print()

print(predictions_random_forest_test)

print()

print("OOF Out-of-fold rmse:", oof_rmse/num_of_splits)



f, ax = plt.subplots(2, figsize=(12,7))





sns.set(rc={'figure.figsize':(9,14)})

sns.distplot(train_df.revenue, ax=ax[0])

ax[0].set_title("Train Set Revenue Histogram")

sns.distplot(predictions_random_forest_test, ax=ax[1])

ax[1].set_title("Test Set Revenue Prediction Histogram")

f.tight_layout()
def bayesian_tuning_random_forest(xtrain, ytrain):

    

    from skopt import BayesSearchCV

    from sklearn.ensemble import RandomForestRegressor

    

    

    # Classifier

    bayes_cv_tuner = BayesSearchCV(

        estimator = RandomForestRegressor(

            random_state=42

        ),

        search_spaces = {

            'n_estimators': (10, 500),

            'max_depth': (1, 10),

            'min_samples_split': (2, 20),

            'min_samples_leaf': (1, 20)

        },

        cv = KFold(

            n_splits=kfold_splits,

            shuffle=True,

            random_state=42

        ),

        scoring = 'neg_mean_squared_error',

        n_jobs = 2,

        n_iter = 12,   

        verbose = 0,

        refit = True,

        random_state = 42

    )



    def status_print(optim_result):

        """Status callback durring bayesian hyperparameter search"""



        # Get all the models tested so far in DataFrame format

        all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    



        # Get current parameters and the best parameters    

        best_params = pd.Series(bayes_cv_tuner.best_params_)

        print('Model #{}\nBest score: {}\nBest params: {}\n'.format(

            len(all_models),

            np.round(bayes_cv_tuner.best_score_, 4),

            bayes_cv_tuner.best_params_

        ))

        

    result_random_forest = bayes_cv_tuner.fit(xtrain, ytrain, callback = status_print)

    return result_random_forest

    

# Fit the model

#xtrain, ytrain = prepare_for_tuning(X, y, type_of_training=type_of_training)

result_random_forest = bayesian_tuning_random_forest(xtrain, ytrain)
result_random_forest.best_params_
from sklearn.ensemble import RandomForestRegressor



from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import mean_squared_error

import scikitplot as skplt



# create a 70/30 stratified split of the data 

xtrain, xvalid, ytrain, yvalid = train_test_split(X, y, random_state=42, test_size=0.3)



predictions_random_forest_tuned_test = np.zeros(len(test_df))

num_fold = 0

num_of_splits = kfold_splits

oof_rmse = 0



folds = KFold(n_splits=num_of_splits, shuffle=False, random_state = 42)



for train_index, valid_index in folds.split(xtrain, ytrain):

    xtrain_stra, xvalid_stra = xtrain.iloc[train_index,:], xtrain.iloc[valid_index,:]

    ytrain_stra, yvalid_stra = ytrain.iloc[train_index], ytrain.iloc[valid_index]



    print()

    print("Fold:", num_fold)

    print()

    

    clf_random_forest_tuned = RandomForestRegressor(random_state=42, 

                                              n_estimators = result_random_forest.best_params_['n_estimators'],

                                              min_samples_leaf = result_random_forest.best_params_['min_samples_leaf'],

                                              min_samples_split = result_random_forest.best_params_['min_samples_split'])



    clf_random_forest_tuned.fit(xtrain_stra, ytrain_stra)



    predictions_valid = clf_random_forest_tuned.predict(xvalid)

    rmse_valid = np.sqrt(mean_squared_error(yvalid, predictions_valid))

    print("Fold xvalid rmse:", rmse_valid)

    oof_rmse += rmse_valid



    predictions_random_forest_tuned_test += clf_random_forest_tuned.predict(test_df[xtrain.columns])/num_of_splits





predictions_random_forest_tuned_test = np.expm1(predictions_random_forest_tuned_test)

print()

print(predictions_random_forest_test)

print()

print("OOF Out-of-fold rmse:", oof_rmse/num_of_splits)



f, ax = plt.subplots(2, figsize=(12,7))





sns.set(rc={'figure.figsize':(9,14)})

sns.distplot(train_df.revenue, ax=ax[0])

ax[0].set_title("Train Set Revenue Histogram")

sns.distplot(predictions_random_forest_tuned_test, ax=ax[1])

ax[1].set_title("Test Set Revenue Prediction Histogram")

f.tight_layout()
from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import mean_squared_error



import lightgbm as lgb



params = {

    "metric": 'rmse',

    "verbosity": -1

}



# create a 70/30 stratified split of the data 

xtrain, xvalid, ytrain, yvalid = train_test_split(X, y, random_state=42, test_size=0.3)



predictions_test_lgb = np.zeros(len(test_df))

num_fold = 0

oof_rmse = 0

num_of_splits = kfold_splits



folds = KFold(n_splits=num_of_splits, shuffle=False, random_state = 42)



for train_index, valid_index in folds.split(xtrain, ytrain):

    xtrain_stra, xvalid_stra = xtrain.iloc[train_index,:], xtrain.iloc[valid_index,:]

    ytrain_stra, yvalid_stra = ytrain.iloc[train_index], ytrain.iloc[valid_index]



    print()

    print("Fold:", num_fold)

    print()

    

    model_lgb = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)

    model_lgb.fit(xtrain_stra, ytrain_stra, 

        eval_set=[(xtrain_stra, ytrain_stra), (xvalid_stra, yvalid_stra)], eval_metric='rmse',

        verbose=100, early_stopping_rounds=1000)



    predictions_valid = model_lgb.predict(xvalid)

    rmse_valid = np.sqrt(mean_squared_error(yvalid, predictions_valid))

    print("Fold" ,num_fold, "xvalid rmse:", rmse_valid)

    num_fold = num_fold + 1

    

    oof_rmse += rmse_valid



    predictions_test_lgb += model_lgb.predict(test_df[xtrain.columns])/num_of_splits

    



predictions_test_lgb = np.expm1(predictions_test_lgb)

print()

print(predictions_test_lgb)

print("OOF Out-of-fold rmse:", oof_rmse/num_of_splits)



f, ax = plt.subplots(2, figsize=(12,7))



sns.set(rc={'figure.figsize':(9,14)})

sns.distplot(train_df.revenue, ax=ax[0])

ax[0].set_title("Train Set Revenue Histogram")

sns.distplot(predictions_test_lgb, ax=ax[1])

ax[1].set_title("Test Set Revenue Prediction Histogram")

f.tight_layout()



#xgb.plot_importance(clf_stra_fs_tuned_xgb)
def bayesian_tuning_lgb(xtrain, ytrain):

    

    from skopt import BayesSearchCV    

    

    # Classifier

    bayes_cv_tuner = BayesSearchCV(

        estimator = lgb.LGBMRegressor(

            boosting_type='gbdt', n_jobs=2, nthread = 4, verbose=-1

        ),

        search_spaces = {

            'num_leaves': (10, 100),

            'min_data_in_leaf': (10, 100),

            'n_estimators': (50, 100),

            'max_depth': (3, 12),

            'learning_rate': (0.01, 0.2, 'log-uniform'),

            "feature_fraction": (0.1, 1, 'uniform'),

            "bagging_fraction": (0.1, 1, 'uniform'),

            'lambda_l1': (0.1, 1, 'log-uniform'),

            'lambda_l2': (0.1, 1, 'log-uniform')

        },

        cv = KFold(

            n_splits=kfold_splits,

            shuffle=True,

            random_state=42

        ),

        scoring = 'neg_mean_squared_error',

        n_jobs = 1,

        n_iter = 12,   

        verbose = 0,

        refit = True,

        random_state = 42

    )



    def status_print(optim_result):

        """Status callback durring bayesian hyperparameter search"""



        # Get all the models tested so far in DataFrame format

        all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    



        # Get current parameters and the best parameters    

        best_params = pd.Series(bayes_cv_tuner.best_params_)

        print('Model #{}\nBest score: {}\nBest params: {}\n'.format(

            len(all_models),

            np.round(bayes_cv_tuner.best_score_, 4),

            bayes_cv_tuner.best_params_

        ))

        

    result_lgbm = bayes_cv_tuner.fit(xtrain, ytrain, callback = status_print)

    return result_lgbm

    

# Fit the model

#xtrain, ytrain = prepare_for_tuning(X, y, type_of_training=type_of_training)

result_lgbm = bayesian_tuning_lgb(xtrain, ytrain)
result_lgbm.best_params_
params = {

    'num_leaves': result_lgbm.best_params_["num_leaves"],

    'min_data_in_leaf': result_lgbm.best_params_["min_data_in_leaf"],

    'max_depth': result_lgbm.best_params_["max_depth"],

    'learning_rate': result_lgbm.best_params_["learning_rate"],

    "boosting": "gbdt",

    "feature_fraction": result_lgbm.best_params_["feature_fraction"],

    "bagging_freq": 1,

    "bagging_fraction": result_lgbm.best_params_["bagging_fraction"],

    "bagging_seed": 11,

    "metric": 'rmse',

    "lambda_l1": result_lgbm.best_params_["lambda_l1"],

    "lambda_l2": result_lgbm.best_params_["lambda_l2"],

    "verbosity": -1

}





from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import mean_squared_error



import lightgbm as lgb



# create a 70/30 stratified split of the data 

xtrain, xvalid, ytrain, yvalid = train_test_split(X, y, random_state=42, test_size=0.3)



predictions_test_lgb_tuned = np.zeros(len(test_df))

num_fold = 0

oof_rmse = 0

num_of_splits = kfold_splits



folds = KFold(n_splits=num_of_splits, shuffle=False, random_state = 42)



for train_index, valid_index in folds.split(xtrain, ytrain):

    xtrain_stra, xvalid_stra = xtrain.iloc[train_index,:], xtrain.iloc[valid_index,:]

    ytrain_stra, yvalid_stra = ytrain.iloc[train_index], ytrain.iloc[valid_index]



    print()

    print("Fold:", num_fold)

    print()

    

    model_lgb_tuned = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)

    model_lgb_tuned.fit(xtrain_stra, ytrain_stra, 

        eval_set=[(xtrain_stra, ytrain_stra), (xvalid_stra, yvalid_stra)], eval_metric='rmse',

        verbose=100, early_stopping_rounds=1000)



    predictions_valid = model_lgb_tuned.predict(xvalid)

    rmse_valid = np.sqrt(mean_squared_error(yvalid, predictions_valid))

    print("Fold" ,num_fold, "xvalid rmse:", rmse_valid)

    num_fold = num_fold + 1

    oof_rmse += rmse_valid



    predictions_test_lgb_tuned += model_lgb_tuned.predict(test_df[xtrain.columns])/num_of_splits

    



predictions_test_lgb_tuned = np.expm1(predictions_test_lgb_tuned)

print()

print(predictions_test_lgb_tuned)

print("OOF Out-of-fold rmse:", oof_rmse/num_of_splits)



f, ax = plt.subplots(2, figsize=(12,7))



sns.set(rc={'figure.figsize':(9,14)})

sns.distplot(train_df.revenue, ax=ax[0])

ax[0].set_title("Train Set Revenue Histogram")

sns.distplot(predictions_test_lgb_tuned, ax=ax[1])

ax[1].set_title("Test Set Revenue Prediction Histogram")

f.tight_layout()



#xgb.plot_importance(clf_stra_fs_tuned_xgb)
from sklearn.feature_selection import SelectFromModel



max_selected_features = 200

sel = SelectFromModel(clf_stra_xgb, max_features = max_selected_features, threshold=0.005, prefit=True)



feature_idx = sel.get_support()

selected_features_xgb = X.columns[feature_idx]
selected_features_xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

import time

import random

from sklearn.metrics import mean_squared_error



import xgboost as xgb



# create a 70/30 stratified split of the data 

xtrain, xvalid, ytrain, yvalid = train_test_split(X[selected_features_xgb], y, random_state=42, test_size=0.3)



import xgboost as xgb



start_time = time.time()



predictions_test_xgb_fs = np.zeros(len(test_df))

num_fold = 0

oof_rmse = 0

num_of_splits = kfold_splits



folds = KFold(n_splits=num_of_splits, shuffle=False, random_state = 42)



for train_index, valid_index in folds.split(xtrain, ytrain):

    xtrain_stra, xvalid_stra = xtrain.iloc[train_index,:], xtrain.iloc[valid_index,:]

    ytrain_stra, yvalid_stra = ytrain.iloc[train_index], ytrain.iloc[valid_index]



    print()

    print("Fold:", num_fold)

    print()

    

    

    clf_stra_fs_xgb = xgb.XGBRegressor(n_estimators=10000, seed=42, nthread = -1)



    clf_stra_fs_xgb.fit(xtrain_stra, ytrain_stra, eval_set=[(xtrain_stra, ytrain_stra), (xvalid_stra, yvalid_stra)], 

                early_stopping_rounds=1000, eval_metric='rmse', verbose=100)



    predictions_valid = clf_stra_fs_xgb.predict(xvalid)

    rmse_valid = np.sqrt(mean_squared_error(yvalid, predictions_valid))

    print("Fold" ,num_fold, "xvalid rmse:", rmse_valid)

    num_fold = num_fold + 1

    

    oof_rmse += rmse_valid



    predictions_test_xgb_fs += clf_stra_fs_xgb.predict(test_df[xtrain.columns])/num_of_splits

    



predictions_test_xgb_fs = np.expm1(predictions_test_xgb_fs)

print(predictions_test_xgb_fs)

print("OOF Out-of-fold rmse:", oof_rmse/num_of_splits)



f, ax = plt.subplots(2, figsize=(12,7))



sns.set(rc={'figure.figsize':(9,14)})

sns.distplot(train_df.revenue, ax=ax[0])

ax[0].set_title("Train Set Revenue Histogram")

sns.distplot(predictions_test_xgb_fs, ax=ax[1])

ax[1].set_title("Test Set Revenue Prediction Histogram")

f.tight_layout()



xgb.plot_importance(clf_stra_fs_xgb)
result = bayesian_tuning(xtrain, ytrain)
from sklearn.model_selection import train_test_split, KFold

import time

import random

from sklearn.metrics import mean_squared_error



import xgboost as xgb



# create a 70/30 stratified split of the data 

xtrain, xvalid, ytrain, yvalid = train_test_split(X[selected_features_xgb], y, random_state=42, test_size=0.3)



predictions_test_xgb_fs_tuned = np.zeros(len(test_df))

num_fold = 0

oof_rmse = 0

num_of_splits = kfold_splits



folds = KFold(n_splits=num_of_splits, shuffle=False, random_state = 42)



for train_index, valid_index in folds.split(xtrain, ytrain):

    xtrain_stra, xvalid_stra = xtrain.iloc[train_index,:], xtrain.iloc[valid_index,:]

    ytrain_stra, yvalid_stra = ytrain.iloc[train_index], ytrain.iloc[valid_index]



    print()

    print("Fold:", num_fold)

    print()

    

    

    clf_stra_fs_tuned_xgb = xgb.XGBRegressor(colsample_bytree = result.best_params_["colsample_bytree"],

                                    gamma=result.best_params_["gamma"],                 

                                    learning_rate=result.best_params_["learning_rate"],

                                    max_depth=result.best_params_["max_depth"],

                                    min_child_weight=result.best_params_["min_child_weight"],

                                    n_estimators=10000,

                                    reg_alpha=result.best_params_["reg_alpha"],

                                    reg_lambda=result.best_params_["reg_lambda"],

                                    subsample=result.best_params_["subsample"],

                                    seed=42,

                                    nthread = -1)



    clf_stra_fs_tuned_xgb.fit(xtrain_stra, ytrain_stra, eval_set=[(xtrain_stra, ytrain_stra), (xvalid_stra, yvalid_stra)], 

                early_stopping_rounds=1000, eval_metric='rmse', verbose=100)



    predictions_valid = clf_stra_fs_tuned_xgb.predict(xvalid)

    rmse_valid = np.sqrt(mean_squared_error(yvalid, predictions_valid))

    print("Fold" ,num_fold, "xvalid rmse:", rmse_valid)

    num_fold = num_fold + 1

    

    oof_rmse += rmse_valid



    predictions_test_xgb_fs_tuned += clf_stra_fs_tuned_xgb.predict(test_df[xtrain.columns])/num_of_splits

    



predictions_test_xgb_fs_tuned = np.expm1(predictions_test_xgb_fs_tuned)

print(predictions_test_xgb_fs_tuned)

print("OOF Out-of-fold rmse:", oof_rmse/num_of_splits)



f, ax = plt.subplots(2, figsize=(12,7))



sns.set(rc={'figure.figsize':(9,14)})

sns.distplot(train_df.revenue, ax=ax[0])

ax[0].set_title("Train Set Revenue Histogram")

sns.distplot(predictions_test_xgb_fs_tuned, ax=ax[1])

ax[1].set_title("Test Set Revenue Prediction Histogram")

f.tight_layout()



xgb.plot_importance(clf_stra_fs_tuned_xgb)
from sklearn.feature_selection import SelectFromModel



max_selected_features = 200

sel = SelectFromModel(model_lgb, max_features = max_selected_features, threshold=0.005, prefit=True)



feature_idx = sel.get_support()

selected_features_lgb = X.columns[feature_idx]
selected_features_lgb
from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import mean_squared_error



import lightgbm as lgb



params = {

    "metric": 'rmse',

    "verbosity": -1

}



# create a 70/30 stratified split of the data 

xtrain, xvalid, ytrain, yvalid = train_test_split(X[selected_features_lgb], y, random_state=42, test_size=0.3)



predictions_test_lgb_fs = np.zeros(len(test_df))

num_fold = 0

oof_rmse = 0

num_of_splits = kfold_splits



folds = KFold(n_splits=num_of_splits, shuffle=False, random_state = 42)



for train_index, valid_index in folds.split(xtrain, ytrain):

    xtrain_stra, xvalid_stra = xtrain.iloc[train_index,:], xtrain.iloc[valid_index,:]

    ytrain_stra, yvalid_stra = ytrain.iloc[train_index], ytrain.iloc[valid_index]



    print()

    print("Fold:", num_fold)

    print()

    

    model_lgb_fs = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)

    model_lgb_fs.fit(xtrain_stra, ytrain_stra, 

        eval_set=[(xtrain_stra, ytrain_stra), (xvalid_stra, yvalid_stra)], eval_metric='rmse',

        verbose=100, early_stopping_rounds=1000)



    predictions_valid = model_lgb_fs.predict(xvalid)

    rmse_valid = np.sqrt(mean_squared_error(yvalid, predictions_valid))

    print("Fold" ,num_fold, "xvalid rmse:", rmse_valid)

    num_fold = num_fold + 1

    

    oof_rmse += rmse_valid



    predictions_test_lgb_fs += model_lgb_fs.predict(test_df[xtrain.columns])/num_of_splits

    



predictions_test_lgb_fs = np.expm1(predictions_test_lgb_fs)

print()

print(predictions_test_lgb_fs)

print("OOF Out-of-fold rmse:", oof_rmse/num_of_splits)



f, ax = plt.subplots(2, figsize=(12,7))



sns.set(rc={'figure.figsize':(9,14)})

sns.distplot(train_df.revenue, ax=ax[0])

ax[0].set_title("Train Set Revenue Histogram")

sns.distplot(predictions_test_lgb_fs, ax=ax[1])

ax[1].set_title("Test Set Revenue Prediction Histogram")

f.tight_layout()



#xgb.plot_importance(clf_stra_fs_tuned_xgb)
result_lgb = bayesian_tuning_lgb(xtrain, ytrain)
params = {

    'num_leaves': result_lgbm.best_params_["num_leaves"],

    'min_data_in_leaf': result_lgbm.best_params_["min_data_in_leaf"],

    'max_depth': result_lgbm.best_params_["max_depth"],

    'learning_rate': result_lgbm.best_params_["learning_rate"],

    "boosting": "gbdt",

    "feature_fraction": result_lgbm.best_params_["feature_fraction"],

    "bagging_freq": 1,

    "bagging_fraction": result_lgbm.best_params_["bagging_fraction"],

    "bagging_seed": 11,

    "metric": 'rmse',

    "lambda_l1": result_lgbm.best_params_["lambda_l1"],

    "lambda_l2": result_lgbm.best_params_["lambda_l2"],

    "verbosity": -1

}





from sklearn.model_selection import train_test_split, KFold

from sklearn.metrics import mean_squared_error



import lightgbm as lgb



# create a 70/30 stratified split of the data 

xtrain, xvalid, ytrain, yvalid = train_test_split(X[selected_features_lgb], y, random_state=42, test_size=0.3)



predictions_test_lgb_fs_tuned = np.zeros(len(test_df))

num_fold = 0

oof_rmse = 0

num_of_splits = kfold_splits



folds = KFold(n_splits=num_of_splits, shuffle=False, random_state = 42)



for train_index, valid_index in folds.split(xtrain, ytrain):

    xtrain_stra, xvalid_stra = xtrain.iloc[train_index,:], xtrain.iloc[valid_index,:]

    ytrain_stra, yvalid_stra = ytrain.iloc[train_index], ytrain.iloc[valid_index]



    print()

    print("Fold:", num_fold)

    print()

    

    model_lgb_fs_tuned = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)

    model_lgb_fs_tuned.fit(xtrain_stra, ytrain_stra, 

        eval_set=[(xtrain_stra, ytrain_stra), (xvalid_stra, yvalid_stra)], eval_metric='rmse',

        verbose=100, early_stopping_rounds=1000)



    predictions_valid = model_lgb_fs_tuned.predict(xvalid)

    rmse_valid = np.sqrt(mean_squared_error(yvalid, predictions_valid))

    print("Fold" ,num_fold, "xvalid rmse:", rmse_valid)

    num_fold = num_fold + 1

    oof_rmse += rmse_valid



    predictions_test_lgb_fs_tuned += model_lgb_fs_tuned.predict(test_df[xtrain.columns])/num_of_splits

    



predictions_test_lgb_fs_tuned = np.expm1(predictions_test_lgb_fs_tuned)

print()

print(predictions_test_lgb_fs_tuned)

print("OOF Out-of-fold rmse:", oof_rmse/num_of_splits)



f, ax = plt.subplots(2, figsize=(12,7))



sns.set(rc={'figure.figsize':(9,14)})

sns.distplot(train_df.revenue, ax=ax[0])

ax[0].set_title("Train Set Revenue Histogram")

sns.distplot(predictions_test_lgb_fs_tuned, ax=ax[1])

ax[1].set_title("Test Set Revenue Prediction Histogram")

f.tight_layout()



#xgb.plot_importance(clf_stra_fs_tuned_xgb)
####### xgb with the other models



predictions_test_xgb_extra_trees = (0.5 * predictions_test_xgb) + (0.5 * predictions_extra_trees_test)

predictions_test_xgb_random_forest = (0.5 * predictions_test_xgb) + (0.5 * predictions_random_forest_test)

predictions_test_extra_trees_random_forest = (0.5 * predictions_extra_trees_test) + (0.5 * predictions_random_forest_test)



predictions_test_tuned_xgb_extra_trees = (0.5 * predictions_test_xgb_tuned) + (0.5 * predictions_extra_trees_tuned_test)

predictions_test_tuned_xgb_random_forest = (0.5 * predictions_test_xgb_tuned) + (0.5 * predictions_random_forest_tuned_test)

predictions_test_tuned_extra_trees_random_forest = (0.5 * predictions_extra_trees_tuned_test) + (0.5 * predictions_random_forest_tuned_test)



predictions_test_xgb_fs_extra_trees =  (0.5 * predictions_test_xgb_fs) + (0.5 * predictions_extra_trees_test)

predictions_test_xgb_fs_tuned_extra_trees = (0.5 * predictions_test_xgb_fs_tuned) + (0.5 * predictions_extra_trees_test)



predictions_test_baseline_xgb_tuned_extra_trees = (0.5 * predictions_test_xgb) + (0.5 * predictions_extra_trees_tuned_test)



####### lgb with the other models



predictions_test_lgb_xgb = (0.5 * predictions_test_lgb) + (0.5 * predictions_test_xgb)

predictions_test_tuned_lgb_xgb = (0.5 * predictions_test_lgb_tuned) + (0.5 * predictions_test_xgb_tuned)



predictions_test_lgb_extra_trees = (0.5 * predictions_test_lgb) + (0.5 * predictions_extra_trees_test)

predictions_test_lgb_random_forest = (0.5 * predictions_test_lgb) + (0.5 * predictions_random_forest_test)



predictions_test_tuned_lgb_extra_trees = (0.5 * predictions_test_lgb_tuned) + (0.5 * predictions_extra_trees_tuned_test)

predictions_test_tuned_lgb_random_forest = (0.5 * predictions_test_lgb_tuned) + (0.5 * predictions_random_forest_tuned_test)



predictions_test_xgb_fs_lgb =  (0.5 * predictions_test_xgb_fs) + (0.5 * predictions_test_lgb_fs)

predictions_test_xgb_fs_tuned_lgb = (0.5 * predictions_test_xgb_fs_tuned) + (0.5 * predictions_test_lgb_fs_tuned)





predictions_test_baseline_lgb_tuned_extra_trees = (0.5 * predictions_test_lgb) + (0.5 * predictions_extra_trees_tuned_test)
# xgb baseline

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_test_xgb

submission.to_csv('clf_xgb_baseline.csv', index=False)



# xgb tuning

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_test_xgb_tuned

submission.to_csv('clf_xgb_tuned.csv', index=False)



# lgb baseline

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_test_lgb

submission.to_csv('clf_lgb_baseline.csv', index=False)



# lgb tuning

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_test_lgb_tuned

submission.to_csv('clf_lgb_tuned.csv', index=False)



# extra trees baseline

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_extra_trees_test

submission.to_csv('clf_extra_trees_baseline.csv', index=False)



# extra trees tuning

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_extra_trees_tuned_test

submission.to_csv('clf_extra_trees_tuned.csv', index=False)



# xgb baseline with feature selection

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_test_xgb_fs

submission.to_csv('clf_xgb_fs_baseline.csv', index=False)



# xgb tuning with feature selection

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_test_xgb_fs_tuned

submission.to_csv('clf_xgb_fs_tuned.csv', index=False)



# lgb baseline with feature selection

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_test_lgb_fs

submission.to_csv('clf_lgb_fs_baseline.csv', index=False)



# lgb tuning with feature selection

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_test_lgb_fs_tuned

submission.to_csv('clf_lgb_fs_tuned.csv', index=False)



# Blend 1

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_test_xgb_extra_trees

submission.to_csv('blend_xgb_extra_trees_baselines.csv', index=False)



# Blend 2

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_test_xgb_random_forest

submission.to_csv('blend_xgb_random_forest_baselines.csv', index=False)



# Blend 3

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_test_extra_trees_random_forest

submission.to_csv('blend_extra_trees_random_forest_baselines.csv', index=False)



# Blend 4

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_test_tuned_xgb_extra_trees

submission.to_csv('blend_xgb_extra_trees_tuned.csv', index=False)



# Blend 5

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_test_tuned_xgb_random_forest

submission.to_csv('blend_xgb_random_forest_tuned.csv', index=False)



# Blend 6

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_test_tuned_extra_trees_random_forest

submission.to_csv('blend_extra_trees_random_forest_tuned.csv', index=False)



# Blend 7

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_test_baseline_xgb_tuned_extra_trees

submission.to_csv('blend_baseline_xgb_tuned_extra_trees.csv', index=False)



# Blend 8

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_test_xgb_fs_extra_trees

submission.to_csv('blend_xgb_fs_extra_trees.csv', index=False)



# Blend 9

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_test_xgb_fs_tuned_extra_trees

submission.to_csv('blend_xgb_fs_tuned_extra_trees.csv', index=False)



# Blend 10

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_test_lgb_xgb

submission.to_csv('blend_lgb_xgb.csv', index=False)



# Blend 11

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_test_tuned_lgb_xgb

submission.to_csv('blend_tuned_lgb_xgb.csv', index=False)



# Blend 12

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_test_lgb_extra_trees

submission.to_csv('blend_tuned_lgb_extra_trees.csv', index=False)



# Blend 13

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_test_lgb_extra_trees

submission.to_csv('blend_tuned_lgb_extra_trees.csv', index=False)



# Blend 14

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_test_lgb_random_forest

submission.to_csv('blend_tuned_lgb_random_forest.csv', index=False)



# Blend 15

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_test_tuned_lgb_extra_trees

submission.to_csv('blend_tuned_lgb_extra_trees.csv', index=False)



# Blend 16

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_test_tuned_lgb_random_forest

submission.to_csv('blend_tuned_lgb_random_forest.csv', index=False)



# Blend 17

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_test_baseline_lgb_tuned_extra_trees

submission.to_csv('blend_baseline_lgb_tuned_extra_trees.csv', index=False)



# Blend 18

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_test_xgb_fs_lgb

submission.to_csv('blend_xgb_fs_lgb.csv', index=False)



# Blend 19

submission = pd.read_csv('../input/sample_submission.csv')

submission['revenue'] = predictions_test_xgb_fs_tuned_lgb

submission.to_csv('blend_xgb_fs_tuned_lgb.csv', index=False)
