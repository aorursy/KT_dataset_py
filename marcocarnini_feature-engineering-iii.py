import pandas as pd



df = pd.read_csv("../input/train.csv")

print(df.info())
from sklearn.metrics.scorer import make_scorer



def rmlse(y, y0):

    assert len(y) == len(y0)

    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(np.clip(y0, 0, None)), 2)))



rmsle_scorer = make_scorer(rmlse, greater_is_better=False)
import pandas as pd



df = pd.read_csv("../input/train.csv")

print(df.info())
test = pd.read_csv("../input/test.csv")

print(test.info())
print(df.genres)
import numpy as np



allgenres = set([i["name"] for j in df.genres[df.genres.notnull()] for i in eval(j)])

d = {i: np.zeros(df.shape[0], dtype=int) for i in allgenres}

d["Missing"] = np.zeros(df.shape[0], dtype=int)

genres_matrix = pd.DataFrame(d)



genres_matrix["Missing"][pd.isnull(df.genres)] = 1.0

for j,i in enumerate(pd.notnull(df.genres)):

    if i:

        for k in eval(df.genres[j]):

            genres_matrix.loc[j, k["name"]] += 1

genres_matrix["genres_number"] = genres_matrix.apply(sum, axis=1)
print(set(genres_matrix.genres_number))
print(df.production_companies[0:10])
allcompanies = set([i["name"] for j in df.production_companies[df.production_companies.notnull()] for i in eval(j)])

d = {i: np.zeros(df.shape[0], dtype=int) for i in allcompanies}

d["Missing"] = np.zeros(df.shape[0], dtype=int)

companies_matrix = pd.DataFrame(d)



companies_matrix["Missing"][pd.isnull(df.production_companies)] = 1.0

for j,i in enumerate(pd.notnull(df.production_companies)):

    if i:

        for k in eval(df.production_companies[j]):

            companies_matrix.loc[j, k["name"]] += 1

companies_matrix["companies_number"] = companies_matrix.sum(axis=1)-companies_matrix.Missing
import pandas as pd

import numpy as np



df = pd.read_csv("../input/train.csv")

train = pd.DataFrame(df[["budget", "popularity", "runtime", "status", "original_language"]])

train = pd.get_dummies(train)

test = pd.read_csv("../input/test.csv")

dfte = pd.DataFrame(test[["budget", "popularity", "runtime", "status", "original_language"]])

dfte = pd.get_dummies(dfte)

missing_columns = set(dfte.columns) - set(train.columns)

for _ in missing_columns:

    train[_] = 0

missing_columns = set(train.columns) - set(dfte.columns)

for _ in missing_columns:

    dfte[_] = 0
train.loc[1335, "runtime"] = 130.0

train.loc[2302, "runtime"] = 90.0

train["homepage_missing"] = np.array(df.homepage.isna(), dtype=int)

train["belongs_to_collection_missing"] = np.array(df.belongs_to_collection.isna(), dtype=int)

train["release_day"] = [int(i.split("/")[1]) for i in df.release_date]

train["release_month"] = [int(i.split("/")[0]) for i in df.release_date]

train["release_year"] = [int(i.split("/")[2]) for i in df.release_date]

train["release_year"] = [2000+i if i < 18 else 1900+i for i in train.release_year]



train["poster_length"] = 0

train.loc[df.poster_path.notnull(), "poster_length"] = [len(i) for i in df.poster_path[df.poster_path.notnull()]]



label = df["revenue"]
train["contains_com"] = 0

train["contains_uk"] = 0

train["contains_fr"] = 0

train["contains_de"] = 0

train["contains_net"] = 0

train["contains_kr"] = 0

train["contains_disney"] = 0

train["contains_sony"] = 0

train["contains_warnerbros"] = 0

train["contains_indexhtml"] = 0

train["contains_movie"] = 0

train["contains_wikipedia"] = 0

train["count_slash"] = 0



train.loc[df.homepage.notnull(), "contains_com"] = [1 if ((i != "") & (".com" in i)) else 0 for i in df.homepage[df.homepage.notnull()]]

train.loc[df.homepage.notnull(), "contains_uk"] = [1 if ((i != "") & (".uk" in i)) else 0 for i in df.homepage[df.homepage.notnull()]]

train.loc[df.homepage.notnull(), "contains_fr"] = [1 if ((i != "") & (".fr" in i)) else 0 for i in df.homepage[df.homepage.notnull()]]

train.loc[df.homepage.notnull(), "contains_de"] = [1 if ((i != "") & (".de" in i)) else 0 for i in df.homepage[df.homepage.notnull()]]

train.loc[df.homepage.notnull(), "contains_net"] = [1 if ((i != "") & (".net" in i)) else 0 for i in df.homepage[df.homepage.notnull()]]

train.loc[df.homepage.notnull(), "contains_kr"] = [1 if ((i != "") & (".kr" in i)) else 0 for i in df.homepage[df.homepage.notnull()]]

train.loc[df.homepage.notnull(), "contains_disney"] = [1 if ((i != "") & ("disney" in i)) else 0 for i in df.homepage[df.homepage.notnull()]]

train.loc[df.homepage.notnull(), "contains_sony"] = [1 if ((i != "") & ("sony" in i)) else 0 for i in df.homepage[df.homepage.notnull()]]

train.loc[df.homepage.notnull(), "contains_warnerbros"] = [1 if ((i != "") & ("warnerbros" in i)) else 0 for i in df.homepage[df.homepage.notnull()]]

train.loc[df.homepage.notnull(), "contains_indexhtml"] = [1 if ((i != "") & ("index.html" in i)) else 0 for i in df.homepage[df.homepage.notnull()]]

train.loc[df.homepage.notnull(), "contains_movie"] = [1 if ((i != "") & ("movie" in i.lower())) else 0 for i in df.homepage[df.homepage.notnull()]]

train.loc[df.homepage.notnull(), "contains_wikipedia"] = [1 if ((i != "") & ("wikipedia" in i)) else 0 for i in df.homepage[df.homepage.notnull()]]

train.loc[df.homepage.notnull(), "count_slash"] = [len(i.split("/")) for i in df.homepage[df.homepage.notnull()]]
import numpy as np



d = {i: np.zeros(df.shape[0], dtype=int) for i in allgenres}

d["Missing"] = np.zeros(df.shape[0], dtype=int)

genres_matrix = pd.DataFrame(d)



genres_matrix["Missing"][pd.isnull(df.genres)] = 1.0

for j,i in enumerate(pd.notnull(df.genres)):

    if i:

        for k in eval(df.genres[j]):

            genres_matrix.loc[j, k["name"]] += 1

genres_matrix["genres_number"] = genres_matrix.sum(axis=1)-genres_matrix.Missing

print(genres_matrix.shape)
allcompanies = set([i["name"] for j in df.production_companies[df.production_companies.notnull()] for i in eval(j)])

d = {i: np.zeros(train.shape[0], dtype=int) for i in allcompanies}

d["Missing"] = np.zeros(train.shape[0], dtype=int)

companies_matrix = pd.DataFrame(d)



companies_matrix["Missing"][pd.isnull(df.production_companies)] = 1.0

for j,i in enumerate(pd.notnull(df.production_companies)):

    if i:

        for k in eval(df.production_companies[j]):

            companies_matrix.loc[j, k["name"]] += 1

companies_matrix["companies_number"] = companies_matrix.sum(axis=1)-companies_matrix.Missing

print(companies_matrix.shape)
train["row"] = np.linspace(0, train.shape[0], train.shape[0], dtype=int)

genres_matrix["row"]= np.linspace(0, train.shape[0], train.shape[0], dtype=int)

companies_matrix["row"]= np.linspace(0, train.shape[0], train.shape[0], dtype=int)

train = pd.concat([train, genres_matrix, companies_matrix], axis=1, join="inner")

train.drop(["row"], axis = 1, inplace=False)

print(train.shape)
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score



model  = RandomForestRegressor(n_estimators=100, random_state=2019)

scores_randomforest = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

print(-np.mean(scores_randomforest), "+/-" ,np.std(scores_randomforest))
model  = RandomForestRegressor(n_estimators=100)

model.fit(train, label)
dfte["homepage_missing"] = np.array(test.homepage.isna(), dtype=int)

dfte["belongs_to_collection_missing"] = np.array(test.belongs_to_collection.isna(), dtype=int)

dfte.loc[243, "runtime"] = 93.0

dfte.loc[1489, "runtime"] = 91.0

dfte.loc[1632, "runtime"] = 100.0

dfte.loc[3817, "runtime"] = 90.0



test.loc[828, "release_date"] = "03/30/2001"

dfte["release_day"] = [int(i.split("/")[1]) for i in test.release_date]

dfte["release_month"] = [int(i.split("/")[0]) for i in test.release_date]

dfte["release_year"] = [int(i.split("/")[2]) for i in test.release_date]

dfte["release_year"] = [2000+i if i < 18 else 1900+i for i in dfte.release_year]



dfte["poster_length"] = 0

dfte.loc[test.poster_path.notnull(), "poster_length"] = [len(i) for i in test.poster_path[test.poster_path.notnull()]]
dfte["contains_com"] = 0

dfte["contains_uk"] = 0

dfte["contains_fr"] = 0

dfte["contains_de"] = 0

dfte["contains_net"] = 0

dfte["contains_kr"] = 0

dfte["contains_disney"] = 0

dfte["contains_sony"] = 0

dfte["contains_warnerbros"] = 0

dfte["contains_indexhtml"] = 0

dfte["contains_movie"] = 0

dfte["contains_wikipedia"] = 0

dfte["count_slash"] = 0



dfte.loc[test.homepage.notnull(), "contains_com"] = [1 if ((i != "") & (".com" in i)) else 0 for i in test.homepage[test.homepage.notnull()]]

dfte.loc[test.homepage.notnull(), "contains_uk"] = [1 if ((i != "") & (".uk" in i)) else 0 for i in test.homepage[test.homepage.notnull()]]

dfte.loc[test.homepage.notnull(), "contains_fr"] = [1 if ((i != "") & (".fr" in i)) else 0 for i in test.homepage[test.homepage.notnull()]]

dfte.loc[test.homepage.notnull(), "contains_de"] = [1 if ((i != "") & (".de" in i)) else 0 for i in test.homepage[test.homepage.notnull()]]

dfte.loc[test.homepage.notnull(), "contains_net"] = [1 if ((i != "") & (".net" in i)) else 0 for i in test.homepage[test.homepage.notnull()]]

dfte.loc[test.homepage.notnull(), "contains_kr"] = [1 if ((i != "") & (".kr" in i)) else 0 for i in test.homepage[test.homepage.notnull()]]

dfte.loc[test.homepage.notnull(), "contains_disney"] = [1 if ((i != "") & ("disney" in i)) else 0 for i in test.homepage[test.homepage.notnull()]]

dfte.loc[test.homepage.notnull(), "contains_sony"] = [1 if ((i != "") & ("sony" in i)) else 0 for i in test.homepage[test.homepage.notnull()]]

dfte.loc[test.homepage.notnull(), "contains_warnerbros"] = [1 if ((i != "") & ("warnerbros" in i)) else 0 for i in test.homepage[test.homepage.notnull()]]

dfte.loc[test.homepage.notnull(), "contains_indexhtml"] = [1 if ((i != "") & ("index.html" in i)) else 0 for i in test.homepage[test.homepage.notnull()]]

dfte.loc[test.homepage.notnull(), "contains_movie"] = [1 if ((i != "") & ("movie" in i.lower())) else 0 for i in test.homepage[test.homepage.notnull()]]

dfte.loc[test.homepage.notnull(), "contains_wikipedia"] = [1 if ((i != "") & ("wikipedia" in i)) else 0 for i in test.homepage[test.homepage.notnull()]]

dfte.loc[test.homepage.notnull(), "count_slash"] = [len(i.split("/")) for i in test.homepage[test.homepage.notnull()]]
import numpy as np



d = {i: np.zeros(test.shape[0], dtype=int) for i in allgenres}

d["Missing"] = np.zeros(test.shape[0], dtype=int)

genres_matrix = pd.DataFrame(d)



genres_matrix["Missing"][pd.isnull(test.genres)] = 1.0

for j,i in enumerate(pd.notnull(test.genres)):

    if i:

        for k in eval(test.genres[j]):

            genres_matrix.loc[j, k["name"]] += 1

genres_matrix["genres_number"] = genres_matrix.sum(axis=1)-genres_matrix.Missing

print(genres_matrix.shape)
d1 = {i: np.zeros(test.shape[0], dtype=int) for i in allcompanies}

d1["Missing"] = np.zeros(test.shape[0], dtype=int)

companies_matrix = pd.DataFrame(d1)



companies_matrix["Missing"][pd.isnull(test.production_companies)] = 1.0

for j,i in enumerate(pd.notnull(test.production_companies)):

    if i:

        for k in eval(test.production_companies[j]):

            if (k["name"] in d1.keys()):

                companies_matrix.loc[j, k["name"]] += 1

companies_matrix["companies_number"] = companies_matrix.sum(axis=1)-companies_matrix.Missing

print(companies_matrix.shape)
dfte["row"] = np.linspace(0, dfte.shape[0], dfte.shape[0], dtype=int)

genres_matrix["row"]= np.linspace(0, dfte.shape[0], dfte.shape[0], dtype=int)

companies_matrix["row"]= np.linspace(0, dfte.shape[0], dfte.shape[0], dtype=int)

dfte = pd.concat([dfte, genres_matrix, companies_matrix], axis=1, join="inner")

dfte.drop(["row"], axis = 1, inplace=False)
predictions = model.predict(dfte)

predictions = np.clip(predictions, 0, None)

submission = pd.DataFrame({

    "id" : test.id,

    "revenue": predictions

})

submission.to_csv("submission.csv", index=False)