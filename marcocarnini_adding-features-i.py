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
print(df.poster_path[0:10])
extension = [i.split(".")[-1] for i in df.poster_path.dropna()]

print(set(extension))

length = [len(i) for i in df.poster_path.dropna()]

labels = [i for i in df.revenue[df.poster_path.notnull()]]

temp = pd.DataFrame({"length": length, "labels": labels})

print(temp.groupby(length).agg(["mean", "count"]))
df["poster_length"] = 0

df.loc[df.poster_path.notnull(), "poster_length"] = [len(i) for i in df.poster_path[df.poster_path.notnull()]]

print(set(df.poster_length))
print(set(df.homepage))
temp = list(df.homepage[df.homepage.dropna()].index)

temp_df = pd.DataFrame({

    "contains_com": [".com" in i for i in temp],

    "contains_uk": [".uk" in i for i in temp],

    "contains_fr": [".fr" in i for i in temp],

    "contains_de": [".de" in i for i in temp],

    "contains_net": [".net" in i for i in temp],

    "contains_kr": [".kr" in i for i in temp],

    "contains_disney": ["disney" in i for i in temp],

    "contains_sony": ["sony" in i for i in temp],

    "contains_warnerbros": ["warnerbros" in i for i in temp],

    "contains_indexhtml": ["index.html" in i for i in temp],

    "contains_movie": ["movie" in i.lower() for i in temp],

    "contains_wikipedia": ["wikipedia" in i.lower() for i in temp],

    "count_slash": [len(i.split("/")) for i in temp]

})

print(np.mean(temp_df.contains_com))

print(np.mean(temp_df.contains_uk))

print(np.mean(temp_df.contains_fr))

print(np.mean(temp_df.contains_de))

print(np.mean(temp_df.contains_net))

print(np.mean(temp_df.contains_kr))

print(np.mean(temp_df.contains_disney))

print(np.mean(temp_df.contains_sony))

print(np.mean(temp_df.contains_warnerbros))

print(np.mean(temp_df.contains_indexhtml))

print(np.mean(temp_df.contains_movie))

print(np.mean(temp_df.contains_wikipedia))

print(set(temp_df.count_slash))
import pandas as pd

import numpy as np



df = pd.read_csv("../input/train.csv")

train = pd.DataFrame(df[["budget", "popularity", "runtime", "status", "original_language"]])

train = pd.get_dummies(train)

test = pd.read_csv("../input/test.csv")

dfte = pd.DataFrame(test[["budget", "popularity", "runtime", "status", "original_language"]])

dfte = pd.get_dummies(dfte)

missing_columns = set(dfte.columns) - set(train.columns)

print(missing_columns)

for _ in missing_columns:

    train[_] = 0

missing_columns = set(train.columns) - set(dfte.columns)

print(missing_columns)

for _ in missing_columns:

    dfte[_] = 0
train.loc[1335, "runtime"] = 130.0

train.loc[2302, "runtime"] = 90.0

train["homepage_missing"] = np.array(df.homepage.isna(), dtype=int)

train["belongs_to_collection_missing"] = np.array(df.belongs_to_collection.isna(), dtype=int)

train["release_day"] = [i.split("/")[1] for i in df.release_date]

train["release_month"] = [i.split("/")[0] for i in df.release_date]

train["release_year"] = [i.split("/")[2] for i in df.release_date]

train["release_year"] = ["20"+i if int(i) < 18 else "19"+i for i in train.release_year]



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

dfte["release_day"] = [i.split("/")[1] for i in test.release_date]

dfte["release_month"] = [i.split("/")[0] for i in test.release_date]

dfte["release_year"] = [i.split("/")[2] for i in test.release_date]

dfte["release_year"] = ["20"+i if int(i) < 18 else "19"+i for i in dfte.release_year]



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
predictions = model.predict(dfte)

predictions = np.clip(predictions, 0, None)

submission = pd.DataFrame({

    "id" : test.id,

    "revenue": predictions

})

submission.to_csv("submission.csv", index=False)