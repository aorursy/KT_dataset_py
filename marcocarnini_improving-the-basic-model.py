import pandas as pd



df = pd.read_csv("../input/train.csv")

print(df.info())
from sklearn.metrics.scorer import make_scorer



def rmlse(y, y0):

    assert len(y) == len(y0)

    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(np.clip(y0, 0, None)), 2)))



rmsle_scorer = make_scorer(rmlse, greater_is_better=False)
print(set(df.status))
import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score



temp = df[["runtime"]].dropna()

imputation = np.median(temp.runtime)

train = pd.DataFrame(df[["budget", "popularity", "status"]])

train = pd.get_dummies(train)

train["runtime"] = df.runtime.fillna(imputation)

train["homepage_missing"] = np.array(df.homepage.isna(), dtype=int)

train["belongs_to_collection_missing"] = np.array(df.belongs_to_collection.isna(), dtype=int)

label = df["revenue"]



model  = RandomForestRegressor(n_estimators=100, random_state=2019)

scores_randomforest = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

print(-np.mean(scores_randomforest), "+/-" ,np.std(scores_randomforest))
print(set(df.release_date))
print(sum(df.release_date.isna()))
train["release_day"] = [i.split("/")[1] for i in df.release_date]

train["release_month"] = [i.split("/")[0] for i in df.release_date]

train["release_year"] = [i.split("/")[2] for i in df.release_date]

train["release_year"] = ["20"+i if int(i) < 18 else "19"+i for i in train.release_year]

print(train.head())
train = pd.DataFrame(df[["budget", "popularity", "runtime", "status"]])

train = pd.get_dummies(train)

train.runtime = train.runtime.fillna(imputation)

train["homepage_missing"] = np.array(df.homepage.isna(), dtype=int)

train["belongs_to_collection_missing"] = np.array(df.belongs_to_collection.isna(), dtype=int)

train["release_day"] = [i.split("/")[1] for i in df.release_date]

train["release_month"] = [i.split("/")[0] for i in df.release_date]

train["release_year"] = [i.split("/")[2] for i in df.release_date]

train["release_year"] = ["20"+i if int(i) < 18 else "19"+i for i in train.release_year]

label = df["revenue"]



model  = RandomForestRegressor(n_estimators=100)

model.fit(train, label)
test = pd.read_csv("../input/test.csv")

dfte = pd.DataFrame(test[["budget", "popularity", "runtime"]])

dfte["homepage_missing"] = np.array(test.homepage.isna(), dtype=int)

dfte["belongs_to_collection_missing"] = np.array(test.belongs_to_collection.isna(), dtype=int)

dfte.runtime = dfte.runtime.fillna(imputation)



dfte["release_day"] = [i.split("/")[1] for i in test.release_date]

dfte["release_month"] = [i.split("/")[0] for i in test.release_date]

dfte["release_year"] = [i.split("/")[2] for i in test.release_date]

dfte["release_year"] = ["20"+i if int(i) < 18 else "19"+i for i in dfte.release_year]
print(sum(test.release_date.isna()))
missing_row = test[test.release_date.isna()]

print(missing_row)

print(missing_row.title)
print(test.title[0])

print(test.release_date[0])
df = pd.read_csv("../input/train.csv")

print(df[df.runtime.isna()].title)

print(df[df.runtime.isna()].release_date)
df = pd.read_csv("../input/train.csv")

train = pd.DataFrame(df[["budget", "popularity", "runtime", "status"]])

train = pd.get_dummies(train)



train.loc[1335, "runtime"] = 130.0

train.loc[2302, "runtime"] = 90.0

train["homepage_missing"] = np.array(df.homepage.isna(), dtype=int)

train["belongs_to_collection_missing"] = np.array(df.belongs_to_collection.isna(), dtype=int)

train["release_day"] = [i.split("/")[1] for i in df.release_date]

train["release_month"] = [i.split("/")[0] for i in df.release_date]

train["release_year"] = [i.split("/")[2] for i in df.release_date]

train["release_year"] = ["20"+i if int(i) < 18 else "19"+i for i in train.release_year]

label = df["revenue"]



model  = RandomForestRegressor(n_estimators=100)

model.fit(train, label)
print(test[test.runtime.isna()].title)

print(test[test.runtime.isna()].release_date)
test = pd.read_csv("../input/test.csv")

dfte = pd.DataFrame(test[["budget", "popularity", "runtime", "status"]])

dfte = pd.get_dummies(dfte)



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
predictions = model.predict(dfte)

predictions = np.clip(predictions, 0, None)

submission = pd.DataFrame({

    "id" : test.id,

    "revenue": predictions

})

submission.to_csv("submission.csv", index=False)
print(dfte.info())
print(train.info())
df = pd.read_csv("../input/train.csv")

train = pd.DataFrame(df[["budget", "popularity", "runtime", "status"]])

train = pd.get_dummies(train)

train["status_Post Production"] = 0



train.loc[1335, "runtime"] = 130.0

train.loc[2302, "runtime"] = 90.0

train["homepage_missing"] = np.array(df.homepage.isna(), dtype=int)

train["belongs_to_collection_missing"] = np.array(df.belongs_to_collection.isna(), dtype=int)

train["release_day"] = [i.split("/")[1] for i in df.release_date]

train["release_month"] = [i.split("/")[0] for i in df.release_date]

train["release_year"] = [i.split("/")[2] for i in df.release_date]

train["release_year"] = ["20"+i if int(i) < 18 else "19"+i for i in train.release_year]

label = df["revenue"]
model  = RandomForestRegressor(n_estimators=100, random_state=2019)

scores_randomforest = cross_val_score(model, train, label, cv=10, scoring=rmsle_scorer)

print(-np.mean(scores_randomforest), "+/-" ,np.std(scores_randomforest))
model  = RandomForestRegressor(n_estimators=100)

model.fit(train, label)
test = pd.read_csv("../input/test.csv")

dfte = pd.DataFrame(test[["budget", "popularity", "runtime", "status"]])

dfte = pd.get_dummies(dfte)



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
predictions = model.predict(dfte)

predictions = np.clip(predictions, 0, None)

submission = pd.DataFrame({

    "id" : test.id,

    "revenue": predictions

})

submission.to_csv("submission.csv", index=False)