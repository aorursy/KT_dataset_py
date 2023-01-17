import numpy as np
import pandas as pd
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.listdir("../input")

act = pd.read_csv("../input/actions.csv", error_bad_lines=False, sep=";")
msno.matrix(act.sample(2000))
act = act.drop(["game_app_category", 
                "game_app_version_id", 
                "game_app_version", 
                "book_type", 
                "publisher_id", 
                "time_to_study",
                "user_id",
#                 "action_info",
                "logical_type",
                "content_type",
                "school_id",
                "action_school_id",
                ], axis=1)

columns_to_encode = ["material_type", "profile_type", "action_type", "action_profile_type"]
act = act.dropna()
pd.Series({column: len(act[column].unique()) for column in columns_to_encode}).plot(kind="bar", title="Уникальные значения")
act["action_type"].value_counts().plot("bar", title="Количество уникальных действий", figsize=(20, 20))
before = act.shape
act = act[
    (act["action_type"] != "обновлён") &
    (act["action_type"] != "удаление") &
    (act["action_type"] != "загрузка в библиотеку") &
    (act["action_type"] != "отправка на модерацию") &
    (act["action_type"] != "одобрено модерацией")
]

print("Before:", before)
print(" After:", act.shape)
fig, subplots = plt.subplots(nrows=2, ncols=2, figsize=(20,20))
subplots_dict = {0: subplots[0, 0], 1: subplots[0, 1], 2: subplots[1, 0], 3: subplots[1, 1]}

for i in range(4):
    info = act[columns_to_encode[i]].value_counts()
    try:  # Можно менять значение
        info.where(100000000 > info).dropna().plot("bar", ax=subplots_dict[i], title=columns_to_encode[i])
    except TypeError:
        info.plot("bar", ax=subplots_dict[i], title=columns_to_encode[i])

plt.show()
before = act.shape
act = act[(act["action_profile_type"] == "TeacherProfile") | (act["action_profile_type"] == "StudentProfile")]
act[act["action_end"] != act["action_start"]].shape
act = act.drop(["action_end"], axis=1)
act = act[act["action_type"] == "оценка"].drop(["action_type"], axis=1)
print("Before:", before)
print(" After:", act.shape)


import datetime
def make_weekday(date):
    date = [int(i) for i in date.split("-")]
    return datetime.date(date[0], date[1], date[2]).weekday()

def make_seconds(time):
    time = [int(float(el)) for el in time.split(":")]
    return time[0] * 3600 + time[1] * 60 + time[2]

def make_cyclic(value, cos, period):
    value *= 2 * np.pi / period
    if cos:
        return np.cos(value)
    return np.sin(value)

def make_sorting_seconds(time):
    date, time = time.split()
    date, time = [int(i) for i in date.split("-")], [int(float(i)) for i in time.split(":")]
    return datetime.datetime(date[0], date[1], date[2], hour=time[0], minute=time[1], second=time[2]).timestamp()

splitting = act["action_start"].str.split(expand=True)
act["week_days"] = splitting[0].apply(make_weekday)
date = splitting[0].str.split("-", expand=True)
date = date.rename(index=str, columns={0: "year", 1: "month", 2: "day"})
for column in date.columns:
    date[column] = date[column].astype(int)

act.reset_index(drop=True, inplace=True)
date.reset_index(drop=True, inplace=True)
splitting.reset_index(drop=True, inplace=True)

act = pd.concat([act, date], axis=1)
splitting[1] = splitting[1].apply(make_seconds)
act["cos_sec"] = splitting[1].apply(lambda x: round(make_cyclic(x, True, 86400), 4))
act["sin_sec"] = splitting[1].apply(lambda x: round(make_cyclic(x, False, 86400), 4))

act["sort_time"] = act["action_start"].apply(make_sorting_seconds)
act = act.drop(["action_start", "profile_id", "profile_type", "action_user_id"], axis=1)
del date
del splitting
act.head(10)
cis = pd.read_csv("../input/cis.csv")
msno.matrix(cis.sample(2000))
pd.Series({column: len(cis[column].unique()) for column in cis.drop(["material_id", "name", "code"], axis=1).columns}).plot("bar")
columns_to_encode = ["material_type", "subject", "education_level"]
fig, subplots = plt.subplots(nrows=3, ncols=1, figsize=(15, 45))
subplots_dict = {0: subplots[0], 1: subplots[1], 2: subplots[2]}

cis = cis.dropna()
for i in range(3):
    info = cis[columns_to_encode[i]].value_counts()
    info.plot("bar", ax=subplots_dict[i], title=columns_to_encode[i])
plt.show()
print(f'Unique ids: {len(cis["material_id"].unique())}\n'
      f'Unique names: {len(cis["name"].unique())}\n'
      f'Unique ids+names: {len((cis["material_id"].astype(str) + " " + cis["name"]).unique())}\n'
      f'Real shape: {cis.shape[0]}')
mat = pd.read_csv("../input/data.csv")
msno.matrix(mat)
mat = mat[mat["deleted_at"].isnull()]
mat = mat.drop(["logical_type", "content_type", "book_type", "game_app_category", "game_app_version_id", "game_app_version", "publisher_id", "time_to_study", "deleted_at", "profile_id", "user_id", "updated_at", "accepted_at", "profile_type"], axis=1)
mat = mat.dropna()

splitting = mat["created_at"].str.split(expand=True)
date = splitting[0].str.split("-", expand=True)
date = date.rename(index=str, columns={0: "creation_year", 1: "creation_month", 2: "creation_day"})
for column in date.columns:
    date[column] = date[column].astype(int)
mat.reset_index(drop=True, inplace=True)
date.reset_index(drop=True, inplace=True)
# splitting.reset_index(drop=True, inplace=True)
mat = pd.concat([mat, date], axis=1).drop(["created_at"], axis=1)
del date
del splitting
mat.head(10)
intersected_columns = list(set(cis.columns.tolist()) & set(mat.columns.tolist()) - {"material_id"})
cis = pd.merge(cis, mat.drop(intersected_columns, axis=1), on="material_id").dropna()

import warnings
warnings.filterwarnings("ignore")

cis = cis[cis["code"].str.count(".") >= 3]
subjects = cis["subject"].unique()
edus = cis["education_level"]
info = {}
for subject in subjects:
    info[subject] = len(cis[cis["subject"] == subject]["code"].unique())
pd.Series(info).sort_values().plot("bar", figsize=(10, 10))
# cis["code"] = cis["code"].str.slice(0, 7)
before = cis.shape[0]
cis = cis.drop_duplicates(["code", "material_id", "subject", "name", "education_level"])
print("Before:", before)
print("After:", cis.shape[0])
pd.Series({5: 38138, 4: 212969, 3: 527662, 2: 832260}).plot("bar", title="Кол-во материалов Уровни обстракций")
com = pd.read_csv("../input/competences.csv")
msno.matrix(com)
com = com.rename(index=str, columns={"name.1": "subject", "name": "education_level"})
info = com["subject"].value_counts()
equal_numbers = [1, 2, 3, 4, 5]
segments = [(10, 20), (30, 40), (50, 60), 
            (100, 200), (300, 400), (500, 600), 
            (1000, 2000), (3000, 4000), (5000, 6000), 
            (10000, 20000), (30000, 40000), (50000, 60000),
           ]

visual_info = {}
for number in equal_numbers:
    visual_info[str(number)] = info.where(info == number).dropna().shape[0]
for start, end in segments:
    visual_info[f"{start}-{end}"] = info[(info < end) & (info > start)].dropna().shape[0]
pd.Series(visual_info).plot("bar", figsize=(20, 20))
import nltk
import pymorphy2
from gensim.models import Word2Vec
from typing import List


class Analyzer:
    def __init__(self):
        self.model = None
        self.morph = pymorphy2.MorphAnalyzer()

    def __normalization_word__(self, word: str):
        return self.morph.parse(word)[0].normal_form

    def normalization(self, text: str) -> List:
        return list(map(lambda x: self.__normalization_word__(x), nltk.word_tokenize(text)))

    def predict(self, sentence, sentence_2):
        return self.model.n_similarity(
            self.normalization(sentence),
            self.normalization(sentence_2)
        )

    def load_model(self, path='word2vec.model'):
        self.model = Word2Vec.load(path)

    def train_model(self, data, size=100, iter=100, window=50, min_count=1, workers=20, **kwargs):
        self.model = Word2Vec(data, size=size, iter=iter, window=window, min_count=min_count, workers=workers, **kwargs)

    def load_data_from_txt(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            data = file.read()
        return list(map(lambda x: self.normalization(x), data.split('\n')))

    def save_model(self, path='word2vec.model'):
        self.model.save(path)


analyzer = Analyzer()
analyzer.load_model("../input/word2vec.model")

print(analyzer.predict(
    'употребление гласных',
    'безударные морфемы'
))

print("Before:")
print("   cis:", len(cis["subject"].unique()), cis.shape[0])
print("   com:", len(com["subject"].unique()), com.shape[0])

subjects = pd.Series(list(set(cis["subject"].unique()) & set(com["subject"].unique()))).tolist()
cis = cis[cis["subject"].isin(subjects)]
com = com[com["subject"].isin(subjects)]


print(" After:")
print("   cis:", len(cis["subject"].unique()), cis.shape[0])
print("   com:", len(com["subject"].unique()), com.shape[0])
cis = cis.drop_duplicates(subset=["material_id", "material_type"])
teacher_dataset = act[act["action_profile_type"] == "TeacherProfile"].drop(["action_profile_type"], axis=1)
teacher_dataset = teacher_dataset.rename(index=str, columns={"action_profile_id": "teacher_id"})
cis = cis.drop(["average_rating"], axis=1)
teacher_dataset = pd.merge(teacher_dataset, cis, left_on=["material_id", "material_type"], right_on=["material_id", "material_type"])
# cis.to_csv("materials.csv", index=False)
import tqdm
import time
users = teacher_dataset["teacher_id"].unique()
hist = {}
for i in tqdm.tqdm(range(len(users))):
    start = time.time()
    mini_data = teacher_dataset[teacher_dataset["teacher_id"] == users[i]].sort_values(by="sort_time")
    mini_data2 = mini_data.drop(mini_data.index[0])
    mini_data2 = mini_data2.rename(index=str, columns={column: column + "_watching" for column in mini_data.columns})
    
    mini_data.reset_index(drop=True, inplace=True)
    mini_data2.reset_index(drop=True, inplace=True)
    mini_data = pd.concat([mini_data, mini_data2], axis=1)
    hist[users[i]] = mini_data
users_data = pd.concat(list(hist.values()))
# users_data = users_data[users_data["material_type_watching"].isna()]
# users_data = users_data.drop([column for column in users_data.columns if "_watching" in column], axis=1)
users_data.to_csv("users.csv")