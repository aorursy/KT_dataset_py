# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



from pathlib import Path

import os

DATA_DIR=Path('/kaggle/input')

for dirname, _, filenames in os.walk(DATA_DIR):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
TRAIN_PATH = DATA_DIR / "3c-shared-task-influence" / "train.csv"

TEST_PATH = DATA_DIR / "3c-shared-task-influence" / "test.csv"
! head /kaggle/input/3c-shared-task-influence/sample_submission.csv
import pandas as pd
df_train = pd.read_csv(str(TRAIN_PATH).replace("influence", "purpose"))

df_train.columns
df_train.head()
df_train = pd.read_csv(TRAIN_PATH).merge(

    pd.read_csv(str(TRAIN_PATH).replace("influence", "purpose"))[["unique_id", "citation_class_label"]],

    on="unique_id"

)

df_train.head()
df_test = pd.read_csv(TEST_PATH).merge(

    pd.read_csv(str(TEST_PATH).replace("influence", "purpose"))[["unique_id"]],

    on="unique_id"

)

df_test.head()
df = pd.concat([

    df_train.assign(split="train"),

    df_test.assign(split="test"),

], axis=0, sort=False).reset_index(drop=True)

df.head()
df.split.value_counts()
df.pivot_table(index="citation_influence_label", columns="split", values="unique_id", aggfunc=len).sort_values("train", ascending=False)
df.pivot_table(index="citation_class_label", columns="split", values="unique_id", aggfunc=len).sort_values("train", ascending=False)
df.pivot_table(index="citing_author", columns="split", values="unique_id", aggfunc=len).sort_values("train", ascending=False)
df.pivot_table(index="cited_author", columns="split", values="unique_id", aggfunc=len).sort_values("train", ascending=False)
df.core_id.value_counts()
import requests

import json

import time
CORE_REQUEST_URL="https://core.ac.uk:443/api-v2/articles/get?metadata=true&fulltext=true&citations=true&similar=false&duplicate=false&urls=false&faithfulMetadata=false&apiKey=gHQKEJ9bMBe7FntNCXqjrASuz0xc5dTU"

def get_paper_data(core_ids, batch_size=10):

    core_ids = list(core_ids)

    for i in range(0, len(core_ids), batch_size):

        batch = core_ids[i:i+batch_size]

        resp = requests.post(CORE_REQUEST_URL, json=batch)

        batch_resp_json = resp.json()

        yield batch_resp_json

        print(f"Found {len(batch_resp_json)} responses. Sleeping for 2 seconds.")

        time.sleep(2)
%%time

CORE_DATA_PATH=Path("./core_data.jsonl")

CORE_DATA_URL="https://gist.githubusercontent.com/napsternxg/f514e8ac039998e129ad187956b7eb9f/raw/00e34089890fd158f0867e1b3436257a511d040f/core_data.jsonl"

COLLECT_CORE_DATA=False

if not CORE_DATA_PATH.exists():

    print(f"File {CORE_DATA_PATH} does not exists.")

    if COLLECT_CORE_DATA:

        print(f"Collecting data via the CORE API")

        core_data = sum(get_paper_data(df.core_id.unique().tolist(), batch_size=10), [])

    else:

        print(f"Fetching {CORE_DATA_URL}.")

        resp = requests.get(CORE_DATA_URL).text

        core_data = []

        for line in resp.splitlines():

            d = json.loads(line)

            core_data.append(d)

        print(f"Found {len(core_data)} records.")

    print(f"Found {len(core_data)} records. Writing to {CORE_DATA_PATH}")

    with open(CORE_DATA_PATH, "w+") as fp:

        for d in core_data:

            print(json.dumps(d), file=fp)

else:

    print(f"File {CORE_DATA_PATH} exists. Loading.")

    with open(CORE_DATA_PATH) as fp:

        core_data = []

        for line in fp:

            d = json.loads(line)

            core_data.append(d)

    print(f"Found {len(core_data)} records.")
core_data[0]["data"]["year"]
found_keys = [

    int(d["data"]["id"])

    for d in core_data

    if d["status"] == "OK"

]

len(found_keys)
not_found_ids = [

    k

    for k in df.core_id.unique()

    if k not in set(found_keys)

]

len(not_found_ids)
df.assign(

    found_core_data=lambda x: ~x.core_id.isin(set(not_found_ids))

).head()
df_found = pd.DataFrame([

    d["data"]

    for d in core_data

    if d["status"] == "OK"

]).assign(core_id=lambda x: x["id"].astype(int))

df_found.head()
df_found[df_found.citations.apply(lambda x: len(x) != 0)].citations.shape
df_found[df_found.topics.apply(lambda x: len(x) != 0)].topics.shape
df_found[~df_found.description.isnull()].description.head().values
df_found.year.value_counts()
df_found.columns
import numpy as np

from sklearn.compose import ColumnTransformer

from sklearn.feature_extraction.text import TfidfVectorizer
ct = ColumnTransformer([

    ("citing_tfidf", TfidfVectorizer(), "citing_title"),

    ("cited_tfidf", TfidfVectorizer(), "cited_title"),

    ("citation_context_tfidf", TfidfVectorizer(),"citation_context"),

])

ct.fit(df)

df_features = ct.transform(df)

df_features.shape
df_features
df_features[[0, 1, 5]]
TASKS={

    "purpose": "citation_class_label",

    "influence": "citation_influence_label"

}



def generate_data(df, label, split="train"):

    split_idx = df[(df.split == split)].index.tolist()

    X = df_features[split_idx]

    y = df.iloc[split_idx][label]

    print(f"{split}: X={X.shape}, y={y.shape}")

    return X, y, split_idx



def submission_pipeline(model, df, df_features, task, model_key=None, to_dense=False):

    label = TASKS[task]

    X_train, y_train, train_idx = generate_data(df, label, split="train")

    X_test, y_test, test_idx = generate_data(df, label, split="test")

    print(f"Training model")

    if to_dense:

        X_train = X_train.toarray()

        X_test = X_test.toarray()

    model.fit(X_train, y_train.astype(int))

    y_test = model.predict(X_test)

    print("Output label dist")

    print(pd.Series(y_test).value_counts())

    submission_file=f"{model_key}_{task}_submission.csv"

    print(f"Writing submission file: {submission_file}")

    df.iloc[test_idx][["unique_id"]].assign(**{label: y_test}).to_csv(submission_file, index=False)

    return model
from sklearn.ensemble import GradientBoostingClassifier
for task in TASKS:

    model = GradientBoostingClassifier()

    submission_pipeline(model, df, df_features, task, model_key="gbt")
from sklearn.ensemble import RandomForestClassifier
for task in TASKS:

    model = RandomForestClassifier(n_jobs=-1)

    submission_pipeline(model, df, df_features, task, model_key="rf")
import sklearn

sklearn.__version__
from sklearn.neural_network import MLPClassifier
for task in TASKS:

    model = MLPClassifier(hidden_layer_sizes=(256,256,128))

    submission_pipeline(model, df, df_features, task, model_key="mlp", to_dense=False)