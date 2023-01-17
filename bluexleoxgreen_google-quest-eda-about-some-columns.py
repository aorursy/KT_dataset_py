import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_rows', 100)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

%matplotlib inline

import matplotlib.pyplot as plt

from scipy.stats import spearmanr



df_train = pd.read_csv("../input/google-quest-challenge/train.csv")

df_test  = pd.read_csv("../input/google-quest-challenge/test.csv")
qtp = df_train[df_train["question_type_spelling"] > 0][["question_body", "category", "question_type_spelling", "url"]]

print(f"{len(qtp)} / {len(df_train)} QAs")

qtp
indexs = []

for i in range(len(df_train)):

    if df_train.iloc[i, 8] in qtp["url"].values:

        indexs.append(i)

df_train.iloc[indexs].sort_values("url")[["qa_id", "question_body", "answer", "question_type_spelling"]]
def count_spelling_feature(text):

    symbols = ["ʊ", "ə", "ɹ", "ɪ", "ʒ", "ɑ", "ʌ", "ɔ", "æ", "ː", "ɜ",  "adjective", "pronounce"]

    count = 0

    for s in symbols:

        count += text.count(s)

    return count



df_train[df_train["question_body"].apply(lambda x: count_spelling_feature(x)) > 0][["url", "question_body", "answer","category","question_type_spelling"]]
df_test[df_test["question_body"].apply(lambda x: count_spelling_feature(x)) > 0][["url", "question_body", "answer","category"]]
cols = ["answer_plausible", "question_not_really_a_question", "question_type_spelling", "answer_relevance"]

for col in cols: 

    df_train[col].hist()

    plt.title(col)

    plt.show();
true_values = np.append(np.zeros(500), 1)



pred_values = np.append(1, np.zeros(500))

sp = spearmanr(true_values, pred_values).correlation

score = (0.4 * 29 + sp) / 30

print(f"correlation score of 1 column: {sp}, LB score: {score}")
pred_values = np.append(np.zeros(500), 1)

sp = spearmanr(true_values, pred_values).correlation

score = (0.4 * 29 + sp) / 30

print(f"correlation score of 1 column: {sp}, LB score: {score}")
# If the ranking is correct but there is noise in the prediction

pred_values = np.append(np.zeros(500), 1)

pred_values = pred_values + np.random.normal(0, 1e-7, pred_values.shape[0])

sp = spearmanr(true_values, pred_values).correlation

score = (0.4 * 29 + sp) / 30

print(f"correlation score of 1 column: {sp}, LB score: {score}")
# If 80% of the predicted value is unified at 0

pred_values = np.append(np.append(np.zeros(100) + np.random.normal(0, 1e-7, 100), np.zeros(400)), 1) 

sp = spearmanr(true_values, pred_values).correlation

score = (0.4 * 29 + sp) / 30

print(f"correlation score of 1 column: {sp}, LB score: {score}")