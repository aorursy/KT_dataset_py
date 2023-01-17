import pandas as pd

import numpy as np

import random

import matplotlib.pyplot as plt

import seaborn as sns
chunk = 100000

df = pd.read_csv('../input/riiid-test-answer-prediction/train.csv', chunksize=chunk, iterator=True)

train = pd.concat(df, ignore_index=True)
questions = pd.read_csv("../input/riiid-test-answer-prediction/questions.csv")

lectures = pd.read_csv("../input/riiid-test-answer-prediction/lectures.csv")
print(f"Train Shape: {train.shape}\nQuestions Shape: {questions.shape}\nLectures Shape: {lectures.shape}")
random.seed(11)

samp = random.sample(range(len(train)),int(0.1*len(train)))

print(f"No of Samples: {len(samp)}")
train_samp = train.iloc[samp,:].copy()

del(train)

print(f"Shape of Train Sample: {train_samp.shape}")
train_samp.head()
train_samp.describe().T
train_samp.dtypes
train_samp.nunique()
np.round(train_samp.isnull().mean()*100,2)
categorical_features = ["content_type_id","user_answer","answered_correctly","prior_question_had_explanation"]
for col in categorical_features:

    print(f"{col} | dtype: {train_samp[col].dtypes} | nunique: {train_samp[col].nunique()}\n{train_samp[col].value_counts()}\n\n")
for col in categorical_features:

    sns.countplot(train_samp[col])

    plt.title(col)

    plt.show()
prior_correct = train_samp.loc[train_samp["answered_correctly"]>=0,["prior_question_had_explanation","answered_correctly"]].copy()

ct = pd.crosstab(index=prior_correct["prior_question_had_explanation"],columns=prior_correct["answered_correctly"],normalize="index")

ct.plot(kind="bar",stacked=True,figsize=(10,5))

plt.ylabel("% of students");