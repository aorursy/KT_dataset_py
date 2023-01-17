!pip install researchpy
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas.api.types import CategoricalDtype
import matplotlib.pyplot as plt
import researchpy as rp
from scipy.stats import kruskal
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
alpha = 0.05
df = pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")
df.head()
df.info()
df["avg_score"] = (df["math score"] + df["reading score"] + df["writing score"]) / 3
df.head()
paredu_mappings = {
    "some high school": 1,
    "high school": 2,
    "associate's degree": 3,
    "some college": 4,
    "bachelor's degree": 5,
    "master's degree": 6
}
test_prep_mappings = {
    "none": 0,
    "completed": 1
}
lunch_mappings = {
    "free/reduced": 0,
    "standard": 1
}
cat_type = CategoricalDtype(categories=paredu_mappings.keys(), ordered=True)
df["parental level of education"] = df["parental level of education"].astype(cat_type)

cat_type = CategoricalDtype(categories=test_prep_mappings.keys(), ordered=True)
df["test preparation course"] = df["test preparation course"].astype(cat_type)

cat_type = CategoricalDtype(categories=lunch_mappings.keys(), ordered=True)
df["lunch"] = df["lunch"].astype(cat_type)

df.head()
df.info()
df.corr()
fig, axes = plt.subplots(2, 2, figsize=(20, 10))

df["avg_score"].hist(ax=axes[0,0])
axes[0,0].set_title('avg score')
df["math score"].hist(ax=axes[0,1])
axes[0,1].set_title('math score')
df["reading score"].hist(ax=axes[1,0])
axes[1,0].set_title('reading score')
df["writing score"].hist(ax=axes[1,1])
axes[1,1].set_title('writing score')
fig.suptitle('Scores histograms', fontsize=12)

# Under the null hypothesis, the two distributions are identical, F(x)=G(x)
from scipy.stats import shapiro

for score_name in ["avg_score", "math score", "reading score", "writing score"]:
    stat, p_value = shapiro(df[score_name])
    #print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p_value > alpha:
        print(f'{score_name} looks Gaussian (fail to reject H0)')
    else:
        print(f'{score_name} does NOT look Gaussian (reject H0)')


df["parental level of education ord"] = df["parental level of education"].apply(lambda x: paredu_mappings[x]).astype(int)
df["test preparation course ord"] = df["test preparation course"].apply(lambda x: test_prep_mappings[x]).astype(int)
df["lunch ord"] = df["lunch"].apply(lambda x: lunch_mappings[x]).astype(int)

# Gender is not really ordinal but let's try it anyway
df["gender ord"] = df["gender"].apply(lambda x: 1 if x=='female' else 0)

df.info()
sns.heatmap(df.corr(), annot=True)
df.groupby("parental level of education").count().iloc[:,1].plot(kind="bar")
df.boxplot(column="avg_score", by="parental level of education", figsize=(20,10))
df.groupby(["parental level of education"])["avg_score", "math score", "reading score", "writing score"].mean().plot.bar()
plt.show()
df.hist(column="avg_score", by="parental level of education", figsize=(20,10))
rp.summary_cont(df.groupby("parental level of education")['avg_score'])
list(df["parental level of education"].unique())
# Get scores for each group
edu_groups = list(df["parental level of education"].unique())
edu_group_scores = [df[df["parental level of education"]==g]["avg_score"].values for g in edu_groups]

stat, p = kruskal(*edu_group_scores)
print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret
if p > alpha:
    print('Group have same distributions of scores (fail to reject H0)')
else:
    print('Parental education groups have different distributions of scores (reject H0) ')
df.groupby("race/ethnicity").count().iloc[:,1].plot(kind="bar")
rp.summary_cont(df.groupby("race/ethnicity")['avg_score'])
df.groupby(["race/ethnicity"])["avg_score", "math score", "reading score", "writing score"].mean().plot.bar()
plt.show()
df.boxplot(column="avg_score", by="race/ethnicity", figsize=(20,10))
# Get scores for each group
eth_groups = list(df["race/ethnicity"].unique())
eth_group_scores = [df[df["race/ethnicity"]==g]["avg_score"].values for g in eth_groups]

# Test
stat, p = kruskal(*eth_group_scores)
print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret
if p > alpha:
    print('race/ethnicity groups have same distributions of scores (fail to reject H0)')
else:
    print('race/ethnicity groups have different distributions of scores (reject H0) ')
df.groupby("gender").count().iloc[:,1].plot(kind="bar")
df.groupby(["gender"])["avg_score", "math score", "reading score", "writing score"].mean().plot.bar()
plt.show()
df.hist(column="avg_score", by="gender", figsize=(20,10))
df.boxplot(column="avg_score", by="gender", figsize=(20,10))
# Get scores for each group
gender_groups = list(df["gender"].unique())
gender_group_scores = [df[df["gender"]==g]["avg_score"].values for g in gender_groups]

# Test
stat, p = kruskal(*gender_group_scores)
print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret
if p > alpha:
    print('gender groups have same distributions of scores (fail to reject H0)')
else:
    print('gender groups have different distributions of scores (reject H0) ')
df.groupby(["test preparation course"])["avg_score", "math score", "reading score", "writing score"].mean().plot.bar()
plt.show()
# Get scores for each group
test_prep_groups = list(df["test preparation course"].unique())
test_prep_group_scores = [df[df["test preparation course"]==g]["avg_score"].values for g in test_prep_groups]

# Test
stat, p = kruskal(*test_prep_group_scores)
print('Statistics=%.3f, p=%.3f' % (stat, p))

# interpret
if p > alpha:
    print('test_prep_group_scores have same distributions of scores (fail to reject H0)')
else:
    print('test_prep_group_scores have different distributions of scores (reject H0) ')
