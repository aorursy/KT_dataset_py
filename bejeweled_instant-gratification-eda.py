# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns



sns.set_style("whitegrid")
random_state = 42

np.random.seed(random_state)

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

sample_sub = pd.read_csv("../input/sample_submission.csv")
df_train.shape, df_test.shape, sample_sub.shape
display(df_train.head())

display(df_test.head())

display(sample_sub.head())
features = [col for col in df_train.columns if col not in ["id", "target"]]
plt.figure(figsize=[16,9])

sns.heatmap(df_train[features].corr())
plt.figure(figsize=[16,9])

sns.heatmap(df_test[features].corr())
sns.countplot(df_train["target"])
plt.figure(figsize=[16,9])

df_train.boxplot()
plt.figure(figsize=[16,9])

df_test.boxplot()
df_train.drop_duplicates().shape
def plot_feature_scatter(df1, df2, features):

    i = 0

    plt.figure()

    fig, ax = plt.subplots(4,4,figsize=(14,14))



    for feature in features:

        i += 1

        plt.subplot(4,4,i)

        plt.scatter(df1[feature], df2[feature], marker='+')

        plt.xlabel(feature, fontsize=9)

    plt.show();
c = 0

for j in range(16, len(features)+1, 16):

    plot_feature_scatter(df_train.iloc[:len(df_test)], df_test, features[c:j])

    c += 16
df_train["wheezy-copper-turtle-magic"].min(), df_train["wheezy-copper-turtle-magic"].max()
df_train["wheezy-copper-turtle-magic"].nunique(), \

df_test["wheezy-copper-turtle-magic"].nunique()
plt.figure(figsize=[9,5])

plt.subplot(1,2,1)

df_train.groupby("wheezy-copper-turtle-magic").size().sort_values()[::-1].hist(bins=20)

plt.title("train")



plt.subplot(1,2,2)

df_test.groupby("wheezy-copper-turtle-magic").size().sort_values()[::-1].hist(bins=20)

plt.title("test")



plt.show()
def plot_feature_boxplot(df, features):

    i = 0

    plt.figure()

    fig, ax = plt.subplots(4,4,figsize=(14,14))



    for feature in features:

        i += 1

        plt.subplot(4,4,i)

        sns.boxplot(df[feature])

        plt.xlabel(feature, fontsize=9)

    plt.show();
c = 0

for j in range(16, len(features)+1, 16):

    plot_feature_boxplot(df_train, features[c:j])

    c += 16
def plot_feature_distplot(df, features):

    i = 0

    plt.figure()

    fig, ax = plt.subplots(4,4,figsize=(14,14))



    for feature in features:

        i += 1

        plt.subplot(4,4,i)

        sns.distplot(df[feature])

        plt.xlabel(feature, fontsize=9)

    plt.show();
c = 0

for j in range(16, len(features)+1, 16):

    plot_feature_distplot(df_train, features[c:j])

    c += 16
c = 0

for j in range(16, len(features)+1, 16):

    plot_feature_distplot(df_test, features[c:j])

    c += 16