# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_variants_df = pd.read_csv("../input/training_variants")

test_variants_df = pd.read_csv("../input/test_variants")

train_text_df = pd.read_csv("../input/training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

test_text_df = pd.read_csv("../input/test_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])

train_variants_df.head()
import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()
group_df = train_variants_df.Class.value_counts()

plt.figure(figsize = (12,8))

sns.barplot(group_df.index, group_df.values, alpha=0.8, color=color[1])

plt.ylabel('Frequency', fontsize = 12)

plt.xlabel('Class', fontsize = 12)

plt.title('Frequency of Classes', fontsize = 16)

plt.show()
group_Gene_df = train_variants_df.Gene.value_counts().reset_index()

group_Gene_df = group_Gene_df.head(50)

plt.figure(figsize = (12,8))

sns.barplot(group_Gene_df['index'], group_Gene_df.Gene, alpha=0.8, color=color[0])

plt.ylabel('Frequency', fontsize = 12)

plt.xlabel('Gene', fontsize = 12)

plt.title('Frequency of Genes', fontsize = 16)

plt.xticks(rotation='vertical')

plt.show()
group_Var_df = train_variants_df.Variation.value_counts().reset_index()

group_Var_df = group_Var_df.head(50)

plt.figure(figsize = (12,8))

sns.barplot(group_Var_df['index'], group_Var_df.Variation, alpha=0.8, color=color[0])

plt.ylabel('Frequency', fontsize = 12)

plt.xlabel('Variation', fontsize = 12)

plt.title('Frequency of Variation', fontsize = 16)

plt.xticks(rotation='vertical')

plt.show()
for i in range(10):

    group_df = train_variants_df[train_variants_df.Gene == group_Gene_df["index"][i]]

    group_df = group_df.groupby(["Variation"])["Class"].aggregate("mean").reset_index()

    print(group_df.Class.value_counts())

for i in range(4):

    group_df = train_variants_df[train_variants_df.Variation == group_Var_df["index"][i]]

    group_df = group_df.groupby(["Gene"])["Class"].aggregate("mean").reset_index()

    print(group_df.Class.value_counts())
#I want to change the Gene labels to 11 labels

group_df = train_variants_df



for i in range(len(group_df)):

    if group_df.loc[i]["Gene"] == group_Gene_df["index"][0]:

        group_df["Gene"][i] = group_Gene_df["index"][0]

    elif group_df.loc[i]["Gene"] == group_Gene_df["index"][1]:

        group_df["Gene"][i] = group_Gene_df["index"][1]

    elif group_df.loc[i]["Gene"] == group_Gene_df["index"][2]:

        group_df["Gene"][i] = group_Gene_df["index"][2]

    elif group_df.loc[i]["Gene"] == group_Gene_df["index"][3]:

        group_df["Gene"][i] = group_Gene_df["index"][3]

    elif group_df.loc[i]["Gene"] == group_Gene_df["index"][4]:

        group_df["Gene"][i] = group_Gene_df["index"][4]

    elif group_df.loc[i]["Gene"] == group_Gene_df["index"][5]:

        group_df["Gene"][i] = group_Gene_df["index"][5]

    elif group_df.loc[i]["Gene"] == group_Gene_df["index"][6]:

        group_df["Gene"][i] = group_Gene_df["index"][6]

    elif group_df.loc[i]["Gene"] == group_Gene_df["index"][7]:

        group_df["Gene"][i] = group_Gene_df["index"][7]

    elif group_df.loc[i]["Gene"] == group_Gene_df["index"][8]:

        group_df["Gene"][i] = group_Gene_df["index"][8]

    elif group_df.loc[i]["Gene"] == group_Gene_df["index"][9]:

        group_df["Gene"][i] = group_Gene_df["index"][9]

    else:

        group_df["Gene"][i] = "Others"
for i in range(len(group_df)):

    if group_df.loc[i]["Variation"] == group_Var_df["index"][0]:

        group_df["Variation"][i] = group_Var_df["index"][0]

    elif group_df.loc[i]["Variation"] == group_Var_df["index"][1]:

        group_df["Variation"][i] = group_Var_df["index"][1]

    elif group_df.loc[i]["Variation"] == group_Var_df["index"][2]:

        group_df["Variation"][i] = group_Var_df["index"][2]

    elif group_df.loc[i]["Variation"] == group_Var_df["index"][3]:

        group_df["Variation"][i] = group_Var_df["index"][3]

    elif group_df.loc[i]["Variation"] == group_Var_df["index"][4]:

        group_df["Variation"][i] = group_Var_df["index"][4]

    else:

        group_df["Variation"][i] = "Others"
group_df = group_df.groupby(["Gene", "Variation"])["Class"].aggregate("mean").reset_index()

group_df = group_df.pivot('Gene', 'Variation', 'Class')

plt.figure(figsize=(12,6))

sns.heatmap(group_df)

plt.title("Gene, Varition, and Class")

plt.show()