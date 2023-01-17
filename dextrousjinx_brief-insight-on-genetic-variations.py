import os

import math

import numpy as np

import pandas as pd

import seaborn as sns



%matplotlib inline

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
for f in os.listdir('../input'):

    size_bytes = round(os.path.getsize('../input/' + f)/ 1000, 2)

    size_name = ["KB", "MB"]

    i = int(math.floor(math.log(size_bytes, 1024)))

    p = math.pow(1024, i)

    s = round(size_bytes / p, 2)

    print(f.ljust(20) + str(s).ljust(7) + size_name[i])
train_variants_df = pd.read_csv("../input/training_variants")

test_variants_df = pd.read_csv("../input/test_variants")

train_text_df = pd.read_csv("../input/training_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])

test_text_df = pd.read_csv("../input/test_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])

print("Train Variant".ljust(15), train_variants_df.shape)

print("Train Text".ljust(15), train_text_df.shape)

print("Test Variant".ljust(15), test_variants_df.shape)

print("Test Text".ljust(15), test_text_df.shape)
train_variants_df.head()
print("For training data, there are a total of", len(train_variants_df.ID.unique()), "IDs,", end='')

print(len(train_variants_df.Gene.unique()), "unique genes,", end='')

print(len(train_variants_df.Variation.unique()), "unique variations and ", end='')

print(len(train_variants_df.Class.unique()),  "classes")
plt.figure(figsize=(12,8))

sns.countplot(x="Class", data=train_variants_df, palette="Blues_d")

plt.ylabel('Frequency', fontsize=14)

plt.xlabel('Class', fontsize=14)

plt.title("Distribution of genetic mutation classes", fontsize=18)

plt.show()
gene_group = train_variants_df.groupby("Gene")['Gene'].count()

minimal_occ_genes = gene_group.sort_values(ascending=True)[:10]

print("Genes with maximal occurences\n", gene_group.sort_values(ascending=False)[:10])

print("\nGenes with minimal occurences\n", minimal_occ_genes)
fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(15,15))



for i in range(3):

    for j in range(3):

        gene_count_grp = train_variants_df[train_variants_df["Class"]==((i*3+j)+1)].groupby('Gene')["ID"].count().reset_index()

        sorted_gene_group = gene_count_grp.sort_values('ID', ascending=False)

        sorted_gene_group_top_7 = sorted_gene_group[:7]

        sns.barplot(x="Gene", y="ID", data=sorted_gene_group_top_7, ax=axs[i][j])

train_text_df.head()
train_text_df.loc[:, 'Text_count']  = train_text_df["Text"].apply(lambda x: len(x.split()))

train_text_df.head()
train_full = train_variants_df.merge(train_text_df, how="inner", left_on="ID", right_on="ID")

train_full[train_full["Class"]==1].head()
count_grp = train_full.groupby('Class')["Text_count"]

count_grp.describe()
train_full[train_full["Text_count"]==1.0]
train_full[train_full["Text_count"]<500.0]
plt.figure(figsize=(12,8))

gene_count_grp = train_full.groupby('Gene')["Text_count"].sum().reset_index()

sns.violinplot(x="Class", y="Text_count", data=train_full, inner=None)

sns.swarmplot(x="Class", y="Text_count", data=train_full, color="w", alpha=.5);

plt.ylabel('Text Count', fontsize=14)

plt.xlabel('Class', fontsize=14)

plt.title("Text length distribution", fontsize=18)

plt.show()
fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(15,15))



for i in range(3):

    for j in range(3):

        gene_count_grp = train_full[train_full["Class"]==((i*3+j)+1)].groupby('Gene')["Text_count"].mean().reset_index()

        sorted_gene_group = gene_count_grp.sort_values('Text_count', ascending=False)

        sorted_gene_group_top_7 = sorted_gene_group[:7]

        sns.barplot(x="Gene", y="Text_count", data=sorted_gene_group_top_7, ax=axs[i][j])
def top_tfidf_feats(row, features, top_n=10):

    topn_ids = np.argsort(row)[::-1][:top_n]

    top_feats = [(features[i], row[i]) for i in topn_ids]

    df = pd.DataFrame(top_feats)

    df.columns = ['feature', 'tfidf']

    return df



def top_feats_in_doc(Xtr, features, row_id, top_n=10):

    row = np.squeeze(Xtr[row_id].toarray())

    return top_tfidf_feats(row, features, top_n)



def top_mean_feats(Xtr, features, grp_ids=None, min_tfidf=0.1, top_n=10):

    if grp_ids:

        D = Xtr[grp_ids].toarray()

    else:

        D = Xtr.toarray()



    D[D < min_tfidf] = 0

    tfidf_means = np.mean(D, axis=0)

    return top_tfidf_feats(tfidf_means, features, top_n)



def top_feats_by_class(Xtr, y, features, min_tfidf=0.1, top_n=10):

    dfs = []

    labels = np.unique(y)

    for label in labels:

        ids = np.where(y==label)

        feats_df = top_mean_feats(Xtr, features, ids, min_tfidf=min_tfidf, top_n=top_n)

        feats_df.label = label

        dfs.append(feats_df)

    return dfs



def plot_tfidf_classfeats_h(dfs):

    fig = plt.figure(figsize=(12, 100), facecolor="w")

    x = np.arange(len(dfs[0]))

    for i, df in enumerate(dfs):

        #z = int(str(int(i/3)+1) + str((i%3)+1))

        ax = fig.add_subplot(9, 1, i+1)

        ax.spines["top"].set_visible(False)

        ax.spines["right"].set_visible(False)

        ax.set_frame_on(False)

        ax.get_xaxis().tick_bottom()

        ax.get_yaxis().tick_left()

        ax.set_xlabel("Mean Tf-Idf Score", labelpad=16, fontsize=16)

        ax.set_ylabel("Gene", labelpad=16, fontsize=16)

        ax.set_title("Class = " + str(df.label), fontsize=18)

        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))

        ax.barh(x, df.tfidf, align='center')

        ax.set_yticks(x)

        ax.set_ylim([-1, x[-1]+1])

        yticks = ax.set_yticklabels(df.feature)

        plt.subplots_adjust(bottom=0.09, right=0.97, left=0.15, top=0.95, wspace=0.52)

    plt.show()
tfidf = TfidfVectorizer(

	min_df=5, max_features=16000, strip_accents='unicode',lowercase =True,

	analyzer='word', token_pattern=r'\w+', use_idf=True, 

	smooth_idf=True, sublinear_tf=True, stop_words = 'english').fit(train_full["Text"])



Xtr = tfidf.fit_transform(train_full["Text"])

y = train_full["Class"]

features = tfidf.get_feature_names()

top_dfs = top_feats_by_class(Xtr, y, features)
plot_tfidf_classfeats_h(top_dfs)