#Importing libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score





%matplotlib inline



#Getting data



#I owe this to SRK's work here:

#https://www.kaggle.com/sudalairajkumar/two-sigma-financial-modeling/simple-exploration-notebook/notebook

with pd.HDFStore("../input/train.h5", "r") as train:

    df = train.get("train")





input_variables = [x for x in df.columns.values if x not in ['id','y','timestamp']]





df_id_vs_variable = df[['id']+input_variables]       #Removes 'y' and 'timestamp'

df_id_vs_variable = df_id_vs_variable.fillna(0)      #Replace na by 0



def makeBinary(x):

    if abs(x) > 0.00000:

        return 1

    else:

        return 0



df_id_vs_variable = df_id_vs_variable.groupby('id').agg('sum').applymap(makeBinary)





n_clust = 2



km = KMeans(n_clusters=n_clust, n_init=20).fit(df_id_vs_variable)

clust = km.predict(df_id_vs_variable)





#Init table of indexes

df_clust_index = {}

for i in range(0,n_clust):

    df_clust_index[i]=[]



#Fill the cluster index

for i in range(0,len(clust)):

    df_clust_index[clust[i]].append(i)



for i in range(0,n_clust):

    df_clust_index[i] = df_id_vs_variable.iloc[df_clust_index[i]].index.values







df_clust = []



for i in range(0,n_clust):

    df_clust.append(df.loc[df.id.isin(df_clust_index[i])])
non_null_0 = df_id_vs_variable.loc[clust==0].sum() / df_id_vs_variable.loc[clust==0].shape[0]

non_null_1 = df_id_vs_variable.loc[clust==1].sum() / df_id_vs_variable.loc[clust==1].shape[0]





df_non_null_comparison = pd.concat([non_null_0,non_null_1],axis=1)



bar_width = 1

index = np.arange(df_non_null_comparison.shape[0])



fig, ax = plt.subplots(figsize=(12,50))



rects1 = plt.barh(index ,  np.array(df_non_null_comparison[0]), bar_width/2,

                 color='b',

                 label='Cluster 0')



rects1 = plt.barh(index + bar_width/2,  np.array(df_non_null_comparison[1]), bar_width/2,

                 color='r',

                 label='Cluster 1')





#plt.figure(figsize=(20,50))

plt.legend()

plt.xlabel('Percentage of Null-Values')

plt.ylabel('Features')

plt.yticks(index + bar_width, df_non_null_comparison.index.values)

plt.tight_layout()

plt.show()
non_null_threshold = 0.95



col_0 = non_null_0.loc[non_null_0 > non_null_threshold].index.values

col_1 = non_null_1.loc[non_null_1 > non_null_threshold].index.values
col_0 = ['derived_2', 'fundamental_0', 'fundamental_2', 'fundamental_7', 'fundamental_8', 'fundamental_10', 'fundamental_11', 'fundamental_13', 'fundamental_14', 'fundamental_15', 'fundamental_16', 'fundamental_18', 'fundamental_19', 'fundamental_21', 'fundamental_23', 'fundamental_29', 'fundamental_30', 'fundamental_33', 'fundamental_35', 'fundamental_36', 'fundamental_37', 'fundamental_39', 'fundamental_41', 'fundamental_42', 'fundamental_43', 'fundamental_44', 'fundamental_45', 'fundamental_46', 'fundamental_48', 'fundamental_50', 'fundamental_53', 'fundamental_54', 'fundamental_55', 'fundamental_56', 'fundamental_59', 'fundamental_60', 'fundamental_62', 'technical_1', 'technical_2', 'technical_3', 'technical_6', 'technical_7', 'technical_11', 'technical_13', 'technical_17', 'technical_19', 'technical_20', 'technical_21', 'technical_22', 'technical_24', 'technical_27', 'technical_30', 'technical_33', 'technical_35', 'technical_36', 'technical_40', 'technical_41']

col_1 = ['technical_1', 'technical_2', 'technical_3', 'technical_5', 'technical_6',

     'technical_7', 'technical_11', 'technical_13', 'technical_14', 'technical_17',

     'technical_19', 'technical_20', 'technical_21', 'technical_22', 'technical_24',

     'technical_27', 'technical_30', 'technical_33', 'technical_34', 'technical_35',

     'technical_36', 'technical_40', 'technical_41', 'technical_43']



common_cols = [x for x in col_0 if x in col_1]



print("Cluster 0 has {0} columns".format(len(col_0)))

print("Cluster 1 has {0} columns".format(len(col_1)))



print("Number of features present in both clusters: {0}".format(len(common_cols)))

print("Number of features that are only present in cluster 0: {0}".format(len([x for x in col_0 if x not in col_1])))

print("The following features are only present in cluster 1: {0}".format(len([x for x in col_1 if x not in col_0])))
from scipy import stats



y_0 = df_clust[0].y.dropna().values

y_1 = df_clust[1].y.dropna().values 



print("{:01.3f}".format(stats.ks_2samp(y_0, y_1)[1]))


for i in range(0,n_clust):

    n, bins, patches = plt.hist(df_clust[i].y.dropna().values, 50, normed=1, facecolor='green', alpha=0.75)

    plt.xlabel('y Value')

    plt.ylabel('Occurence')

    plt.title(r'Distribution of y Value for Cluster '+str(i))

    plt.show()

    print("Mean value: {:.3e}".format(df_clust[i].y.dropna().mean()))

    print("Standard deviation: {:.3e}".format(df_clust[i].y.dropna().std()))

    print("Median value: {:.3e}".format(df_clust[i].y.dropna().median()))

    print("Skew: {:.3e}".format(df_clust[i].y.dropna().skew()))

    print("Kurtosis: {:.3e}".format(df_clust[i].y.dropna().kurtosis()))
p_values_y = map(lambda x: stats.ks_2samp(df_clust[0][x].dropna().values, df_clust[1][x].dropna().values)[1],common_cols)



def isrejected(pval,th = 0.05):

    if pval < th:

        return "rejected"

    else:

        return "not rejected"

for pv in list(p_values_y):

    print("{:.3e}: {}".format(pv,isrejected(pv)))
df_clust[1].shape
#This code's purpose is to make sure that the cluster 0 in my previous analysis is the same here.



if(non_null_0.loc[non_null_0 > non_null_threshold].index.isin(col_0).sum() < 24):

    temp_df = df_clust[0]

    df_clust[0] = df_clust[1]

    df_clust[1] = temp_df
cl=0

cols=[col_0,col_1]

for cl in [0,1]:

    remaining_cols = [x for x in cols[cl] if x not in cols[1-cl]]



    df_corr = df_clust[cl][common_cols + remaining_cols]

    df_corr = df_corr.corr()



    df_corr = df_corr[common_cols].loc[df_corr.index.isin(remaining_cols)]



    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    f, ax = plt.subplots(figsize=(11, 9))

    sns.heatmap(df_corr, cmap=cmap, vmax=1,

                square=True, xticklabels=True, yticklabels=True,

                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)