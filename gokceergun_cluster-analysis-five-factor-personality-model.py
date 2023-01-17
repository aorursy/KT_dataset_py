import pandas as pd

import matplotlib.pyplot as plt
# reading the dataset

df = pd.read_csv('../input/big-five-personality-test/IPIP-FFM-data-8Nov2018/data-final.csv', sep='\t')

df.head()
df.IPC.value_counts()
# keep the records where the IPC column equals to 1.

df = df.loc[df["IPC"]==1]
# to check whether it keeps correct number of records. 

df.shape
df.drop(df.columns[50:], axis=1, inplace=True)

# alternatively:

# df = df.iloc[:, 0:50]

df.head()
print(df.isnull().any().sum())

df.isnull().sum().sort_values(ascending = False)

# deleting the missing values

df.dropna(inplace = True)

print(df.shape)

print(df.isnull().sum())  # to check if we delete all cases with missing values.
# to check whether the reverse items re-coded correctly - 1

df[["EXT2","EXT4", "EST2", "AGR1", "CSN8","OPN6"]].head()

#re-encoding reverse items

df.EXT2 = 6 - df.EXT2.values

df.EXT4 = 6 - df.EXT4.values

df.EXT6 = 6 - df.EXT6.values

df.EXT8 = 6 - df.EXT8.values

df.EXT10 = 6 - df.EXT10.values

df.EST2 = 6 - df.EST2.values

df.EST4 = 6 - df.EST4.values

df.AGR1 = 6 - df.AGR1.values

df.AGR3 = 6 - df.AGR3.values

df.AGR5 = 6 - df.AGR5.values

df.AGR7 = 6 - df.AGR7.values

df.CSN2 = 6 - df.CSN2.values

df.CSN4 = 6 - df.CSN4.values

df.CSN6 = 6 - df.CSN6.values

df.CSN8 = 6 - df.CSN8.values

df.OPN2 = 6 - df.OPN2.values

df.OPN4 = 6 - df.OPN4.values

df.OPN6 = 6 - df.OPN6.values



# alternatively : 

#df['EXT2'] = df['EXT2'].map({1:5, 2:4, 3:3, 4:2, 5:1})
# to check whether the reverse items re-coded correctly - 2

df[["EXT2","EXT4", "EST2", "AGR1", "CSN8","OPN6"]].head()
df_sample = df[0:5000]



# Visualize the elbow

from sklearn.cluster import KMeans

from yellowbrick.cluster import KElbowVisualizer



kmeans = KMeans()

visualizer = KElbowVisualizer(kmeans, k=(2,15))

visualizer.fit(df_sample)

visualizer.poof()
# Set up k-means

k_means = KMeans(n_clusters = 5)



#define 5 clusters and fit the model

k_fit = k_means.fit(df)
# Predicting the Clusters

pd.options.display.max_columns = 10

predictions = k_fit.labels_

df['Clusters'] = predictions

print(df.head())

df["Clusters"].unique()
# calculating total scale score



df["extraversion"] = 0

df["neuroticism"] = 0

df["agreeableness"] = 0

df["conscientiousness"] = 0

df["openness"] = 0

df["extraversion"]= (df.EXT1 + df.EXT2 + df.EXT3 + df.EXT4 + df.EXT5 + df.EXT6 + df.EXT7 + df.EXT8 + df.EXT9 + df.EXT10)/10

df["neuroticism"] = (df.EST1 + df.EST2 + df.EST3 + df.EST4 + df.EST5 + df.EST6 + df.EST7 + df.EST8 + df.EST9 + df.EST10)/10

df["agreeableness"] = (df.AGR1 + df.AGR2 + df.AGR3 + df.AGR4 + df.AGR5 + df.AGR6 + df.AGR7 + df.AGR8 + df.AGR9 + df.AGR10)/10

df["conscientiousness"] = (df.CSN1 + df.CSN2 + df.CSN3 + df.CSN4 + df.CSN5 + df.CSN6 + df.CSN7 + df.CSN8 + df.CSN9 + df.CSN10)/10

df["openness"] = (df.OPN1 + df.OPN2 + df.OPN3 + df.OPN4 + df.OPN5 + df.OPN6 + df.OPN7 + df.OPN8 + df.OPN9 + df.OPN10)/10

df.head()
# summary statistics of the total scores

df[["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness"]].describe()

table = df.groupby('Clusters')["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness"].mean()

print(table)



table.plot(figsize=(14,9), kind="bar", colormap='Paired')
df_total_scores = df[["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness"]]

print(df_total_scores.head())

print(df_total_scores.mean(axis=0))

df_total_scores = df_total_scores.apply(lambda x: (x-x.mean())/x.std(), axis = 0)

print(round(df_total_scores.std()))

print(round(df_total_scores.mean(axis=0)))

print(df_total_scores.head())

df_total_scores["clusters"] = df["Clusters"]

table = df_total_scores.groupby('clusters')["extraversion", "neuroticism", "agreeableness", "conscientiousness", "openness"].mean()

print(table)



table.plot(figsize=(14,9), kind="bar", colormap='Paired')