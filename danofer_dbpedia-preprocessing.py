import pandas as pd

from sklearn.model_selection import train_test_split
df = pd.read_csv("../input/DBP_wiki_data.csv",usecols=['text', 'l1', 'l2', 'l3', 'wiki_name']).drop_duplicates(subset="text")



print(df.shape)

print(df.nunique())

df.head()
df.columns
df["word_count"] = df.text.str.split().str.len()

print(df["word_count"].quantile([0.02,0.98]))

df["word_count"].describe()
print(df["word_count"].quantile([0.01,0.03,0.98,0.99]))

df["word_count"].describe()
df.loc[df["word_count"]<5]
print("Old shape",df.shape[0])

df = df.loc[(df["word_count"]>10) & (df["word_count"]<500)]

print("New shape",df.shape[0])



df["word_count"].describe()
# no need to export word count col. 

# WE also drop the wiki page name

df.drop(["word_count","wiki_name"],axis=1,inplace=True)
df.l3.value_counts()
df.l2.value_counts()
print("orig",df.shape)

df_train,df_test = train_test_split(df, test_size=0.18, random_state=42,stratify = df["l3"])



print("Train size",df_train.shape, "\n Test size",df_test.shape)
## validation set split:



df_train,df_val = train_test_split(df_train, test_size=0.13, random_state=42,stratify = df_train["l3"])

print("Final Train size",df_train.shape, "\nValidation size",df_val.shape)
df_train.to_csv("DBPEDIA_train_v1.csv.gz",index=False,compression="gzip")

df_val.to_csv("DBPEDIA_val_v1.csv.gz",index=False,compression="gzip")

df_test.to_csv("DBPEDIA_test_v1.csv.gz",index=False,compression="gzip")