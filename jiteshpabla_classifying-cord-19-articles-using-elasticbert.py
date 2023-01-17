import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
dfm = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
dfm
#fill the blanks with a placehonder character
dfm = dfm.fillna("x")
df2 = dfm[['title', 'abstract']]
#make sure both coulms are strings
df2.title = df2.title.astype(str)
df2.abstract = df2.abstract.astype(str)
df2
import csv
df2.to_csv('metadata_out.csv', index=False, quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
DATA_DIR = "/kaggle/input/cord19-elasticbert-query-results/"
vdf = pd.read_csv(DATA_DIR+"BERT_vaccines.csv")
tdf = pd.read_csv(DATA_DIR+"BERT_therapeutics.csv")
vdf2 = pd.read_csv(DATA_DIR+"BERT_coronavirus vaccine.csv")
tdf2 = pd.read_csv(DATA_DIR+"BERT_coronavirus therapeutics.csv")
def cleanup(df):
  df[['abstract','title']] = df._source.str.split("'title':",expand=True)
  df["title"] = df.title.str[2:-2]
  df["abstract"] = df.abstract.str[14:-3]
cleanup(vdf)
cleanup(vdf2)
cleanup(tdf)
cleanup(tdf2)
vdf
metadf_title= dfm[["title"]]
#metadf_title = metadf_title.drop_duplicates()
metadf_title

merged_vdf = vdf.merge(vdf2, on=['title'], 
                   how='inner', indicator=True, suffixes=('', '_y'))
merged_vdf["score"] = merged_vdf[["_score", "_score_y"]].values.max(1)
merged_vdf.drop(list(merged_vdf.filter(regex='_y$')), axis=1, inplace=True)
merged_vdf = merged_vdf[["title", "score"]]
merged_vdf.drop_duplicates(inplace=True)
merged_vdf
vdf_concat = pd.concat([vdf, vdf2])
#vdf_concat = vdf_concat[["title"]]
vdf_concat.drop_duplicates(subset=["title"], inplace=True)
vdf_concat["score"] = vdf_concat["_score"]
vdf_concat = vdf_concat[["title", "score"]]
vdf_concat
vdf_concat2 = vdf_concat.merge(merged_vdf, on =["title"], how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only']
vdf_concat2["score"] = vdf_concat2["score_x"]
vdf_concat2 = vdf_concat2[["title", "score"]]
vdf_concat2
vdf_final = pd.concat([vdf_concat2, merged_vdf])
vdf_final.drop_duplicates(subset=["title"],inplace=True)
vdf_final
merged_tdf = tdf.merge(tdf2, on=['title'], 
                   how='inner', indicator=True, suffixes=('', '_y'))
merged_tdf["score"] = merged_tdf[["_score", "_score_y"]].values.max(1)
merged_tdf.drop(list(merged_tdf.filter(regex='_y$')), axis=1, inplace=True)
merged_tdf = merged_tdf[["title", "score"]]
merged_tdf.drop_duplicates(inplace=True)
merged_tdf
tdf_concat = pd.concat([tdf, tdf2])
#tdf_concat = tdf_concat[["title"]]
tdf_concat.drop_duplicates(subset=["title"], inplace=True)
tdf_concat["score"] = tdf_concat["_score"]
tdf_concat = tdf_concat[["title", "score"]]
tdf_concat
tdf_concat2 = tdf_concat.merge(merged_tdf, on =["title"], how = 'outer' ,indicator=True).loc[lambda x : x['_merge']=='left_only']
tdf_concat2["score"] = tdf_concat2["score_x"]
tdf_concat2 = tdf_concat2[["title", "score"]]
tdf_concat2
tdf_final = pd.concat([tdf_concat2, merged_tdf])
tdf_final.drop_duplicates(subset=["title"],inplace=True)
tdf_final
metadf_final = metadf_title
metadf_final["class"] = 0
metadf_final
#the common ones between VACNNIES AND THERPEUTICS
merged_all = tdf_final.merge(vdf_final, on=['title'], 
                   how='inner', indicator=True, suffixes=('', '_y'))
merged_all.drop(list(merged_all.filter(regex='_y$')), axis=1, inplace=True)
#merged_all.drop_duplicates(inplace=True)
merged_all
x = y = 0
for i, row in metadf_final.iterrows():
  if (row["title"] in vdf_final.values) and (row["title"] in tdf_final.values):
    #vdf_final.loc[df['title'] == row["title"]]
    vi = vdf_final.index[vdf_final['title'] == row["title"]].tolist()[0]
    ti = tdf_final.index[tdf_final['title'] == row["title"]].tolist()[0]
    #print(vdf_final.iloc[vi].score, tdf_final.iloc[ti].score)
    if vdf_final.iloc[vi].score >= tdf_final.iloc[ti].score:
      metadf_final.loc[i,'class'] = 1
      x = x+1
    else:
      metadf_final.loc[i,'class'] = 2
      y = y+1
  elif row["title"] in vdf_final.values:
    metadf_final.loc[i,'class'] = 1
  elif row["title"] in tdf_final.values:
    metadf_final.loc[i,'class'] = 2
print("common articles split into virus and therapeutics respectively")
print(x,y)
metadf_final
#rename column to support query syntax
metadf_final = metadf_final.rename(columns={"class": "classif"}, errors="raise")
metadf_final.query("`classif` == 1")
metadf_final.query("`classif` == 2")