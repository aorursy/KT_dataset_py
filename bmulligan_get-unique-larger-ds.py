import pandas as pd
import numpy as np
RGB = pd.read_csv('../input/HPAv18.csv')
len(RGB)
RGB.drop_duplicates(keep='first',inplace=True)
len(RGB)
##RGB.to_csv('input/HPAv18RGB.csv',index=False)
##time
RGB["Gene"] = ""
RGB["ind"] = ""
for index, row in RGB.iterrows():
    row['Gene'] = row['Id'][:15]
    row['ind'] = row['Id'][16:]
ids = RGB["ind"]
dupl = RGB[ids.isin(ids[ids.duplicated()])].sort_values('ind')
len(dupl)
uniqe_dupl = dupl.ind.drop_duplicates(keep='first')
len(uniqe_dupl)
##time
merged_labels = []
for row in uniqe_dupl:
    ds = dupl[dupl['ind']==row]
    tg = set()
    for i,r in ds.iterrows():
        for t in r['Target'].split(" "):
            if t not in tg:
                tg.add(str(t))
    merged_labels.append([" ".join(list(tg)),row])
    ##time
merged_labels = []
for row in uniqe_dupl:
    ds = dupl[dupl['ind']==row]
    tg = set()
    for i,r in ds.iterrows():
        for t in r['Target'].split(" "):
            if t not in tg:
                tg.add(str(t))
    merged_labels.append([" ".join(list(tg)),row])
df_marged = pd.DataFrame(merged_labels,columns=["Target","ind"])
df_marged.head()
RGB = RGB.drop(['Id','Gene'],axis=1)
RGB.head()
df_marged = pd.concat([df_marged,RGB])
len(df_marged)
without_dupl = df_marged.drop_duplicates(subset='ind', keep='first')
len(without_dupl)
without_dupl = without_dupl.rename(columns={'ind': 'Id'}).reindex(columns=["Id","Target"])
without_dupl.head(10)
without_dupl.to_csv("HPAv18RBG_wodpl.csv",index=False)

