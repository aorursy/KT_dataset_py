import pandas as pd
import seaborn as sns
df = sns.load_dataset('diamonds')
df.head()
df.info()
df_table=df["table"]
sns.boxplot(x=df_table);
Q1 = df_table.quantile(0.25)
Q3 = df_table.quantile(0.75)
IQR = Q3-Q1
IQR
alt_sinir=Q3-1.5*IQR
alt_sinir
ust_sinir = Q3 + 1.5*IQR
ust_sinir
df_table<alt_sinir
df_table > ust_sinir
(df_table < alt_sinir) | (df_table > ust_sinir)
aykiri=(df_table < alt_sinir)
df_table[aykiri]#aykırı değerlere erişmek için
df_table[aykiri].index#aykrı değerlerin indekslerine erişmek için
aykiri_df=pd.DataFrame(data=df_table[aykiri])#dataframe'ye çevirme
aykiri_df.head()
type(df_table)
df_table=pd.DataFrame(data=df_table)
df_table.shape
new_df_table =df_table[~((df_table < (alt_sinir)) | (df_table > (ust_sinir))).any(axis = 1)]
new_df_table.shape
df.table.mean()#ortalama değeri hesaplama
df_table[aykiri]=df.table.mean()
df_table[aykiri]
Q1 = df_table.quantile(0.25)
Q3 = df_table.quantile(0.75)
IQR = Q3-Q1
ust_sinir = Q3 + 1.5*IQR
ust_sinir
alt_sinir=Q3-1.5*IQR
alt_sinir
aykiri=(df_table < alt_sinir)
df_table[aykiri]=alt_sinir
df_table[aykiri]
sns.boxplot(x=df_table)