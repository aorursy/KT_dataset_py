import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df=pd.read_csv('/kaggle/input/question-pairs-dataset/questions.csv')

df.shape
df.isna().sum()
df.dropna(axis='rows',inplace=True)

df.shape
df1=df[['qid1','question1']].rename(columns={'qid1':'qid','question1':'question'})

df2=df[['qid2','question2']].rename(columns={'qid2':'qid','question2':'question'})
df_q1q2=pd.concat([df1,df2],axis=0)

df_q1q2.shape
df_q1q2.drop_duplicates(inplace=True)

df_q1q2.shape
df_q1q2.head()
df_wrongmappings=df_q1q2.groupby('qid').size()[df_q1q2.groupby('qid').size()>1].to_frame().reset_index()
df_q1q2_wrong=df_wrongmappings[['qid']].merge(df_q1q2,how='inner',on='qid')
df_q1q2_wrong.head(100)
df_q1q2_wrong.shape