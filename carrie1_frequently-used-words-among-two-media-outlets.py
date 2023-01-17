import pandas as pd

import warnings

warnings.filterwarnings('ignore')



df = pd.read_json('../input/Sarcasm_Headlines_Dataset.json', lines=True)



df.info()



print('---------------------------------------')



df.head()
df_big = df.headline.str.split(" ", n=df.headline.str.len().max(), expand=True) 

df_big.head()
df2 = pd.concat([df, df_big], axis=1)

df2[['article_link','site','page']] = df['article_link'].str.split('.',expand=True,n=2)

df2.head()
pd.crosstab(df2.site,df2.is_sarcastic)
melt_df = pd.melt(df2, id_vars=['site','is_sarcastic'],value_vars=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,

                                                                  23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38])

melt_df.head(100)
g = melt_df.groupby(['site','value'])['is_sarcastic'].count().reset_index()

g.sort_values(by='is_sarcastic',ascending=False,inplace=True)

g.head()
common_words = ['to','of','in','and','the','with','on','for','by','at','a','from','out','up','is','this','are','be','as','that',

               'it','not','about','into','over','about','all','you','how','your','what','after','he','she','his','her','will',

               'who','where','why','an','why','has','an','my','have','one','can','we','more','no','when','their','these','was',

               'only','do','than','get','like','could','but','before','after']

df3 = g[~g.value.isin(common_words)]

df3.head(50)
df3[df3.value=='local']
df3[df3.value=='area']
df3[df3.value=='report:']
df3[df3.value=='trump']
df3[df3.value=='hillary']
df3[df3.value=='clinton']
df3[df3.value=='bernie']
df3[df3.value=='obama']