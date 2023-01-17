import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 

from scipy.stats import pearsonr

%matplotlib inline 
df=pd.read_csv('../input/top2018.csv')
df.info()
df['Duration_min']=df['duration_ms']/60000
df.drop(columns='duration_ms',inplace=True)
sns.heatmap(df.corr(),cmap="YlOrRd")
df['artists'].value_counts().head(10)
sns.set_style(style='darkgrid')

sns.distplot(df['danceability'],hist=True,kde=True)
# Set conditions

Vd=df['danceability']>=0.75

Ld=(df['danceability']>=0.5) & (df['danceability']<0.75)

Nd=df['danceability']<0.5
# Create DataFrame 
data=[Vd.sum(),Ld.sum(),Nd.sum()]

Dance=pd.DataFrame(data,columns=['percent'],

                   index=['Very','Regular','Instrumental'])
Dance
sns.distplot(df['energy'])
# Set conditions

Ve=df['energy']>=0.75

Re=(df['energy']>=0.5) & (df['energy']<0.75)

Le=df['energy']<0.5
#Create DataFrame

data=[Ve.sum(),Re.sum(),Le.sum()]

Energy=pd.DataFrame(data,columns=['percent'],

                   index=['Very Energy','Regular Energy','Low Energy'])
Energy
Correlation=df[['danceability','energy','valence','loudness','tempo']]
sns.heatmap(Correlation.corr(),annot=True,cmap="YlOrRd")
sns.jointplot(data=Correlation,y='energy',x='loudness',kind='reg',stat_func=pearsonr)
df['Rhythm']=df['tempo']
df.loc[df['tempo']>168,'Rhythm']='Presto'

df.loc[(df['tempo']>=110) & (df['tempo']<=168),'Rhythm']='Allegro'

df.loc[(df['tempo']>=76) & (df['tempo']<=108),'Rhythm']='Andante'

df.loc[(df['tempo']>=66) & (df['tempo']<=76),'Rhythm']='Adagio'

df.loc[df['tempo']<65,'Rhythm']='Length'
df['Rhythm'].value_counts()
sns.set_style(style='darkgrid')

Rhy=df['Rhythm'].value_counts()

Rhy_DF=pd.DataFrame(Rhy)

sns.barplot(x=Rhy_DF.Rhythm, y=Rhy_DF.index, palette="viridis")

plt.title('Popular keys')
df[['name','artists','danceability','valence','tempo','Rhythm']].sort_values(by='danceability',ascending=False).head(10)
df[['name','artists','energy','valence','tempo','Rhythm']].sort_values(by='energy',ascending=False).head(10)
df[['name','artists','energy','valence','tempo','Rhythm']].sort_values(by='valence',ascending=False).head(10)
df['artists'].value_counts().head(4)
XXXTENT=df[df['artists']=='XXXTENTACION']
XXXTENT[['name','danceability','energy','loudness','valence','tempo','Rhythm']]
PMalone=df[df['artists']=='Post Malone']
PMalone[['name','danceability','energy','loudness','valence','tempo','Rhythm']]
Drake=df[df['artists']=='Drake']
Drake[['name','danceability','energy','loudness','valence','tempo','Rhythm']]
Edshe=df[df['artists']=='Ed Sheeran']

Edshe[['name','danceability','energy','loudness','valence','acousticness','tempo','Rhythm']]
Mayores=df[df['mode']==1]

Menores=df[df['mode']==0]
# Variables separation according to the scale to which Major or minor belongs
MayoresD=Mayores[Mayores['danceability']>=0.5]

MenoresD=Menores[Menores['danceability']>=0.5]
# We eliminate the columns that say nothing in the study
MayoresD=Mayores.drop(columns=['mode','time_signature'])

MenoresD=Menores.drop(columns=['mode','time_signature'])
# Heat map for Major scales
sns.heatmap(MayoresD.corr(),cmap="YlOrRd")
# Heat map for Less scales
sns.heatmap(MenoresD.corr(),cmap="YlOrRd")
# We create the variables and assign the columns that we want to correlate
MaycorD=MayoresD[['danceability','energy','valence','loudness','tempo']]

MencorD=MenoresD[['danceability','energy','valence','loudness','tempo']]
# Major scale correlation
sns.heatmap(MaycorD.corr(),annot=True,cmap="YlOrRd")
sns.heatmap(MencorD.corr(),annot=True,cmap="YlOrRd")
df.loc[ df['key']==0 ,'key']='C'    

df.loc[ df['key']==1 ,'key']='C#'    

df.loc[ df['key']==2 ,'key']='D'    

df.loc[ df['key']==3 ,'key']='D#'    

df.loc[ df['key']==4 ,'key']='E'    

df.loc[ df['key']==5 ,'key']='F'    

df.loc[ df['key']==6 ,'key']='F#'    

df.loc[ df['key']==7 ,'key']='G'    

df.loc[ df['key']==8 ,'key']='G#'    

df.loc[ df['key']==9 ,'key']='A'    

df.loc[ df['key']==10 ,'key']='A#' 

df.loc[ df['key']==11 ,'key']='B' 
sns.set_style(style='darkgrid')

keys=df['key'].value_counts()

key_DF=pd.DataFrame(keys)

sns.barplot(x=key_DF.key, y=key_DF.index, palette="viridis")

plt.title('Popular keys')
df[['danceability','energy','valence','key']].groupby(by='key').mean().sort_values(by='danceability',ascending=False)