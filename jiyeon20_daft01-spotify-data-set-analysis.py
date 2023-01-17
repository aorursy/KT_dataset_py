# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
filename = "/kaggle/input/spotify-dataset-19212020-160k-tracks/data.csv"
df = pd.read_csv(filename)
df.head()
df_after_2016 = df[df['year'] >= 2016]
df_after_2016 = df_after_2016[df_after_2016['year'] < 2018]

df_after_2016 = df_after_2016[df_after_2016['popularity'] >= 75]

df_after_2016.head()



import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO


# 일단 feature 사이 관계가 있는지 보자~

corr = df_after_2016[['acousticness','danceability','energy',
'instrumentalness','liveness','tempo','valence']].corr()

sns.set(style='whitegrid')
%matplotlib inline

plt.figure(figsize=(13,10))
sns.heatmap(corr, annot=True)

 
            
features = ['energy', 'valence', 'danceability']
plt.figure(figsize=(8,7))

sns.distplot(df_after_2016['energy'], color = 'skyblue', label = 'Energy', hist=True, kde=False,rug=False)
sns.distplot(df_after_2016['valence'], color = 'pink', label = 'Valence', hist=True, kde=False,rug=False)
sns.distplot(df_after_2016['danceability'], color = 'green', label = 'Danceability', hist=True, kde=False,rug=False,axlabel = ' <-negatvie                 positive->')
plt.title('All')
plt.legend(title="Features")
plt.show()


file = "/kaggle/input/spotifys-worldwide-daily-song-ranking/data.csv"
streams = pd.read_csv(file)
streams.head()


# coachella mid april
# lollapalooza early august

# 2017년 1월부터 8월까지만 
jan_to_sep = streams[(streams['Date'] >= '2017-01-01') & (streams['Date'] < '2017-09-01')]

# 필요없는 컬럼 제거
jan_to_sep = jan_to_sep.drop(['URL','Region'], axis=1)

only_streams = jan_to_sep.drop(['Position', 'Date'], axis=1)

# 아티스트와 트랙별로 누적 스트리밍수 얻기
stream_sum = only_streams.groupby(by=['Artist','Track Name']).sum().groupby(level=[0]).cumsum()


# 누적  1천만 번 이상 스트리밍된 아티스트만

stream_sum_big = stream_sum[stream_sum['Streams'] > 10000000] 

# stream_sum_big.head()

# merge위해 컬럼명 동일하게 변경 (아티스트와 트랙명)

df_after_2016 = df_after_2016.rename(columns = {'artists' : 'Artist'})

df_after_2016 = df_after_2016.rename(columns = {'name' : 'Track Name'})

# 첫번째 데이터 프레임의 아티스트명 양 옆 캐릭터 정리
df_after_2016['Artist'] = df_after_2016['Artist'].map(lambda x: x.lstrip("['").rstrip("']"))


# merge(key는 Artist, Track Name) : 2016~2017년의 천만번이상 스트리밍된 아티스트와 트랙들

merged = pd.merge(stream_sum_big, df_after_2016, on=['Artist', 'Track Name'], how='inner')

merged.tail()


# 이런 노래들이 많이 스트리밍 되는데 이 곡들의 피처들은 이러니 비슷한 피처를 가진 노래들을 부른 가수를 부르자..


sns.distplot(merged['danceability'], color = 'skyblue', label = 'Danceability', hist=True, kde=False,rug=False)
sns.distplot(merged['valence'], color = 'pink', label = 'Valence', hist=True, kde=False,rug=False,axlabel = ' ')


sns.lmplot(x='valence', y='danceability', data=merged, ci=None, line_kws={'color': 'red'})




plt.figure(figsize=(8,7))
     
sns.distplot(merged['energy'], color = 'skyblue', label = 'Energy', hist=True, kde=False,rug=False)
sns.distplot(merged['valence'], color = 'pink', label = 'Valence', hist=True, kde=False,rug=False)
sns.distplot(merged['danceability'], color = 'green', label = 'Danceability', hist=True, kde=False,rug=False,axlabel = ' <-negatvie                 positive->')
plt.title('over 10 million streams')
plt.legend(title="Features")   
plt.show()


sns.lmplot(x='valence', y='energy', data=merged, ci=None, line_kws={'color': 'red'})


corr = merged[['acousticness','danceability','energy',
'instrumentalness','liveness','tempo','valence']].corr()

sns.set(style='whitegrid')
%matplotlib inline

plt.figure(figsize=(13,10))
sns.heatmap(corr, annot=True)

import numpy as np
# 페스티벌 라인업
coachella = ['Radiohead', 'Travis Scott', 'Mac Miller', 'Bon Iver', 'DJ Snake', 'ScHoolboy Q', 'Tory Lanez','Kendrick Lamar', 'Lorde', 'The xx','Father John Misty', 'Empire of the Sun', 'Crystal Castles', 'Phantogram', 'Denzel Curry', 'Broods', 'Tennis', 'Bonobo', 'The Interrupters', 'Oh Wonder', 'Capital Cities', 'Lady Gaga', 'Future', 'Martin Garrix',  'Gucci Mane', 'Chicano Batman', 'Thundercat']
lollapalooza = ['The Killers', 'alt-j','Run The Jewels', 'Big Sean', 'The Head and the Heart', 'Lil Uzi Vert','Migos', 'Tove Lo', 'Mac DeMarco', 'Russ', 'Sylvan Esso', 'Whitney', '21 Savage', 'Car Seat Headrest', 'SLANDER' , 'KAYTRANADA', 'Getter', 'Majid Jordan', 'Highly Suspect', 'Aminé', '6LACK', 'Joseph', 'Wiz Khalifa', 'Lorde', 'Migos', 'Cage the Elephant', 'Liam Gallagher', 'Oliver Tree','Muse','George Ezra', 'Declan McKenna','Atlas Genius', 'Spoon', 'Jon Bellion', 'A-Trak', 'HONNE', 'Kaytranada', 'Gryffin', 'The Drums', 'Capital Cities', 'The Killers', 'Blink-182', 'Ryan Adams', 'Tegan And Sara', 'Foster The People','Phantogram', 'Saint Jhn','Kaleo','Crystal Castles','Mondo Cozmo', 'Gramatik','The xx','Zara Larsson', 'Chance the Rapper', 'Glass Animals', 'Mac DeMarco', 'Russ', 'alt-J','Vance Joy', 'Royal Blood','Highly Suspect','Warpaint', 'Alvvays', 'The Head and the Heart', 'Alison Wonderland']
pitchfork = ['LCD Soundsystem', 'A Tribe Called Quest', 'Francis and the Lights', 'Solange', 'The Avalanches', 'Pinegrove', 'Dirty Projectors','Vince Staples','Frankie Cosmos', 'Kamaiyah','Thurston Moore', 'Danny Brown', 'Hiss Golden Messenger', 'Madame Gandhi', 'William Tyler', 'Arca & Jesse Kanda','Priest', 'Dawn', 'Vagabon','The Feelies','Angel Olsen', 'Mitski','PJ Harvey','Jeff Rosenstock','George Clinton', 'Weyes Blood','Cherry Glazerr','Madlib', 'Francis and the Lights', 'Joey Purp','Kilo Kish','Jamila Woods','Nicholas Jaar', 'Hamilton Leithauser','Colin Stetson','American Football', 'Ride', 'Ne-Hi','Isaiah Rashad']

whole = coachella + lollapalooza + pitchfork

# 라인업 리스트를 데이터 프레임으로 변경
lineup = pd.DataFrame(whole, columns=['Artist'])

# 각 페스티벌을 데이터프레임으로 변환 
coachella = pd.DataFrame(coachella, columns=['Artist'])
lollapalooza = pd.DataFrame(lollapalooza, columns=['Artist'])
pitchfork = pd.DataFrame(pitchfork, columns=['Artist'])

# 원 df의 컬럼명 변경, 아티스트 양 옆 캐릭터 제거
df = df.rename(columns = {'artists' : 'Artist'})
df['Artist'] = df['Artist'].map(lambda x: x.lstrip("['").rstrip("']"))

# 라인업 df와 원래df를 merge
lineup_features = pd.merge(lineup, df, on=['Artist'], how='inner')
lineup_features.shape


lineup_features = lineup_features[(lineup_features['popularity'] >= 50)]


lineup_dropped = lineup_features.drop(['Artist', 'duration_ms', 'explicit','id','key','name','popularity','release_date', 'year'], axis=1)

lineup_dropped.head()

# PCA 사용

from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler 
from sklearn.manifold import TSNE

scaler = StandardScaler()

Z = scaler.fit_transform(lineup_dropped)

pca = PCA(2)

pca.fit(Z)

print("\n Eigenvectors: \n", pca.components_)
print("\n Eigenvalues: \n",pca.explained_variance_)

B = pca.transform(Z)
print("\n Projected Data: \n", B)

B = pd.DataFrame(B, columns = ["PC1", "PC2"])


kmeans_pca = KMeans(n_clusters = 2)
kmeans_pca.fit(B)
labels_pca  = kmeans_pca.labels_

pca_data = pd.Series(labels_pca)
B["clusters"] = pca_data.values



print(lineup_features.shape)
print(B.shape)


lineup_features.reset_index(drop=True, inplace=True)
B.reset_index(drop=True, inplace=True)
combined_lineup = pd.concat([lineup_features, B], axis=1)

# seaborn 스캐터 플롯
import seaborn as sns
sns.set(rc={'figure.figsize':(20,20)})

ax = combined_lineup.plot.scatter(
    x='PC1',y='PC2', c='clusters', colorbar=False)
for i, txt in enumerate(combined_lineup['Artist'].tolist()):
    ax.annotate(txt, (combined_lineup.PC1[i], combined_lineup.PC2[i]))

sns.scatterplot(x = "PC1", y = "PC2", hue = "clusters", data = B, palette = 'pastel')
plt.xlim(-4, 6 )
plt.ylim(-4, 5)

plt.show()





# plotly scatter plot

import plotly.express as px
import plotly.io as po

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

fig = px.scatter(combined_lineup, x='PC1', y='PC2', color = 'clusters', title = 'Clustered Artists', hover_data=['Artist'])
fig.show()

from plotly.offline import plot
plot(fig, filename='spotify.html', auto_open=False)

# # kmeans 사용

# model = KMeans(n_clusters=3)

# kY = model.fit(lineup_dropped)

# predict = pd.DataFrame(kY.predict(lineup_dropped))
# predict.columns=['predict']




# # 원 데이터프레임과 예상결과 합치기
# lineup_dropped["clusters"] = predict.values

# lineup_dropped.head()

# clustered_lineup = pd.merge(lineup_features,lineup_dropped, on=['acousticness', 'danceability', 'energy', 'instrumentalness',
#        'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence'], how='inner')

# clustered_lineup.head()

# 각 페스티벌 데이터프레임을 원래 데이터프레임과 inner merge
coachella_features = pd.merge(coachella, df, on=['Artist'], how='inner')
lollapalooza_features = pd.merge(lollapalooza, df, on=['Artist'], how='inner')
pitchfork_features = pd.merge(pitchfork, df, on=['Artist'], how='inner')


# 코첼라 코릴레이션
corr_c = coachella_features[['acousticness','danceability','energy',
'instrumentalness','liveness','tempo','valence']].corr()

sns.set(style='whitegrid')
%matplotlib inline

plt.figure(figsize=(13,10))
sns.heatmap(corr_c, annot=True)




plt.figure(figsize=(8,7))

sns.distplot(coachella_features['energy'], color = 'skyblue', label = 'Energy', hist=True, kde=False,rug=False)
sns.distplot(coachella_features['valence'], color = 'pink', label = 'Valence', hist=True, kde=False,rug=False)
sns.distplot(coachella_features['danceability'], color = 'green', label = 'Danceability', hist=True, kde=False,rug=False,axlabel = ' <-negatvie                 positive->')
plt.title('Coachella')
plt.legend(title="Features")
plt.show()




# 롤라팔루자
corr_l = lollapalooza_features[['acousticness','danceability','energy',
'instrumentalness','liveness','tempo','valence']].corr()

sns.set(style='whitegrid')
%matplotlib inline

plt.figure(figsize=(13,10))
sns.heatmap(corr_l, annot=True)

plt.figure(figsize=(8,7))

sns.distplot(lollapalooza_features['energy'], color = 'skyblue', label = 'Energy', hist=True, kde=False,rug=False)
sns.distplot(lollapalooza_features['valence'], color = 'pink', label = 'Valence', hist=True, kde=False,rug=False)
sns.distplot(lollapalooza_features['danceability'], color = 'green', label = 'Danceability', hist=True, kde=False,rug=False,axlabel = ' <-negatvie                 positive->')
plt.title('Lollapalooza')
plt.legend(title="Features")
plt.show()


#피치포크
corr_p = pitchfork_features[['acousticness','danceability','energy',
'instrumentalness','liveness','tempo','valence']].corr()

sns.set(style='whitegrid')
%matplotlib inline

plt.figure(figsize=(13,10))
sns.heatmap(corr_p, annot=True)


plt.figure(figsize=(8,7))
sns.distplot(pitchfork_features['energy'], color = 'skyblue', label = 'Energy', hist=True, kde=False,rug=False)
sns.distplot(pitchfork_features['valence'], color = 'pink', label = 'Valence', hist=True, kde=False,rug=False)
sns.distplot(pitchfork_features['danceability'], color = 'green', label = 'Danceability', hist=True, kde=False,rug=False,axlabel = '<-negatvie                 positive->')
plt.title('Pitchfork')
plt.legend(title="Features")
plt.show()



lineup_essential = lineup_features.drop(['Artist', 'duration_ms', 'explicit','id','key','name','popularity','release_date', 'year','danceability', 'instrumentalness', 'liveness','loudness','speechiness'], axis=1)


# PCA 사용

from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler 
from sklearn.manifold import TSNE

scaler = StandardScaler()

X = scaler.fit_transform(lineup_essential)

pca = PCA(2)

pca.fit(X)

print("\n Eigenvectors: \n", pca.components_)
print("\n Eigenvalues: \n",pca.explained_variance_)

C = pca.transform(X)
print("\n Projected Data: \n", C)

C = pd.DataFrame(C, columns = ["PC1", "PC2"])


kmeans_pca = KMeans(n_clusters = 2)
kmeans_pca.fit(C)
labels_pca  = kmeans_pca.labels_

pca_data = pd.Series(labels_pca)
C["clusters"] = pca_data.values

lineup_features.reset_index(drop=True, inplace=True)
C.reset_index(drop=True, inplace=True)
combined_lineup_essential = pd.concat([lineup_features, C], axis=1)



# plotly scatter plot

import plotly.express as px
import plotly.io as po

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

fig = px.scatter(combined_lineup_essential, x='PC1', y='PC2', color = 'clusters', title = 'Clustered Artists', hover_data=['Artist'])
fig.show()

from plotly.offline import plot
plot(fig, filename='spotify.html', auto_open=False)


