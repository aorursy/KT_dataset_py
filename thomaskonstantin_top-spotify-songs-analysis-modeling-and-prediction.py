import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as ex

import plotly.graph_objs as go

import plotly.figure_factory as ff

sns.set_style('darkgrid')
s_data =pd.read_csv('/kaggle/input/top50spotify2019/top50.csv',encoding='ISO-8859-1')

s_data.head(5)
s_data.drop(s_data.columns[0] ,axis=1,inplace=True)
s_data.info()
s_data.describe(include='all')
number_of_unique_artists = len(s_data['Artist.Name'].value_counts().to_list())

number_of_unique_genres = len(s_data['Genre'].value_counts().to_list())

print("Number Of Unique Artists: ",number_of_unique_artists,' | ',' Number Of Unique Genres: ',number_of_unique_genres)
corrs = s_data.corr()

plt.figure(figsize=(20,11))

ax = sns.heatmap(corrs,cmap='Blues',annot=True)
plt.figure(figsize=(20,11))

ax = sns.countplot(s_data['Genre'],palette='Greens')

ax.set_xticklabels(ax.get_xticklabels(),rotation=90,size=13)

ax.set_title('Distribution Of Genres Across Our Data',fontsize=16)

ax.patches[2].set_fc('r')
main_genres = ['rock','pop','blues','hip hop','jazz','reggae','techno','trap','regga','rap','r&b']

def check_genre(sir):

    for word in main_genres:

        if sir.find(word) != -1:

            if word == 'rap':

                return 'hip hop'

            else:

                return word

    return sir



s_data['Main.Genre'] = s_data['Genre'].apply(check_genre)
plt.figure(figsize=(20,11))

ax = sns.countplot(s_data['Main.Genre'],palette='Greens',order=s_data['Main.Genre'].value_counts().index)

ax.set_xticklabels(ax.get_xticklabels(),rotation=90,size=13)

ax.set_title('Distribution Of Genres Across Our Data',fontsize=16)

ax.patches[0].set_fc('r')

ax.patches[1].set_fc((0.75,0,0))

ax.patches[2].set_fc((0.50,0,0))

plt.legend({'Most Frequent Music Genre':0},prop={'size':'16'})

plt.show()
plt.figure(figsize=(20,11))

ax = sns.countplot(s_data['Artist.Name'],palette='Greens',order = s_data['Artist.Name'].value_counts().index,label='Top Artist')

ax.set_xticklabels(ax.get_xticklabels(),rotation=90,size=13)

ax.set_title('Distribution Of Genres Across Our Data',fontsize=16)

ax.patches[0].set_fc('r')

plt.legend(prop={'size':'16'})
#Our Top 10 Artist And Top 10 Genres 

top_10_artist = s_data['Artist.Name'].value_counts()[:10]

top_10_genres = s_data['Genre'].value_counts()[:10]

top_10_songs = s_data.iloc[s_data['Popularity'].nlargest(10).index,:]

top_10_artist.to_frame()

top_10_songs
plt.figure(figsize=(20,11))

ax = sns.distplot(s_data['Popularity'],hist_kws={'color':'r'},kde_kws={'color':'g','lw':'6'})

textstr = '\n'.join(

    

        (   r'$\mu=%.2f$' % (s_data['Popularity'].mean(),)

          , r'$\mathrm{median}=%.2f$' % (s_data['Popularity'].median(),)

          , r'$\sigma=%.2f$' % (s_data['Popularity'].std(),)

          , r'Skew=%.2f' % (s_data['Popularity'].skew(),)

          , r'Kurtosis=%.2f' % (s_data['Popularity'].kurt(),)



        )

    

                  )



props = dict(boxstyle='round', facecolor='red', alpha=0.5)

ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=17,

        verticalalignment='top', bbox=props)

ax.set_title('Distribution Of Popularity Scores In Our Data',fontsize=16)

ax.set_xlabel('Popularity',fontsize=16)

plt.legend()
gender = {'Ed Sheeran':'M','The Chainsmokers':'Group','Shawn Mendes':'M','Post Malone':'M','Sech':'M','Marshmello':'M','Billie Eilish':'F','J Balvin':'M',

         'Lil Nas X':'M','Ariana Grande':'F','Daddy Yankee':'M','Y2K':'M','DJ Snake':'M','Lewis Capaldi':'M','Chris Brown':'M','Khalid':'M','Lizzo':'F','Lauv':'M',

         'Kygo':'M','Ali Gatie':'M','Lady Gaga':'F','Bad Bunny':'M','Lunay':'M','Sam Smith':'M','Anuel AA':'M','Nicky Jam':'M','Lil Tecca':'M','ROSALÍA':'F','Young Thug':'M',

         'Martin Garrix':'M','Katy Perry':'F','Jhay Cortez':'M','Drake':'M','Tones and I':'F','Taylor Swift':'F','Jonas Brothers':'Group','MEDUZA':'M','Maluma':'M'}

s_data['Artist.Gender'] = s_data['Artist.Name'].apply(lambda x: gender[x])
from nltk.sentiment.vader import SentimentIntensityAnalyzer
top_10_arist_df = s_data[s_data['Artist.Name'].isin(top_10_artist.index)]

sia = SentimentIntensityAnalyzer()

top_10_arist_df['Track_Name_Sentiment.c'] = top_10_arist_df['Track.Name'].apply(lambda x: sia.polarity_scores(x)['compound'])

top_10_arist_df
plt.figure(figsize=(20,11))

ax = sns.countplot(top_10_arist_df['Artist.Gender'])

ax.set_title('Distribution Of Gender Among The Top 10 Artists',fontsize=17)
#our top 10 songs

top_10_songs
ex.scatter_polar(s_data,theta='Main.Genre',r='Beats.Per.Minute',color ='Popularity',title='Spread of different genre popularity according to beats per minute')
ex.density_heatmap(s_data,x='Beats.Per.Minute',y='Popularity',title='Popularity counts according to BPM ')


numeric_f = top_10_songs.columns[3:13]

plt.figure(figsize=(20,11))

cor = top_10_songs.corr()

ax = sns.distplot((top_10_songs['Danceability']-top_10_songs['Danceability'].mean())/top_10_songs['Danceability'].std(),hist=False,label='Danceability')

ax = sns.distplot((top_10_songs['Energy']-top_10_songs['Energy'].mean())/top_10_songs['Energy'].std(),hist=False,label='Energy')

ax = sns.distplot((top_10_songs['Valence.']-top_10_songs['Valence.'].mean())/top_10_songs['Valence.'].std(),hist=False,label='Valence')

ax = sns.distplot((top_10_songs['Length.']-top_10_songs['Length.'].mean())/top_10_songs['Length.'].std(),hist=False,label='Length')

ax = sns.distplot((top_10_songs['Beats.Per.Minute']-top_10_songs['Beats.Per.Minute'].mean())/top_10_songs['Beats.Per.Minute'].std(),hist=False,label='Beats.Per.Minute')

ax.set_xlabel('Tansformed Distribution',fontsize=16)

ax.set_title('Normalized Distributions Of The Most Significant Features In Our Top 10 Songs',fontsize=16)

plt.legend(prop={'size':'20'})

plt.show()

sd_data = s_data.copy()

geners_one = pd.get_dummies(sd_data['Main.Genre'],prefix='Genre')

geners_one = geners_one[geners_one.columns[1:]]

sd_data = pd.concat([sd_data,geners_one],axis=1)

sd_data = sd_data.drop(columns='Main.Genre')
sd_data['Track_Name_Sentiment'] = sd_data['Track.Name'].apply(lambda x: sia.polarity_scores(x)['compound'])

sd_data['Track_Name_Length'] = sd_data['Track.Name'].apply(lambda x: len(x))

sd_data['Genre'] =sd_data['Genre'].astype('category').cat.codes

sd_data['Artist.Gender'] =sd_data['Artist.Gender'].astype('category').cat.codes



plt.figure(figsize=(20,11))

sdcor = sd_data.corr()

sns.heatmap(sdcor,annot=True,cmap='Blues')
plt.figure(figsize=(20,11))

p_correaltion=['Speechiness.','Beats.Per.Minute','Valence.','Genre','Genre_hip hop']

fig,axs = plt.subplots(2,2)

fig.set_figheight(15)

fig.set_figwidth(15)

sns.regplot(y=sd_data['Popularity'],x=sd_data[p_correaltion[1]],ax=axs[0,1],color='r')

sns.regplot(y=sd_data['Popularity'],x=sd_data[p_correaltion[0]],ax=axs[0,0])

sns.regplot(y=sd_data['Popularity'],x=sd_data[p_correaltion[2]],ax=axs[1,0],color='g')

sns.regplot(y=sd_data['Popularity'],x=sd_data[p_correaltion[3]],ax=axs[1,1],color='c')
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KNeighborsRegressor
y = sd_data['Popularity']

X = sd_data[['Speechiness.','Beats.Per.Minute','Valence.','Genre','Genre_hip hop','Genre_escape room','Genre_r&b']]

LR_pipe = Pipeline(steps=[('scaler',StandardScaler()),('poly',PolynomialFeatures(degree=1)),('LinearRegression',LinearRegression())])



LR_scores = -1*cross_val_score(LR_pipe,X,y,cv=5,scoring='neg_mean_squared_error')



print('Linear Regression Cross Validation Average MSE: ',LR_scores.mean())

tr = go.Scatter(x = np.arange(1,len(LR_scores)+1),y=LR_scores,mode='lines')

fig = go.Figure(data=[tr],layout={'title':'Linear Regression Fold MSE Scores','xaxis_title':'Fold Number','yaxis_title':'MSE'})

fig.show()
LR_pipe.fit(X,y)

pred = LR_pipe.predict(X)

mse = mean_squared_error(pred,y)

print('Linear Regression Fitted Using All The Data MSE: ',mse )
DT_Pipe =  Pipeline(steps=[('scaler',StandardScaler()),('DT',DecisionTreeRegressor(max_leaf_nodes=10))])

DT_scores = -1*cross_val_score(DT_Pipe,X,y,cv=5,scoring='neg_mean_squared_error')

DT_Pipe.fit(X,y)

print('Decision Tree MSE: ',DT_scores.mean())

tr = go.Scatter(x = np.arange(1,len(DT_scores)+1),y=DT_scores,mode='lines')

fig = go.Figure(data=[tr],layout={'title':'Decision Tree Fold MSE Scores','xaxis_title':'Fold Number','yaxis_title':'MSE'})

fig.show()
RF_Pipe =  Pipeline(steps=[('scaler',StandardScaler()),('DT',RandomForestRegressor(max_leaf_nodes=14,n_estimators=20,random_state=42))])

RF_scores = -1*cross_val_score(RF_Pipe,X,y,cv=5,scoring='neg_mean_squared_error')

RF_Pipe.fit(X,y)

print('RandomForest MSE: ',RF_scores.mean())

tr = go.Scatter(x = np.arange(1,len(RF_scores)+1),y=RF_scores,mode='lines')

fig = go.Figure(data=[tr],layout={'title':'Random Forest Fold MSE Scores','xaxis_title':'Fold Number','yaxis_title':'MSE'})

fig.show()
Knn_Pipe =  Pipeline(steps=[('scaler',StandardScaler()),('DT',KNeighborsRegressor(n_neighbors=5))])

Knn_scores = -1*cross_val_score(Knn_Pipe,X,y,cv=5,scoring='neg_mean_squared_error')

Knn_Pipe.fit(X,y)

print('Knn MSE: ',Knn_scores.mean())

tr = go.Scatter(x = np.arange(1,len(Knn_scores)+1),y=Knn_scores,mode='lines')

fig = go.Figure(data=[tr],layout={'title':'KNN Fold MSE Scores','xaxis_title':'Fold Number','yaxis_title':'MSE'})

fig.show()
pred = LR_pipe.predict(X)*0.2 + RF_Pipe.predict(X)*0.3 + 0.4* DT_Pipe.predict(X) + Knn_Pipe.predict(X)*0.1

mse = mean_squared_error(pred,y)
plt.figure(figsize=(20,11))

ax=sns.lineplot(x=np.arange(0,len(y)),y=y,label = 'Real Popularity Value')

ax = sns.lineplot(x=np.arange(0,len(y)),y=pred,label = 'Model Predicted Popularity Value')

ax.set_xlabel('Song Index',fontsize=16)

ax.set_ylabel('Popularity',fontsize=16)

prop3 = dict(boxstyle='round',facecolor='orange',alpha=0.5)

ax.text(0.05, 0.25, 'MSE : {:.2f}'.format(mse), transform=ax.transAxes, fontsize=17,

        verticalalignment='top', bbox=prop3)

plt.legend(prop={'size':'20'})
import plotly.graph_objs as go



trace1 = go.Scatter(x=np.arange(0,len(y)),y=y,mode='markers',name='Real Values')

trace2 = go.Scatter(x=np.arange(0,len(y)),y=pred,mode='markers',name='Predictions')

data = [trace1,trace2]

layout = dict(title='Prediction vs Real Values',xaxis=dict(title='Sample Number',ticklen=15),yaxis=dict(title='Popularity',ticklen=15))

fig = go.Figure(data=data,layout=layout)

fig.show()