import os

import numpy as np

import pandas as pd

import  matplotlib.pyplot as plt

df  =  pd.read_csv('../input/top50spotify2019/top50.csv', encoding='latin-1')
df.columns = ['Unnamed: 0', 'Track_Name', 'ArtisT_Name', 'Genre', 'Beats_Per_Minute','Energy', 

              'Danceability', 'Loudness_dB', 'Liveness', 'Valence','Length', 'Acousticness', 

              'Speechiness', 'Popularity']
df.head()
del df['Unnamed: 0']
df['ArtisT_Name'].unique()
import seaborn as sns

corr = df.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)
import statsmodels.formula.api as smf

import statsmodels.stats.multicomp as multi



def uni_analysis(df):

    x = (df.select_dtypes(include=['O']))

    p_value = []

    for i in x:

        para = 'Popularity ~ '+str(i)

        model = smf.ols(formula=para, data=df)

        results = model.fit()

        p_value.append(results.f_pvalue)

    df1 = pd.DataFrame(list(zip(x,p_value)), columns =['Variable', 'p_value'])

    df1['Drop_column'] = df1['p_value'].apply(lambda x: 'True' if x > 0.05 else 'False')

    return df1
uni_analysis(df)
del df['ArtisT_Name']
del df['Track_Name']
df.dtypes.value_counts()
df['Beats_Per_Minute'] = df['Beats_Per_Minute'].astype(int)

df.Energy = df.Energy.astype(int)

df.Danceability = df.Danceability.astype(int)

df.Loudness_dB = df.Loudness_dB.astype(int)

df.Liveness = df.Liveness.astype(int)

df.Valence = df.Valence.astype(int)

df.Length = df.Length.astype(int)

df.Acousticness = df.Acousticness.astype(int)

df.Speechiness = df.Speechiness.astype(int)

df.Popularity = df.Popularity.astype(int)
df.dtypes.value_counts()
df = pd.get_dummies(df)
df.shape
awesome_song = pd.DataFrame(data=None, columns= df.columns)
awesome_song = awesome_song.append(pd.Series(), ignore_index=True)

awesome_song
for i in range(len(awesome_song.columns)-1):

    awesome_song.iloc[0,i] = 50
awesome_song.iloc[0,3] = 0
awesome_song.columns
genre_list = ['Genre_atl hip hop', 'Genre_australian pop', 'Genre_big room',

       'Genre_boy band', 'Genre_brostep', 'Genre_canadian hip hop',

       'Genre_canadian pop', 'Genre_country rap', 'Genre_dance pop',

       'Genre_dfw rap', 'Genre_edm', 'Genre_electropop', 'Genre_escape room',

       'Genre_latin', 'Genre_panamanian pop', 'Genre_pop', 'Genre_pop house',

       'Genre_r&b en espanol', 'Genre_reggaeton', 'Genre_reggaeton flow',

       'Genre_trap music']
for x in genre_list:

    awesome_song[x] = 0
awesome_song['Genre_pop'] = 1
awesome_song['Popularity'] = 99
awesome_song['Beats_Per_Minute'] = awesome_song['Beats_Per_Minute'].astype(int)

awesome_song.Energy = df.Energy.astype(int)

awesome_song.Danceability = awesome_song.Danceability.astype(int)

awesome_song.Loudness_dB = awesome_song.Loudness_dB.astype(int)

awesome_song.Liveness = awesome_song.Liveness.astype(int)

awesome_song.Valence = awesome_song.Valence.astype(int)

awesome_song.Length = awesome_song.Length.astype(int)

awesome_song.Acousticness = awesome_song.Acousticness.astype(int)

awesome_song.Speechiness = awesome_song.Speechiness.astype(int)

awesome_song.Popularity = awesome_song.Popularity.astype(int)
#Beat_pm = df['Beats_Per_Minute']
#df1 = df.drop('Beats_Per_Minute', axis=1)
from sklearn.ensemble import RandomForestRegressor

#rf = RandomForestRegressor(n_estimators=820, max_depth=None, n_jobs=-1, min_samples_split=2, min_samples_leaf=1, random_state = 77, oob_score=True, max_features=0.45, criterion="mse")

#rf.fit(df1, Beat_pm)
#predic = rf.predict(df1)
#predic
def linear_report (y_actual, y_pred):

    from sklearn.metrics import mean_squared_error

    from sklearn.metrics import mean_absolute_error

    from sklearn.metrics import r2_score

    

    mae = mean_absolute_error(y_actual, y_pred)

    print('MAE value is: ',mae)

    mse = mean_squared_error(y_actual, y_pred)

    print('MSE value is: ', mse)

    import math

    rmse = math.sqrt(mse)

    print('RMSE value is: ', rmse)

    r2 = r2_score(y_actual, y_pred)

    print('R2 score is: ',r2)

    mape = 100*(mae/y_actual)

    accuracy = 100-np.mean(mape)

    print('Accuracy score is: ',accuracy)
#linear_report(Beat_pm, predic)
#from xgboost import XGBRegressor

#model = XGBRegressor(n_estimators=2000, n_jobs=-1, learning_rate=0.01)

#model.fit(df1, Beat_pm)
#predic = model.predict(df1)
#linear_report(Beat_pm, predic)
#del awesome_song['Beats_Per_Minute']

#awesome_song['Beats_Per_Minute'] = model.predict(awesome_song)
#awesome_song
iter_col = ['Beats_Per_Minute', 'Energy', 'Danceability', 'Loudness_dB', 'Liveness',

       'Valence', 'Length', 'Acousticness', 'Speechiness']
def perfect_song (df, iter_col, awesome):

    from xgboost import XGBRegressor

    for x in iter_col:

        new_col = df[x]

        df1 = df.drop(x, axis=1)

        df1 = df1.reindex(sorted(df1.columns), axis=1)

        model = XGBRegressor(n_estimators=2000, n_jobs=-1, learning_rate=0.01)

        model.fit(df1, new_col)

        pred = model.predict(df1)

        linear_report(new_col, pred)

        

        #Awesome song

        awesome = awesome.drop(x, axis=1)

        awesome = awesome.reindex(sorted(awesome.columns), axis=1)

        awesome[x] = model.predict(awesome)

        print()

    return(awesome)
awesome_song = perfect_song(df,iter_col , awesome_song)
awesome_song.dtypes
for i in range(0,5):

    awesome_song = perfect_song(df, iter_col, awesome_song)
awesome_song
# and that the combination required for an awesome pop song.