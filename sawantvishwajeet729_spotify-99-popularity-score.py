# Import the neccessary packages 

import os

import numpy as np

import pandas as pd

import  matplotlib.pyplot as plt
df  =  pd.read_csv('../input/top-50-spotify/top50.csv', encoding='latin-1')
df.sort_values(by=['Popularity'], ascending=False)
# Lets remove the unwanted index column

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
df = pd.get_dummies(df)
df = df.reindex(sorted(df.columns), axis=1)
df.head()
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler(feature_range=(0,1))

print(scaler.fit(df))
scaled_df = scaler.transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=df.columns)
scaled_df.head()
awesome_song = pd.DataFrame(data=None, columns= df.columns)

awesome_song = awesome_song.append(pd.Series(), ignore_index=True)

awesome_song
awesome_song.iloc[0] = df.iloc[0]
awesome_song
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
awesome_song = awesome_song.reindex(sorted(awesome_song.columns), axis=1)



awesome_song = scaler.transform(awesome_song)
awesome_song = pd.DataFrame(awesome_song, columns=df.columns)
awesome_song
# This is another of my custom function to get the report of a regression model.

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

        #linear_report(new_col, pred)

        

        #Awesome songe

        awesome = awesome.drop(x, axis=1)

        awesome = awesome.reindex(sorted(awesome.columns), axis=1)

        awesome[x] = model.predict(awesome)

        print()

    return(awesome)
awesome_song = perfect_song(scaled_df,iter_col , awesome_song)
awesome_song
for i in range(0,100):

    awesome_song = perfect_song(scaled_df, iter_col, awesome_song)
awesome_song = awesome_song.reindex(sorted(awesome_song.columns), axis=1)

awesome_song
final_song = scaler.inverse_transform(awesome_song)

final_song = pd.DataFrame(final_song, columns=awesome_song.columns)

final_song
data_for_validate = scaled_df.copy()

target = data_for_validate['Popularity']

data_for_validate.drop(['Popularity'], axis=1, inplace =True)



from xgboost import XGBRegressor

data_for_validate = data_for_validate.reindex(sorted(data_for_validate.columns), axis=1)

model = XGBRegressor(n_estimators=2000, n_jobs=-1, learning_rate=0.01)

model.fit(data_for_validate, target)
validate_song = awesome_song.drop(['Popularity'], axis=1)
validate_song = validate_song.reindex(sorted(validate_song.columns), axis=1)

validate_song.columns
model.predict(validate_song)