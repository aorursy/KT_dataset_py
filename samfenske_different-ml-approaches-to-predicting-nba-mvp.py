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
nba=pd.read_csv('/kaggle/input/nba-mvp-votings-through-history/mvp_votings.csv')

nba.columns
#view all of the columns in the dataframe

pd.set_option('display.max_columns', None)

nba
#check methodology of getting index of player



#data for 2017-18 season

ex=nba[nba['season'].isin(['2017-18'])]



#get index of mvp winner

index=[ex['win_pct'].idxmax()]



#now see who the player is by calling the index of the 'player' column

ex['player'][index]
nba[nba['season'].isin(['2015-16']) & nba['player'].isin(['LeBron James'])].index[0]
#set whole column to 'No', then just change to 'Yes' for mvp winners

nba['Mvp?']='No'



#for every season

for season in nba['season'].value_counts().index:

    

    #isolate data from that season

    season_df=nba[nba['season'].isin([season])]

    

    #get the index of player with most mvp points

    index=[season_df['points_won'].idxmax()]

    

    #change player's 'Mvp?' entry to yes

    nba['Mvp?'][index]='Yes'
nba
#move this new column next to mvp voting data



#save column,remove it from dataframe, then insert it where we want it

save=nba['Mvp?']

nba.drop(labels=['Mvp?'], axis=1, inplace = True)

nba.insert(10, 'Mvp?', save)

nba
nba['Mvp?'].value_counts()
len(nba['season'].value_counts())
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor



#blank dataframe that we will add to

predicted_df=pd.DataFrame()



#create model for each season

for season in nba['season'].value_counts().index:

    

    #isolate season data

    season_df=nba[nba['season'].isin([season])]

    y=season_df['award_share']

    features=['per', 'ts_pct', 'usg_pct', 'g', 'mp_per_g', 'pts_per_g', 'trb_per_g',

       'ast_per_g', 'stl_per_g', 'blk_per_g', 'fg_pct', 'fg3_pct', 'ft_pct',

       'ws', 'ws_per_48','win_pct']

    X=season_df[features]

    train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1, test_size=0.4)

    basic_model = DecisionTreeRegressor(random_state=1)

    basic_model.fit(train_X, train_y)

    predictions=basic_model.predict(val_X)

    

    #modify test dataframe to show predictions too

    val_Xdf=val_X

    

    #add column of predictions

    val_Xdf['Prediction']=predictions

    

    #add the correct values

    val_Xdf['award_share']=val_y

    

    #add column for the season

    val_Xdf['season']=season

    

    #add column for player name- this is a bit tricky because we need the index of player as it is in the 'nba' dataframe

    #resetting index creates a column of the original indices that we can use to refer to the indices in the 'nba' dataframe

    val_Xdf['player']=[season_df['player'][index] for index in val_Xdf.reset_index()['index']]

    

    #same methodology here

    val_Xdf['Mvp?']=[season_df['Mvp?'][index] for index in val_Xdf.reset_index()['index']]

    

    #add this dataframe to the dataframe of all the seasons' predictions

    predicted_df=predicted_df.append(val_Xdf)
predicted_df
predicted_df[predicted_df['season'].isin(['2017-18'])]
features=['per', 'ts_pct', 'usg_pct', 'bpm', 'g', 'mp_per_g', 'pts_per_g', 'trb_per_g',

       'ast_per_g', 'stl_per_g', 'blk_per_g', 'fg_pct', 'fg3_pct', 'ft_pct',

       'ws', 'ws_per_48','win_pct']



#have to specify train test split so that we can group seasons together

#make first 30 seasons the training data and the last 8 the testing data

training_seasons=['1980-81', '1981-82', '1984-85', '1982-83', '1998-99', '1996-97',

       '1990-91', '1997-98', '1988-89', '2001-02', '1985-86', '2000-01',

       '2007-08', '1991-92', '1993-94', '2006-07', '1986-87', '1995-96',

       '1987-88', '2013-14', '1999-00', '2012-13', '2004-05', '2003-04',

       '1994-95', '2011-12', '2009-10', '1983-84', '1989-90', '1992-93']

testing_seasons=['2017-18', '2010-11', '2002-03', '2014-15', '2008-09', '2005-06',

       '2016-17', '2015-16']



#training data

training_data=nba[nba['season'].isin(training_seasons)]

train_X=training_data[features]

train_y=training_data['award_share']



#testing data

testing_data=nba[nba['season'].isin(testing_seasons)]

val_X=testing_data[features]

val_y=testing_data['award_share']



basic_model = DecisionTreeRegressor(random_state=1)

basic_model.fit(train_X, train_y)

predictions=basic_model.predict(val_X)
#put testing data and predictions into new dataframe

predicted_df=pd.DataFrame()

val_Xdf=pd.DataFrame(val_X)

val_Xdf['Prediction']=predictions

val_Xdf['award_share']=val_y

val_Xdf['season']=[nba['season'][index] for index in val_Xdf.reset_index()['index']]

val_Xdf['player']=[nba['player'][index] for index in val_Xdf.reset_index()['index']]

val_Xdf['Mvp?']=[nba['Mvp?'][index] for index in val_Xdf.reset_index()['index']]

predicted_df=predicted_df.append(val_Xdf)
predicted_df
#create column indicating whether player actually won the mvp

predicted_df['Mvp prediction']='No'

for season in predicted_df['season'].value_counts().index:

    season_df=predicted_df[predicted_df['season'].isin([season])]

    index=season_df['Prediction'].idxmax()

    mvp=predicted_df['player'][index]

    

    #will only change for the mvp winner, otherwise all others players will be 'no'

    predicted_df['Mvp prediction'][index]='Yes'
predicted_df
predicted_df[predicted_df['season'].isin(['2002-03'])]
predicted_df[predicted_df['season'].isin(['2002-03']) & predicted_df['player'].isin(['Tim Duncan','Dirk Nowitzki'])]
predicted_list=[]

for season in predicted_df['season'].value_counts().index:

    season_df=predicted_df[predicted_df['season'].isin([season])]

    predicted_list.append(season_df)
total=0

for df in predicted_list:

    if df['Mvp?'].equals(df['Mvp prediction'])==True:

        total+=1

total/len(predicted_list)
wrong_seasons=[]

right_seasons=[]

for df in predicted_list:

    if df['Mvp?'].equals(df['Mvp prediction'])==False:

        wrong_seasons.append(df.reset_index()['season'][0])

    else:

        right_seasons.append(df.reset_index()['season'][0])
wrong_seasons
predicted_df[predicted_df['season'].isin(['2017-18'])]
predicted_df[predicted_df['season'].isin(['2010-11'])]
predicted_df[predicted_df['season'].isin(['2005-06'])]
right_seasons
predicted_df[predicted_df['season'].isin(['2015-16'])]
nba.columns
features=['bpm', 'g', 'mp_per_g','ws', 'ws_per_48','win_pct']

#have to specify train test split so that we can group seasons together

#make first 30 seasons the training data and the last 8 the testing data

training_seasons=['1980-81', '1981-82', '1984-85', '1982-83', '1998-99', '1996-97',

       '1990-91', '1997-98', '1988-89', '2001-02', '1985-86', '2000-01',

       '2007-08', '1991-92', '1993-94', '2006-07', '1986-87', '1995-96',

       '1987-88', '2013-14', '1999-00', '2012-13', '2004-05', '2003-04',

       '1994-95', '2011-12', '2009-10', '1983-84', '1989-90', '1992-93']

testing_seasons=['2017-18', '2010-11', '2002-03', '2014-15', '2008-09', '2005-06',

       '2016-17', '2015-16']

# train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1, test_size=0.4)

train_X=nba[nba['season'].isin(training_seasons)][features]

train_y=nba[nba['season'].isin(training_seasons)]['award_share']

val_X=nba[nba['season'].isin(testing_seasons)][features]

val_y=nba[nba['season'].isin(testing_seasons)]['award_share']



basic_model = DecisionTreeRegressor(random_state=1)

basic_model.fit(train_X, train_y)

predictions=basic_model.predict(val_X)
predicted_df=pd.DataFrame()

val_Xdf=pd.DataFrame(val_X)

val_Xdf['Prediction']=predictions

val_Xdf['award_share']=val_y

val_Xdf['season']=[nba['season'][index] for index in val_Xdf.reset_index()['index']]

val_Xdf['player']=[nba['player'][index] for index in val_Xdf.reset_index()['index']]

val_Xdf['Mvp?']=[nba['Mvp?'][index] for index in val_Xdf.reset_index()['index']]

predicted_df=predicted_df.append(val_Xdf)
#create column indicating whether player actually won the mvp

predicted_df['Mvp prediction']=''

for season in predicted_df['season'].value_counts().index:

    season_df=predicted_df[predicted_df['season'].isin([season])]

    mvp=predicted_df['player'][season_df['Prediction'].idxmax()]

    for player in season_df['player']:

        row=predicted_df[predicted_df['season'].isin([season]) & predicted_df['player'].isin([player])].index[0]

        if player==mvp:

            predicted_df['Mvp prediction'][row]='Yes'

        else:

            predicted_df['Mvp prediction'][row]='No'
predicted_list=[]

for season in predicted_df['season'].value_counts().index:

    season_df=predicted_df[predicted_df['season'].isin([season])]

    predicted_list.append(season_df)
wrong_seasons=[]

right_seasons=[]

for df in predicted_list:

    if df['Mvp?'].equals(df['Mvp prediction'])==False:

        wrong_seasons.append(df.reset_index()['season'][0])

    else:

        right_seasons.append(df.reset_index()['season'][0])
wrong_seasons
predicted_df[predicted_df['season'].isin(['2005-06'])]
nba.columns
features=['fga', 'fg3a', 'fta', 'per', 'ts_pct', 'usg_pct',

       'g', 'mp_per_g', 'pts_per_g', 'trb_per_g',

       'ast_per_g', 'stl_per_g', 'blk_per_g', 'fg_pct', 'fg3_pct', 'ft_pct']

#have to specify train test split so that we can group seasons together

#make first 30 seasons the training data and the last 8 the testing data

training_seasons=['1980-81', '1981-82', '1984-85', '1982-83', '1998-99', '1996-97',

       '1990-91', '1997-98', '1988-89', '2001-02', '1985-86', '2000-01',

       '2007-08', '1991-92', '1993-94', '2006-07', '1986-87', '1995-96',

       '1987-88', '2013-14', '1999-00', '2012-13', '2004-05', '2003-04',

       '1994-95', '2011-12', '2009-10', '1983-84', '1989-90', '1992-93']

testing_seasons=['2017-18', '2010-11', '2002-03', '2014-15', '2008-09', '2005-06',

       '2016-17', '2015-16']

# train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1, test_size=0.4)

train_X=nba[nba['season'].isin(training_seasons)][features]

train_y=nba[nba['season'].isin(training_seasons)]['award_share']

val_X=nba[nba['season'].isin(testing_seasons)][features]

val_y=nba[nba['season'].isin(testing_seasons)]['award_share']



basic_model = DecisionTreeRegressor(random_state=1)

basic_model.fit(train_X, train_y)

predictions=basic_model.predict(val_X)
predicted_df=pd.DataFrame()

val_Xdf=pd.DataFrame(val_X)

val_Xdf['Prediction']=predictions

val_Xdf['award_share']=val_y

val_Xdf['season']=[nba['season'][index] for index in val_Xdf.reset_index()['index']]

val_Xdf['player']=[nba['player'][index] for index in val_Xdf.reset_index()['index']]

val_Xdf['Mvp?']=[nba['Mvp?'][index] for index in val_Xdf.reset_index()['index']]

predicted_df=predicted_df.append(val_Xdf)
#create column indicating whether player actually won the mvp

predicted_df['Mvp prediction']=''

for season in predicted_df['season'].value_counts().index:

    season_df=predicted_df[predicted_df['season'].isin([season])]

    mvp=predicted_df['player'][season_df['Prediction'].idxmax()]

    for player in season_df['player']:

        row=predicted_df[predicted_df['season'].isin([season]) & predicted_df['player'].isin([player])].index[0]

        if player==mvp:

            predicted_df['Mvp prediction'][row]='Yes'

        else:

            predicted_df['Mvp prediction'][row]='No'
predicted_list=[]

for season in predicted_df['season'].value_counts().index:

    season_df=predicted_df[predicted_df['season'].isin([season])]

    predicted_list.append(season_df)
wrong_seasons=[]

right_seasons=[]

for df in predicted_list:

    if df['Mvp?'].equals(df['Mvp prediction'])==False:

        wrong_seasons.append(df.reset_index()['season'][0])

    else:

        right_seasons.append(df.reset_index()['season'][0])
wrong_seasons
predicted_df[predicted_df['season'].isin(['2005-06'])]
for row in range(len(nba)):

    if nba['Mvp?'][row]=='Yes':

        nba['Mvp?'][row]=True

    else:

        nba['Mvp?'][row]=False

nba['Mvp?']
nba['Mvp?'].value_counts()
nba[nba['season'].isin(['1980-81'])]
features=['bpm', 'g', 'mp_per_g','ws', 'ws_per_48','win_pct']

#have to specify train test split so that we can group seasons together

#make first 30 seasons the training data and the last 8 the testing data

training_seasons=['1980-81', '1981-82', '1984-85', '1982-83', '1998-99', '1996-97',

       '1990-91', '1997-98', '1988-89', '2001-02', '1985-86', '2000-01',

       '2007-08', '1991-92', '1993-94', '2006-07', '1986-87', '1995-96',

       '1987-88', '2013-14', '1999-00', '2012-13', '2004-05', '2003-04',

       '1994-95', '2011-12', '2009-10', '1983-84', '1989-90', '1992-93']

testing_seasons=['2017-18', '2010-11', '2002-03', '2014-15', '2008-09', '2005-06',

       '2016-17', '2015-16']

# train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1, test_size=0.4)

train_X=nba[nba['season'].isin(training_seasons)][features]

train_y=nba[nba['season'].isin(training_seasons)]['Mvp?']

val_X=nba[nba['season'].isin(testing_seasons)][features]

val_y=nba[nba['season'].isin(testing_seasons)]['Mvp?']



basic_model = DecisionTreeRegressor(random_state=1)

basic_model.fit(train_X, train_y)

predictions=basic_model.predict(val_X)
predicted_df=pd.DataFrame()

val_Xdf=pd.DataFrame(val_X)

val_Xdf['Prediction']=predictions

val_Xdf['Mvp?']=val_y

val_Xdf['season']=[nba['season'][index] for index in val_Xdf.reset_index()['index']]

val_Xdf['player']=[nba['player'][index] for index in val_Xdf.reset_index()['index']]

# val_Xdf['Mvp?']=[nba['Mvp?'][index] for index in val_Xdf.reset_index()['index']]

predicted_df=predicted_df.append(val_Xdf)
predicted_df
#create column indicating whether player actually won the mvp

predicted_df['Mvp prediction']=''

for index in predicted_df.reset_index()['index']:

    if predicted_df['Prediction'][index]==True:

        predicted_df['Mvp prediction'][index]='Yes'

    else:

        predicted_df['Mvp prediction'][index]='No'
predicted_df
predicted_df['Mvp prediction'].value_counts()
predicted_list=[]

for season in predicted_df['season'].value_counts().index:

    season_df=predicted_df[predicted_df['season'].isin([season])]

    predicted_list.append(season_df)
#can't compare boolean to integer columns, can only compare one by one

#have to see if any cell doesn't match up

wrong_seasons=[]

right_seasons=[]

for df in predicted_list:

    df=df.reset_index()

    for row in range(len(df)):

        if df['Mvp?'][row]!=df['Prediction'][row]:

            wrong_seasons.append(df['season'][row])

#         wrong_seasons.append(df.reset_index()['season'][0])

#     else:

#         right_seasons.append(df.reset_index()['season'][0])
wrong_seasons
predicted_df[predicted_df['season'].isin(['2015-16'])]
features=['fga', 'fg3a', 'fta', 'per', 'ts_pct', 'usg_pct',

       'g', 'mp_per_g', 'pts_per_g', 'trb_per_g',

       'ast_per_g', 'stl_per_g', 'blk_per_g', 'fg_pct', 'fg3_pct', 'ft_pct']

training_seasons=['1980-81', '1981-82', '1984-85', '1982-83', '1998-99', '1996-97',

       '1990-91', '1997-98', '1988-89', '2001-02', '1985-86', '2000-01',

       '2007-08', '1991-92', '1993-94', '2006-07', '1986-87', '1995-96',

       '1987-88', '2013-14', '1999-00', '2012-13', '2004-05', '2003-04',

       '1994-95', '2011-12', '2009-10', '1983-84', '1989-90', '1992-93']

testing_seasons=['2017-18', '2010-11', '2002-03', '2014-15', '2008-09', '2005-06',

       '2016-17', '2015-16']

# train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1, test_size=0.4)

train_X=nba[nba['season'].isin(training_seasons)][features]

train_y=nba[nba['season'].isin(training_seasons)]['Mvp?']

val_X=nba[nba['season'].isin(testing_seasons)][features]

val_y=nba[nba['season'].isin(testing_seasons)]['Mvp?']



basic_model = DecisionTreeRegressor(random_state=1)

basic_model.fit(train_X, train_y)

predictions=basic_model.predict(val_X)
predicted_df=pd.DataFrame()

val_Xdf=pd.DataFrame(val_X)

val_Xdf['Prediction']=predictions

val_Xdf['Mvp?']=val_y

val_Xdf['season']=[nba['season'][index] for index in val_Xdf.reset_index()['index']]

val_Xdf['player']=[nba['player'][index] for index in val_Xdf.reset_index()['index']]

predicted_df=predicted_df.append(val_Xdf)
predicted_df['Mvp prediction']=''

for index in predicted_df.reset_index()['index']:

    if predicted_df['Prediction'][index]==True:

        predicted_df['Mvp prediction'][index]='Yes'

    else:

        predicted_df['Mvp prediction'][index]='No'
predicted_list=[]

for season in predicted_df['season'].value_counts().index:

    season_df=predicted_df[predicted_df['season'].isin([season])]

    predicted_list.append(season_df)
wrong_seasons=[]

for df in predicted_list:

    df=df.reset_index()

    for row in range(len(df)):

        if df['Mvp?'][row]!=df['Prediction'][row]:

            wrong_seasons.append(df['season'][row])
wrong_seasons
predicted_df[predicted_df['season'].isin(['2002-03'])]
def predict_model(features,metric):

    training_seasons=['1980-81', '1981-82', '1984-85', '1982-83', '1998-99', '1996-97',

           '1990-91', '1997-98', '1988-89', '2001-02', '1985-86', '2000-01',

           '2007-08', '1991-92', '1993-94', '2006-07', '1986-87', '1995-96',

           '1987-88', '2013-14', '1999-00', '2012-13', '2004-05', '2003-04',

           '1994-95', '2011-12', '2009-10', '1983-84', '1989-90', '1992-93']

    testing_seasons=['2017-18', '2010-11', '2002-03', '2014-15', '2008-09', '2005-06',

           '2016-17', '2015-16']

    train_X=nba[nba['season'].isin(training_seasons)][features]

    train_y=nba[nba['season'].isin(training_seasons)][metric]

    val_X=nba[nba['season'].isin(testing_seasons)][features]

    val_y=nba[nba['season'].isin(testing_seasons)][metric]



    basic_model = DecisionTreeRegressor(random_state=1)

    basic_model.fit(train_X, train_y)

    return basic_model.predict(val_X)
def get_val_X(features):

    return nba[nba['season'].isin(testing_seasons)][features]
def get_val_y(metric):

    return nba[nba['season'].isin(testing_seasons)][metric]
def get_df(predictions,val_X,val_y):

    predicted_df=pd.DataFrame()

    val_Xdf=pd.DataFrame(val_X)

    val_Xdf['Prediction']=predictions

    val_Xdf['Mvp?']=[nba['Mvp?'][index] for index in val_Xdf.reset_index()['index']]

    val_Xdf['season']=[nba['season'][index] for index in val_Xdf.reset_index()['index']]

    val_Xdf['player']=[nba['player'][index] for index in val_Xdf.reset_index()['index']]

    return predicted_df.append(val_Xdf)
def create_list(df):

    predicted_list=[]

    for season in df['season'].value_counts().index:

        season_df=df[df['season'].isin([season])]

        predicted_list.append(season_df)

    return predicted_list
features=['bpm', 'g', 'mp_per_g','ws', 'ws_per_48','win_pct']

predictions=predict_model(features,'votes_first')

p=get_df(predictions,get_val_X(features),get_val_y('votes_first'))

p
#create empty 'Mvp prediction' column that can be modified

p['Mvp prediction']='No'



#for every season

for season in p['season'].value_counts().index:

    

    #isolate data from that season, reset index

    season_df=p[p['season'].isin([season])]

    

    #find index player with most first place votes

    winner=season_df['Prediction'].idxmax()

    

    #go through indices of full dataframe by calling 'index' column

    p['Mvp prediction'][winner]='Yes'          
p
list=create_list(p)

wrong_seasons=[]

right_seasons=[]

for df in list:

    if df[df['Mvp?'].isin([True])].reset_index()['player'][0]==df[df['Mvp prediction'].isin(['Yes'])].reset_index()['player'][0]:

        right_seasons.append(df.reset_index()['season'][0])

    else:

        wrong_seasons.append(df.reset_index()['season'][0])
wrong_seasons
def get_season(df,season):

    return df[df['season'].isin([season])]
features=['fga', 'fg3a', 'fta', 'per', 'ts_pct', 'usg_pct',

       'g', 'mp_per_g', 'pts_per_g', 'trb_per_g',

       'ast_per_g', 'stl_per_g', 'blk_per_g', 'fg_pct', 'fg3_pct', 'ft_pct']

predictions=predict_model(features,'votes_first')

p=get_df(predictions,get_val_X(features),get_val_y('votes_first'))

p['Mvp prediction']='No'

for season in p['season'].value_counts().index:

    season_df=p[p['season'].isin([season])]

    winner=season_df['Prediction'].idxmax()

    p['Mvp prediction'][winner]='Yes'

list=create_list(p)

wrong_seasons=[]

right_seasons=[]

for df in list:

    if df[df['Mvp?'].isin([True])].reset_index()['player'][0]==df[df['Mvp prediction'].isin(['Yes'])].reset_index()['player'][0]:

        right_seasons.append(df.reset_index()['season'][0])

    else:

        wrong_seasons.append(df.reset_index()['season'][0])
wrong_seasons