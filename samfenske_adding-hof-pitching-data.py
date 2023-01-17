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
pitching=pd.read_csv('/kaggle/input/pitchers/pitching.csv').drop(columns='Unnamed: 0')

pitching
hof=pd.read_csv('/kaggle/input/baseball-databank/HallOfFame.csv')

hof
hof=hof[hof['category'].isin(['Player'])]

hof=hof.drop(columns=['needed_note'])

hof['percent']=hof['votes']/hof['ballots']

hof['threshold']=hof['needed']/hof['ballots']

hof
hof_clean=pd.DataFrame()

for player in hof['playerID'].value_counts().index:

    playerdf=pd.DataFrame()

    df=hof[hof['playerID'].isin([player])].reset_index()

    year=df.max()['yearid']

    attempts=len(df)

    if df['inducted'].str.contains('Y').any():

        inducted=df[df['inducted'].isin(['Y'])].reset_index()

        percent=inducted['percent'][0]

        threshold=inducted['threshold'][0]

        votedBy=inducted['votedBy'][0]

        playerdf=playerdf.append({'year':year,'playerID':player,'percent':percent,'threshold':

                                threshold,'years':attempts,'votedBy':votedBy,'inducted':'Y'},ignore_index=True)

    else:

        max_percent=df.max()['percent']

        threshold=df['threshold'][df['percent'].idxmax()]

        votedBy=df['votedBy'].value_counts().index[0]

        playerdf=playerdf.append({'year':year,'playerID':player,'percent':max_percent,'threshold':

                                threshold,'years':attempts,'votedBy':votedBy,'inducted':'N'},ignore_index=True)

    hof_clean=hof_clean.append(playerdf)
hof_clean
combined=pitching.join(hof_clean.set_index('playerID'), on='playerID')

combined
clean=combined[-combined['inducted'].isnull()]

clean
#see if any players need to be weeded out, those who pitched in less than 200 games

clean[clean['G']<200]
clean.to_csv('pitching2.csv')