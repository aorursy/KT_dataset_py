import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



pd.set_option('display.max_columns', None) 



TEDonly = pd.read_csv("../input/TEDonly_final.csv")

TEDonly_speakers = pd.read_csv("../input/TEDonly_speakers_final.csv")

TEDplus = pd.read_csv("../input/TEDplus_final.csv")

TEDplus_speakers = pd.read_csv("../input/TEDplus_speakers_final.csv")
TEDonly.shape
TEDonly.tail(5)
TEDonly.columns
import time

import datetime



TEDonly['duration'] = pd.to_datetime(TEDonly.duration, format="%H:%M:%S")

TEDonly['duration(secs)'] = [(t-datetime.datetime(1900,1,1)).total_seconds() for t in TEDonly.duration]

TEDonly.drop(columns=['duration'], inplace=True)
TEDonly.head(2)
duration_describe = TEDonly['duration(secs)'].describe()



for t in duration_describe.keys(): 

    if not t=='count':

        duration_describe[t]/=60

        

duration_describe
TEDonly_speakers.shape
TEDonly_speakers.head(2)
TEDonly_speakers.columns
TEDonly_speakers.loc[TEDonly_speakers.index[0], 'speaker1_profile']
import re



def find_sex(profile):

    '''

    Finds the sex of the person by checking the pronouns used in his/her

    description.

    

    :param profile: Profile of the speaker

    :returns: `MALE` or `FEMALE` if profile is given else NAN

    '''

    if not pd.isnull(profile):

        n_male_pronouns = len(re.findall("\she\s|\sHe\s|\shis\s|\sHis\s", profile))

        n_female_pronouns = len(re.findall("\sshe\s|\sShe\s|\sher\s|\sHer\s", profile))

        if n_male_pronouns > n_female_pronouns:

            return "MALE"

        else:

            return "FEMALE"

    else:

        return np.nan
TEDonly_speakers['speaker1_sex'] = TEDonly_speakers.apply(lambda row: find_sex(row['speaker1_profile']), axis=1)

TEDonly_speakers['speaker2_sex'] = TEDonly_speakers.apply(lambda row: find_sex(row['speaker2_profile']), axis=1)

TEDonly_speakers['speaker3_sex'] = TEDonly_speakers.apply(lambda row: find_sex(row['speaker3_profile']), axis=1)

TEDonly_speakers['speaker4_sex'] = TEDonly_speakers.apply(lambda row: find_sex(row['speaker4_profile']), axis=1)
sex_dist = pd.DataFrame(columns=['MALE', 'FEMALE'])

sex_dist.loc['Speaker1', :] = TEDonly_speakers['speaker1_sex'].value_counts()

sex_dist.loc['Speaker2', :] = TEDonly_speakers['speaker2_sex'].value_counts()

sex_dist.loc['Speaker3', :] = TEDonly_speakers['speaker3_sex'].value_counts()

sex_dist.loc['Speaker4', :] = TEDonly_speakers['speaker4_sex'].value_counts()

sex_dist.fillna(0, inplace=True)
sex_dist
sex_dist.plot.bar(figsize=(16,8))
TEDonly_speakers.loc[:, 'published'] = pd.to_datetime(TEDonly_speakers['published'])



TEDonly_speakers.loc[:, 'publish_year'] = TEDonly_speakers['published'].dt.year



sex_dist_yearwise = TEDonly_speakers.groupby(['speaker1_sex', 'publish_year'], as_index=True).count().iloc[:, :1]
sex_dist_yearwise.unstack().T.plot.bar(figsize=(16, 8))