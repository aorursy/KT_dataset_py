# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.



from matplotlib import pyplot as plt

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from collections import defaultdict

import seaborn as sns



import pandas as pd

pd.set_option('display.max_columns', None)
debates = pd.read_csv("../input/primary_debates_cleaned.csv")

debates = debates[debates.Speaker != 'AUDIENCE']

debates = debates[debates.Speaker != 'CANDIDATES']

debates = debates[debates.Speaker != 'BELL']

debates = debates[debates.Speaker != 'UNKNOWN']

debates = debates[debates.Speaker != 'OTHER']



debates_list = []

debates_location = []

for location in debates.Location.unique():

    debates_list.append(debates[debates.Location == location])

    debates_location.append(location)

debates_location
def score_speaker(speaker_df):

    df_dict = {}

    for i,response in speaker_df.iterrows():

        scorer = SentimentIntensityAnalyzer()

        scores = scorer.polarity_scores(response.Text)

        df_dict[i] = scores

    df = pd.DataFrame.from_dict(df_dict)

    df = df.T

    diff_score = df['pos'] - df['neg']

    return diff_score
def compile_dicts(debates_location, debates_list):

    master_scores_dict = defaultdict(list)

    master_debates_means = []

    for location in range(len(debates_location)):

        speaker_scores = {}

        print(debates_location[location])

        debate = debates_list[location]

        # Print the number of times a candidate spoke

        #speaker_answers = debate.Speaker.value_counts()

        #speaker_answers.plot.bar()

        #plt.xticks(rotation=90)

        #plt.show()

        # Assemble the polarity scores for each time the speaker spoke

        for speaker in debate.Speaker.unique():

            polarity_score = score_speaker(debate[debate.Speaker == speaker])

            speaker_scores[speaker] = polarity_score

            master_scores_dict[speaker].append(polarity_score)

        speaker_scores_df = pd.DataFrame.from_dict(speaker_scores)

        # Return the mean and standard deviation of all the scores

        speaker_mean_std_df = pd.DataFrame([speaker_scores_df.mean(),speaker_scores_df.std()],index=['mean','std'])



        debate_mean_df = pd.DataFrame([speaker_scores_df.T.mean()])

        mean = debate_mean_df.mean(axis=1)

        master_debates_means.append(mean)



        print("\n<<<<<<<>>>>>\n")

        print("Mean: " + str(mean.values[0]))

        print(speaker_mean_std_df)

        print("\n<<<<<<>>>>>>\n")

    return master_scores_dict, master_debates_means
debater_scores, debate_means = compile_dicts(debates_location, debates_list)
debate_means_df = pd.DataFrame.from_dict(debate_means)

chart = debate_means_df.plot()
plotting_df = pd.DataFrame(dtype='float')

for speaker, scores in debater_scores.items():

    s_scores = pd.Series(scores[0])

    s_scores = s_scores.rename(speaker)

    plotting_df = plotting_df.append(s_scores)



plotting_df = plotting_df.T

means = plotting_df.mean(axis=0)

std = plotting_df.std(axis=0)
plotting_df[['Cruz','Kasich','Clinton','Trump','Sanders']].plot.box()
print(means[['Cruz','Kasich','Clinton','Trump','Sanders']])

print(std[['Cruz','Kasich','Clinton','Trump','Sanders']])
plotting_df[['Rubio','Kasich','Trump','Cruz','Bush','Carson','Fiorina','Christie','Graham','Huckabee','Jindal','Perry','Paul','Walker','Santorum']].plot.box()

plt.xticks(rotation=90)
print(means[['Rubio','Kasich','Trump','Cruz','Bush','Carson','Fiorina','Christie','Graham','Huckabee','Jindal','Perry','Walker','Santorum']])
print(std[['Rubio','Kasich','Trump','Cruz','Bush','Carson','Fiorina','Christie','Graham','Huckabee','Jindal','Perry','Walker','Santorum']])