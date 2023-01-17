# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
daily_life = pd.read_csv('/kaggle/input/european-union-lgbt-survey-2012/LGBT_Survey_DailyLife.csv')

daily_life.columns
question_set = list(dict.fromkeys(daily_life['question_label']))

question_code_set = list(dict.fromkeys(daily_life['question_code']))

for question in range(len(question_set)):

    print(question_code_set[question] + ' - ' + question_set[question])
survey_c1a = daily_life[daily_life['question_code'] == 'c1a_a']

survey_c1b = daily_life[daily_life['question_code'] == 'c1a_b']

survey_c1c = daily_life[daily_life['question_code'] == 'c1a_c']

survey_c1d = daily_life[daily_life['question_code'] == 'c1a_d']

frames = [survey_c1a, survey_c1b, survey_c1c, survey_c1d]

survey_c1 = pd.concat(frames)

survey_c1 = survey_c1[survey_c1['percentage'] != ':']

survey_c1 = survey_c1[survey_c1['CountryCode'] != 'Average']

percs = []

for perc in survey_c1['percentage']:

    percs.append(int(perc))

survey_c1['percentage'] = percs
answer_set = set(survey_c1['answer'])

answer_set

country_set = set(survey_c1['CountryCode'])

country_set
import matplotlib.pyplot as plt



unsure = survey_c1[survey_c1['answer'] == 'Don`t know']

frare = survey_c1[survey_c1['answer'] == 'Fairly rare']

fwide = survey_c1[survey_c1['answer'] == 'Fairly widespread']

vrare = survey_c1[survey_c1['answer'] == 'Very rare']

vwide = survey_c1[survey_c1['answer'] == 'Very widespread']



unsure_perc = np.mean(unsure['percentage'])

frare_perc = np.mean(frare['percentage'])

fwide_perc = np.mean(fwide['percentage'])

vrare_perc = np.mean(vrare['percentage'])

vwide_perc = np.mean(vwide['percentage'])



percs = [unsure_perc, frare_perc, fwide_perc, vrare_perc, vwide_perc]

labels = ['Don`t know', 'Fairly rare', 'Fairly widespread', 'Very rare', 'Very widespread']

tick_count = np.arange(5)



plt.bar(tick_count, percs, tick_label = labels, width = 0.2)

plt.show()
vwide_sorted = vwide.sort_values(by = 'percentage')

vwide_sorted
vwide_perc_mean = np.mean(vwide_sorted['percentage'])

high_vwide = vwide_sorted[vwide_sorted['percentage'] >= vwide_perc_mean]

high_vwide_countries = set(high_vwide['CountryCode'])

high_vwide_countries
top_ten = []

i = -1

while len(top_ten) < 10:

    top_country = list(high_vwide['CountryCode'])[i]

    if top_country not in top_ten:

        top_ten.append(top_country)

    i += -1



for i in range(len(top_ten)):

    print(str(i + 1) + ' - ' + top_ten[i])