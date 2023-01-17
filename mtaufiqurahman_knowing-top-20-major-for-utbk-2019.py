import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
major = pd.read_csv("../input/indonesia-college-entrance-examination-utbk-2019/majors.csv", index_col = 0)



major.head()
univ = pd.read_csv("../input/indonesia-college-entrance-examination-utbk-2019/universities.csv", index_col = 0)



univ.head()
sci_score = pd.read_csv("../input/indonesia-college-entrance-examination-utbk-2019/score_science.csv", index_col = 0)



sci_score.head()
hum_score = pd.read_csv("../input/indonesia-college-entrance-examination-utbk-2019/score_humanities.csv", index_col = 0)



hum_score.head()
sci_score.describe()
hum_score.describe()
sum(sci_score['id_user'].isin(hum_score['id_user']))
#processing some sci_score datas

sci_score['score_total'] = sci_score[['score_mat', 'score_fis', 'score_kim', 'score_bio', 'score_kpu', 'score_kua', 'score_ppu', 'score_kmb']].apply(sum, axis = 1)

sci_score['score_mean'] = sci_score[['score_mat', 'score_fis', 'score_kim', 'score_bio', 'score_kpu', 'score_kua', 'score_ppu', 'score_kmb']].apply(np.mean, axis = 1)

sci_score['test_type'] = 'science'



sci_score.head()
#processing some hum_score datas

hum_score['score_total'] = hum_score[['score_mat', 'score_geo', 'score_sej', 'score_sos', 'score_eko', 'score_kpu', 'score_kua', 'score_ppu', 'score_kmb']].apply(sum, axis = 1)

hum_score['score_mean'] = hum_score[['score_mat', 'score_geo', 'score_sej', 'score_sos', 'score_eko', 'score_kpu', 'score_kua', 'score_ppu', 'score_kmb']].apply(np.mean, axis = 1)

hum_score['test_type'] = 'humanities'



hum_score.head()
major_univ = pd.merge(major, univ, on = 'id_university', how = 'left')



major_univ.set_index('id_major', inplace = True)

major_univ['utbk_capacity'] = 0.4 * major_univ['capacity']

major_univ['utbk_capacity'] = major_univ['utbk_capacity'].apply(int)

major_univ['passed_count'] = 0



major_univ.head()
#merging sci_score with hum_score

test_score = pd.merge(sci_score, hum_score, how = 'outer')

#processing some test_score datas

test_score = pd.merge(test_score, major_univ[['major_name', 'university_name', 'type']], left_on = 'id_first_major',

                      right_on = major_univ.index, how = 'left')

test_score = pd.merge(test_score, major_univ[['major_name', 'university_name', 'type']], left_on = 'id_second_major',

                      right_on = major_univ.index, how = 'left', suffixes = ('_1', '_2'))

test_score.set_index('id_user', inplace = True)

test_score['passing_choice'] = 0

test_score['passing_major'] = ''

test_score['passsing_universities'] = ''

test_score['note'] = ''



test_score.head()
sns.set_style('darkgrid')

fig = plt.figure(figsize = (12,8))

test_score[test_score['test_type'] == 'science']['score_mean'].hist(bins = 50, alpha = 0.5, label = 'science')

test_score[test_score['test_type'] == 'humanities']['score_mean'].hist(bins = 50, alpha = 0.5, label = 'humanities')

plt.legend()

plt.title('Test score mean')
def major_univ_check (cols):

    major_id = str(cols[0])

    univ_id = str(cols[1])

    major_type = cols[2]

    test_type = cols[3]

    

    if (major_type == test_type) & (major_id[:len(univ_id)] == univ_id):

        return True

    else:

        return False
#check whether test_score has any invalid data

drop_index = test_score[(~test_score['id_first_major'].isin(major_univ.index)) & 

                        (~test_score['id_second_major'].isin(major_univ.index))].index

if len(drop_index) != 0:

    test_score.loc[drop_index, 'note'] = 'Error: invalid major/university'

    test_score.loc[drop_index, 'passing_major'] = '-'

    test_score.loc[drop_index, 'passsing_universities'] = '-'



test_score['major_1_check'] = test_score[['id_first_major', 'id_first_university', 'type_1', 'test_type']].apply(major_univ_check, axis = 1)

test_score['major_2_check'] = test_score[['id_second_major', 'id_second_university', 'type_2', 'test_type']].apply(major_univ_check, axis = 1)

false_major_index = test_score[(test_score['major_1_check'] == False) & (test_score['major_2_check'] == False)].index

test_score.loc[false_major_index, 'note'] = 'Error: invalid major/university'

test_score.loc[false_major_index, 'passing_major'] = '-'

test_score.loc[false_major_index, 'passsing_universities'] = '-'



test_score.drop(['type_1', 'type_2'], axis = 1, inplace = True)

test_score.head()
#ranking total_score

test_score['major_1_rank'] = np.NaN

test_score['major_2_rank'] = np.NaN



for major_id in major_univ.index:

    df_temp = test_score[test_score['note'] == ''].copy().reset_index()

    choice_1 = df_temp[(df_temp['major_1_check'] == True) &

                       (df_temp['id_first_major'] == major_id)][['id_user', 'score_total', 'id_first_major']]

    choice_2 = df_temp[(df_temp['major_2_check'] == True) &

                       (df_temp['id_second_major'] == major_id)][['id_user', 'score_total', 'id_second_major']]

    

    major_choice = pd.merge(choice_1, choice_2, how='outer')

    major_choice['score_total'] = major_choice['score_total'].rank(method='max', ascending=False)

    major_choice.set_index('id_user', inplace=True)

    

    for uid in major_choice.index:

        if ~np.isnan(major_choice.loc[uid, 'id_first_major']):

            test_score.loc[uid, 'major_1_rank'] = major_choice.loc[uid, 'score_total'].copy()

    

        if ~np.isnan(major_choice.loc[uid, 'id_second_major']):

            test_score.loc[uid, 'major_2_rank'] = major_choice.loc[uid, 'score_total'].copy()

        

test_score.head()
#check passing_major

test_score.sort_values('score_mean', ascending = False, inplace = True)

checked_rank = major_univ['utbk_capacity'].copy()

for uid in test_score[test_score['note']==''].index:

    major_1 = test_score.loc[uid, 'id_first_major']

    major_2 = test_score.loc[uid, 'id_second_major']

    if ((test_score.loc[uid, 'major_1_check'] == True) & (test_score.loc[uid, 'major_1_rank'] < checked_rank.loc[major_1]) & 

        (major_univ.loc[major_1, 'passed_count'] < major_univ.loc[major_1, 'utbk_capacity'])):

        test_score.loc[uid, 'passing_choice'] = 1

        test_score.loc[uid, 'note'] = 'Pass: ' + ' - '.join([test_score.loc[uid, 'major_name_1'],

                                                             test_score.loc[uid,'university_name_1']])

        test_score.loc[uid, 'passing_major'] = test_score.loc[uid, 'major_name_1']

        test_score.loc[uid, 'passsing_universities'] = test_score.loc[uid,'university_name_1']

        major_univ.loc[major_1, 'passed_count'] = major_univ.loc[major_1, 'passed_count'] + 1

        checked_rank.loc[major_1] = checked_rank.loc[major_1] + 1

        checked_rank.loc[major_2] = checked_rank.loc[major_2] + 1

    elif ((test_score.loc[uid, 'major_2_check'] == True) & (test_score.loc[uid, 'major_2_rank'] < checked_rank.loc[major_2]) &

          (major_univ.loc[major_2, 'passed_count'] < major_univ.loc[major_2, 'utbk_capacity'])):

        test_score.loc[uid, 'passing_choice'] = 2

        test_score.loc[uid, 'note'] = 'Pass: ' + ' - '.join([test_score.loc[uid, 'major_name_2'],

                                                             test_score.loc[uid,'university_name_2']])

        test_score.loc[uid, 'passing_major'] = test_score.loc[uid, 'major_name_2']

        test_score.loc[uid, 'passsing_universities'] = test_score.loc[uid,'university_name_2']

        major_univ.loc[major_2, 'passed_count'] = major_univ.loc[major_2, 'passed_count'] + 1

        checked_rank.loc[major_1] = checked_rank.loc[major_1] + 1

        checked_rank.loc[major_2] = checked_rank.loc[major_2] + 1

    else:

        test_score.loc[uid, 'note'] = 'Failed: not passing any major choices'

        test_score.loc[uid, 'passing_major'] = '-'

        test_score.loc[uid, 'passsing_universities'] = '-'



test_score.drop(['major_name_1', 'university_name_1', 'major_1_check', 'major_name_2', 'university_name_2', 'major_2_check'],

                axis = 1, inplace = True)

test_score
fig = plt.figure (figsize = (12, 6))

plt.title('Passing degree')

sns.countplot('passing_choice', data = test_score, hue = 'test_type')
test_score[test_score['test_type'] == 'science'].groupby(['passing_major', 'passsing_universities'])['score_mean'].min().sort_values(ascending = False).head(20)
test_score[test_score['test_type'] == 'humanities'].groupby(['passing_major', 'passsing_universities'])['score_mean'].min().sort_values(ascending = False).head(20)