# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 6)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

print('The followings are files in the dataset: ')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df_InfoUser = pd.read_csv('/kaggle/input/learning-activity-public-dataset-by-junyi-academy/Info_UserData.csv')
df_LogProblem = pd.read_csv('/kaggle/input/learning-activity-public-dataset-by-junyi-academy/Log_Problem.csv')
df_InfoContent = pd.read_csv('/kaggle/input/learning-activity-public-dataset-by-junyi-academy/Info_Content.csv')
df_InfoUser.head(5)
df_LogProblem.head(5)
df_InfoContent.head(5)
print('Total number of users:', len(df_InfoUser))
df_InfoUser = df_InfoUser.fillna('-1')
count_each_gender = df_InfoUser['gender'].value_counts()
count_each_gender
plt.title('Number of users per gender')
plt.bar(count_each_gender.index, count_each_gender.values)
plt.show()
# Select students from grade 1 to grade 6
df_elem = df_InfoUser[(df_InfoUser['user_grade'] > 0) & (df_InfoUser['user_grade'] < 7)]
df_elem.describe()
plt.plot(df_elem['points'].sort_values().reset_index(drop=True))

plt.title('Distribution of energy points for students in elementary school', fontsize=16)
plt.xlim((1, 50000))
plt.xlabel('Student sorted by energy point count', fontsize=10)
plt.ylim((0, 4100000))
plt.ylabel('Energy points', fontsize=10)
plt.grid()

plt.show()
# Let's pick a random exercise
df_LogProblem_first_ucid = df_LogProblem[df_LogProblem['ucid'] == df_LogProblem['ucid'][1]]
# Calculate number of problems done by each user
df_pcnt = df_LogProblem_first_ucid.groupby('uuid').size().reset_index(name='problem_cnt')
df_pcnt = df_pcnt.sort_values(by=['problem_cnt'])
df_pcnt = df_pcnt.reset_index()

# Sort and plot
pcnt_distribution = df_pcnt['problem_cnt'].value_counts()
pcnt_distribution = pcnt_distribution.sort_index()

plt.bar(pcnt_distribution.index, pcnt_distribution.values)

plt.title('Distribution of problem attempts for students in this exercise', fontsize=16)
plt.xlabel("Number of problems done in this exercise", fontsize=10)
plt.ylabel("User count", fontsize=10)
plt.xlim((0, 25))

plt.show()
print(df_InfoUser['user_city'].value_counts().head(5))
#TOP 5 : Taipei, New Taipei, Taichung, Taoyuan, Kaohsiung
# Lets randomly pick a user and an exercise and observe the learning process!
learning_path = df_LogProblem[(df_LogProblem['uuid'] == "AAITw26FaJFdy0VfpYXlUhEpJnYcjEucad09AXqKmUE=") &
                              (df_LogProblem['ucid'] == "FDFKlshYbN4rO93MtgimwfpEoKerSWp1RFhoSKWXHsY=")]
#sort by problem_number
learning_path = learning_path.sort_values(by=['problem_number']).reset_index()
learning_path = learning_path[['timestamp_TW', 'upid', 'problem_number', 'exercise_problem_repeat_session', 'is_correct']]
learning_path
uuidgb = df_LogProblem.groupby('uuid')
problem_cnt = uuidgb['uuid'].count()
total_time = uuidgb['total_sec_taken'].agg(np.sum)
mean_time_taken = total_time / problem_cnt
print("The mean of mean_time_taken", mean_time_taken.mean())
print("The std of mean_time_taken", mean_time_taken.std())
plt.plot(np.sort(mean_time_taken))

plt.title('Mean time for a user to finish a problem',fontsize=16)
plt.xlabel('Student sorted by average time taken', fontsize=10)
plt.ylabel('time (sec)', fontsize=10)
plt.ylim((0, 200))

plt.grid()
plt.show()
# There are definitely outliers in the time recorded
mean_time_taken[mean_time_taken > 1000]
correct_count = uuidgb['is_correct'].agg(np.sum)
correct_rate = correct_count / problem_cnt
print(f"mean : {correct_rate.mean()}\n std : {correct_rate.std()}\n min : {correct_rate.min()}\n max : {correct_rate.max()}")
plt.plot(np.sort(correct_rate))

plt.title('Distribution of correct rate', fontsize=16)
plt.xlabel('Student sorted by correct rate', fontsize=10)
plt.ylabel('correct rate', fontsize=10)

plt.grid()
plt.show()
# Lets look at the problems in elementary school students
ucid_chosen = df_InfoContent[df_InfoContent['learning_stage']=='elementary']['ucid']
filter_time = (df_LogProblem['timestamp_TW'] >= "2018-09-01") & (df_LogProblem['timestamp_TW'] < "2018-10-01")
filter_ucid = (df_LogProblem['ucid'].isin(ucid_chosen))
df_LogProblem_elem = df_LogProblem[filter_ucid & filter_time].groupby(['ucid']).size().reset_index(name='counts')
df_LogProblem_elem
plt.plot(np.sort(df_LogProblem_elem['counts']))
plt.axhline(df_LogProblem_elem['counts'].mean(), color = 'r',linestyle = '--')

plt.title('How many problems attempts were done for each exercise during 2018/09', fontsize=16)

plt.xlabel('Exercise', fontsize=10)
plt.xlim((1, 741))
plt.ylabel('Number of problem attempts', fontsize=10)
plt.ylim((0, 16000))

plt.grid()
plt.show()
df_InfoContent['learning_stage'].value_counts()
diff_count = df_InfoContent['difficulty'].value_counts()
plt.bar(x=diff_count.index, height=diff_count)

plt.ylim((0, 900))
plt.title('Distribution of Difficulty',fontsize=16)
plt.xlabel('Difficulty', fontsize=10)
plt.ylabel('Problem count', fontsize=10)

plt.show()
df_Problem_Content = df_LogProblem[['ucid', 'is_correct']].merge(df_InfoContent[['ucid', 'difficulty']], how='inner', left_on='ucid', right_on='ucid')

# We remove the content with difficulty unset for now
df_Problem_Content = df_Problem_Content[df_Problem_Content['difficulty'] != 'unset']

df_Problem_Content = df_Problem_Content.groupby(['difficulty', 'is_correct']).size().unstack(level=-1)
df_Problem_Content['correct_rate'] = df_Problem_Content[True] / (df_Problem_Content[True] + df_Problem_Content[False])

df_Problem_Content.sort_values(by=['correct_rate'], ascending=False)
# Randomly pick an exercise
df_userreturn = df_LogProblem[df_LogProblem['ucid'] == "CPI+5YCeEmhqdk6znJeii6jJUNl1QWGEvwCUJ6uLflg="][['timestamp_TW','uuid']].sort_values(by=['timestamp_TW'])
def GetSurvivalRate(df_week1, df_week2):
    user_w1 = set(np.unique(df_week1['uuid']))
    user_w2 = set(np.unique(df_week2['uuid']))
    SurvivalRate = round(len(user_w1.intersection(user_w2)) / len(user_w1) * 100, 3)
    print(f"{SurvivalRate}% users in the first week still stay in this week")
    return SurvivalRate
df_week0 = df_userreturn[(df_userreturn['timestamp_TW'] >= "2018-09-02") & (df_userreturn['timestamp_TW'] < "2018-09-09")]
df_week1 = df_userreturn[(df_userreturn['timestamp_TW'] >= "2018-09-09") & (df_userreturn['timestamp_TW'] < "2018-09-16")]
df_week2 = df_userreturn[(df_userreturn['timestamp_TW'] >= "2018-09-16") & (df_userreturn['timestamp_TW'] < "2018-09-23")]
df_week3 = df_userreturn[(df_userreturn['timestamp_TW'] >= "2018-09-23") & (df_userreturn['timestamp_TW'] < "2018-09-30")]
df_week4 = df_userreturn[(df_userreturn['timestamp_TW'] >= "2018-09-30") & (df_userreturn['timestamp_TW'] < "2018-10-07")]

SR_list = [GetSurvivalRate(df_week0, df_week0),
           GetSurvivalRate(df_week0, df_week1),
           GetSurvivalRate(df_week0, df_week2),
           GetSurvivalRate(df_week0, df_week3),
           GetSurvivalRate(df_week0, df_week4)]
plt.plot(SR_list)

plt.xticks(np.arange(0, 4.1, 1))
plt.yticks(np.arange(0, 101, 10))
plt.title('Survival Rate of Week 0 users in the next few weeks', fontsize=16)
plt.xlabel('Number of week', fontsize=10)
plt.ylabel('Survival Rate', fontsize=10)

for x, y in zip(range(5), SR_list): 
    plt.text(x, y, str(round(y, 2)))

plt.grid()
plt.show()