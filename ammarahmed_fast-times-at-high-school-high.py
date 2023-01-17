# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

#print(os.listdir("../input/StudentsPerformance.csv"))

df=pd.read_csv('../input/StudentsPerformance.csv')



def map_func(number,**kwargs):

    lim_chart=[]

    default_label=''

    for a in kwargs:

        if a=='lim_chart':

            lim_chart=kwargs[a]

        elif a=='default_label':

            default_label=kwargs[a]

        else:

            pass

    for a in lim_chart:

        if number>=a["lower_lim"] and number<=a["higher_lim"]:

            return a['label']

    return default_label 

# Any results you write to the current directory are saved as output.   
df.head()
parents_ed_n_lunch=df.copy()

parents_ed_n_lunch['parents_ed_n_lunch']=0

parents_ed_n_lunch=parents_ed_n_lunch.groupby(['lunch','parental level of education'])['parents_ed_n_lunch'].count()

parents_ed_n_lunch=parents_ed_n_lunch.sort_values()

parents_ed_n_lunch=parents_ed_n_lunch.reset_index()

parents_ed_n_lunch['category']=parents_ed_n_lunch['lunch'] + ' and ' + parents_ed_n_lunch['parental level of education']

parents_ed_n_lunch


labels = parents_ed_n_lunch['category']

sizes =  parents_ed_n_lunch['parents_ed_n_lunch']



np.random.seed(19680801)

plt.rcdefaults()

fig, ax = plt.subplots()



# Example data

#people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')

y_pos = np.arange(len(labels))

#performance = 3 + 10 * np.random.rand(len(people))

#error = np.random.rand(len(people))



ax.barh(y_pos, sizes, align='center',

        color='green', ecolor='black')

ax.set_yticks(y_pos)

ax.set_yticklabels(labels)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('count')

ax.set_title('pupils`s lunch and parent`s school  by the numbers')

plt.show()
avg_score_vs_prep=df.copy()

avg_score_vs_prep['average score']=( avg_score_vs_prep['math score']+avg_score_vs_prep['reading score']+avg_score_vs_prep['writing score'] ) / 3 

avg_score_vs_prep=avg_score_vs_prep.groupby(['test preparation course'])['average score'].mean()

avg_score_vs_prep.sort_values(ascending=False).reset_index()

test_prep_score_metrics=df.copy()

test_prep_score_metrics['average score']=( test_prep_score_metrics['math score']+test_prep_score_metrics['reading score']+test_prep_score_metrics['writing score'] ) / 3 

lim_chart=[{"lower_lim":90,"higher_lim":100,"label":"90 to 100"},

          {"lower_lim":80,"higher_lim":89,"label":"80 to 89"},

          {"lower_lim":70,"higher_lim":79,"label":"70 to 79"},

          {"lower_lim":60,"higher_lim":69,"label":"60 to 69"}

          ]

default_label='below 60'

test_prep_score_metrics['score range']=test_prep_score_metrics['average score'].apply(map_func,lim_chart=lim_chart,default_label=default_label)

#test_prep_score_metrics.head()



#print(test_prep_score_metrics)

test_prep_score_metrics['count']=0

test_prep_score_metrics=test_prep_score_metrics.groupby(['score range','test preparation course'])['count'].count()

test_prep_score_metrics=test_prep_score_metrics.sort_values(ascending=False)

test_prep_score_metrics=test_prep_score_metrics.reset_index()

test_prep_score_metrics
fig, ax = plt.subplots()

color='blue'

x=test_prep_score_metrics[test_prep_score_metrics['test preparation course']=='none']['score range']

y=test_prep_score_metrics[test_prep_score_metrics['test preparation course']=='none']['count']

ax.scatter(x, y, c=color,  label=color,alpha=1, edgecolors='none')



color='red'



x=test_prep_score_metrics[test_prep_score_metrics['test preparation course']=='completed']['score range']

y=test_prep_score_metrics[test_prep_score_metrics['test preparation course']=='completed']['count']

ax.scatter(x, y, c=color,  label=color,alpha=1, edgecolors='none')

ax.legend(['without course preparation','with course preparation'])

ax.grid(True)



plt.show()
male_female_n_lunch=df.copy()

male_female_n_lunch['count']=0

male_female_n_lunch=male_female_n_lunch.groupby(['gender','lunch'])['count'].count()

male_female_n_lunch=male_female_n_lunch.reset_index()

#male_female_n_lunch



n_groups = 2

num_male = male_female_n_lunch[male_female_n_lunch['gender']=='male']['count']

num_female = male_female_n_lunch[male_female_n_lunch['gender']=='female']['count']

fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.35

opacity = 0.4

error_config = {'ecolor': '0.3'}

rects1 = plt.bar(index, num_male, bar_width,

                 alpha=opacity,

                 color='b',

                 error_kw=error_config,

                 label='male')

rects1 = plt.bar(index+bar_width, num_female, bar_width,

                 alpha=opacity,

                 color='r',

                 error_kw=error_config,

                 label='female')

plt.xlabel('lunch category')

plt.ylabel('count')

plt.title('count by gender and lunch category')

plt.xticks(index + bar_width / 2, (['free/reduced','standard']))

plt.legend()

plt.tight_layout()

plt.show()

male_female_n_score=df.copy()

n_groups = 3

fig, ax = plt.subplots()

index = np.arange(n_groups)

bar_width = 0.35

opacity = 0.4

error_config = {'ecolor': '0.3'}

extra=0

scores_labels=['writing score','reading score','math score']

color='r'

for a in male_female_n_score['gender'].unique():



    scores_actual=[

    np.mean(male_female_n_score[male_female_n_score['gender']==a]['writing score']),

    np.mean(male_female_n_score[male_female_n_score['gender']==a]['reading score']),

    np.mean(male_female_n_score[male_female_n_score['gender']==a]['math score'])

    ]

    

    rects1 = plt.bar(index+extra, scores_actual, bar_width,

                     alpha=opacity,

                     color=color,

                     error_kw=error_config,

                     label=a)

    extra=0.35

    color='b'

plt.xlabel('lunch category')

plt.ylabel('count')

plt.title('count by gender and scores')

plt.xticks(index + bar_width / 2, (scores_labels))

plt.legend()

plt.tight_layout()

plt.show()

male_n_female_test_prep=df.copy()

male_n_female_test_prep['count']=0

male_n_female_test_prep=male_n_female_test_prep.groupby(['gender','test preparation course'])['count'].count()

male_n_female_test_prep=male_n_female_test_prep.sort_values().reset_index()

male_n_female_test_prep['category']=male_n_female_test_prep['gender']+'/'+male_n_female_test_prep['test preparation course']

labels = male_n_female_test_prep['category']



sizes =  male_n_female_test_prep['count']

np.random.seed(19680801)

plt.rcdefaults()

fig, ax = plt.subplots()



# Example data

#people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')

y_pos = np.arange(len(labels))

#performance = 3 + 10 * np.random.rand(len(people))

#error = np.random.rand(len(people))



ax.barh(y_pos, sizes, align='center',

        color='green', ecolor='black')

ax.set_yticks(y_pos)

ax.set_yticklabels(labels)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('count')

ax.set_title('test prep course and gender by the numbers')

plt.show()