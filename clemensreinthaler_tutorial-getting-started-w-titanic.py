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
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head()
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
# select rows with id column between 100 and 200, and just return 'postal' and 'web' columns

#data.loc[(data['id'] > 100) & (data['id'] <= 200), ['postal', 'web']] 



women_all = train_data.loc[train_data.Sex == 'female']["Survived"]

women_filter = train_data.loc[(train_data.Sex == 'female') & (train_data.Survived == 1 )] ["Survived"]



rate_women_all = sum(women_all)/len(women_all)

rate_women_filter = sum(women_filter)/len(women_filter)



print("Anzahl aller Frauen: ", len(women_all))

print("Anzahl gefilteter Frauen: ", len(women_filter))

print("% of women who survived:", rate_women_all*100)

print("% of women who survived_Filter:", rate_women_filter*100)
print(women_filter)

print(women_all)
men = train_data.loc[train_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men*100)
import matplotlib.pyplot as plt
age_groups = ['children (0-10)', 'teenager (11-18)', 'young adult (19-30)', 'adult (31-65)', 'elder (65+)' ]

print(age_groups[0])
df_age_groups = pd.DataFrame([[0, 10], [11, 18], [19, 30], [31,64],[65,100]],

     index=['children (0-10)', 'teenager (11-18)', 'young adult (19-30)', 'adult (31-65)', 'elder (65+)'],

     columns=['StartAge', 'EndAge'])

print(df_age_groups)
children_sur = train_data.loc[(train_data.Age >= df_age_groups.loc['children (0-10)', 'StartAge']) & (train_data.Age <= df_age_groups.loc['children (0-10)', 'EndAge']) & (train_data.Survived == 1), ['PassengerId', 'Pclass', 'Sex', 'Age']]

print(len(children_sur))



teenager_sur = train_data.loc[(train_data.Age >= df_age_groups.loc['teenager (11-18)', 'StartAge']) & (train_data.Age <= df_age_groups.loc['teenager (11-18)', 'EndAge']) & (train_data.Survived == 1), ['PassengerId', 'Pclass', 'Sex', 'Age']]

print(len(teenager_sur))



young_adult_sur = train_data.loc[(train_data.Age >= df_age_groups.loc['young adult (19-30)', 'StartAge']) & (train_data.Age <= df_age_groups.loc['young adult (19-30)', 'EndAge']) & (train_data.Survived == 1), ['PassengerId', 'Pclass', 'Sex', 'Age']]

print(len(young_adult_sur))



adult_sur = train_data.loc[(train_data.Age >= df_age_groups.loc['adult (31-65)', 'StartAge']) & (train_data.Age <= df_age_groups.loc['adult (31-65)', 'EndAge']) & (train_data.Survived == 1), ['PassengerId', 'Pclass', 'Sex', 'Age']]

print(len(adult_sur))



elder_sur = train_data.loc[(train_data.Age >= df_age_groups.loc['elder (65+)', 'StartAge']) & (train_data.Age <= df_age_groups.loc['elder (65+)', 'EndAge']) & (train_data.Survived == 1), ['PassengerId', 'Pclass', 'Sex', 'Age']]

print(len(elder_sur))



df_age_groups_sur = pd.DataFrame([['children (0-10)', 0, 10, len(children_sur)], ['teenager (11-18)', 11, 18, len(teenager_sur)], 

                                  ['young adult (19-30)', 19, 30,len(young_adult_sur)], ['adult (31-65)', 31,64,len(adult_sur)],

                                  ['elder (65+)', 65,100,len(elder_sur)]],

     index=[1,2,3,4,5],

     columns=['Category','StartAge', 'EndAge', 'Count'])

print(df_age_groups_sur)



children = train_data.loc[(train_data.Age >= df_age_groups.loc['children (0-10)', 'StartAge']) & (train_data.Age <= df_age_groups.loc['children (0-10)', 'EndAge']), ['PassengerId', 'Pclass', 'Sex', 'Age']]

print(len(children))



teenager = train_data.loc[(train_data.Age >= df_age_groups.loc['teenager (11-18)', 'StartAge']) & (train_data.Age <= df_age_groups.loc['teenager (11-18)', 'EndAge']) , ['PassengerId', 'Pclass', 'Sex', 'Age']]

print(len(teenager))



young_adult = train_data.loc[(train_data.Age >= df_age_groups.loc['young adult (19-30)', 'StartAge']) & (train_data.Age <= df_age_groups.loc['young adult (19-30)', 'EndAge']), ['PassengerId', 'Pclass', 'Sex', 'Age']]

print(len(young_adult))



adult = train_data.loc[(train_data.Age >= df_age_groups.loc['adult (31-65)', 'StartAge']) & (train_data.Age <= df_age_groups.loc['adult (31-65)', 'EndAge']), ['PassengerId', 'Pclass', 'Sex', 'Age']]

print(len(adult))



elder = train_data.loc[(train_data.Age >= df_age_groups.loc['elder (65+)', 'StartAge']) & (train_data.Age <= df_age_groups.loc['elder (65+)', 'EndAge']), ['PassengerId', 'Pclass', 'Sex', 'Age']]

print(len(elder))



df_age_groups_complett = pd.DataFrame([['children (0-10)', 0, 10, len(children)], ['teenager (11-18)', 11, 18, len(teenager)], 

                                  ['young adult (19-30)', 19, 30,len(young_adult)], ['adult (31-65)', 31,64,len(adult)],

                                  ['elder (65+)', 65,100,len(elder)]],

     index=[1,2,3,4,5],

     columns=['Category','StartAge', 'EndAge', 'Count'])

print(df_age_groups_complett)





# select rows with id column between 100 and 200, and just return 'postal' and 'web' columns

#data.loc[(data['id'] > 100) & (data['id'] <= 200), ['postal', 'web']] 
children_sur = train_data.loc[(train_data.Age >= df_age_groups.loc['children (0-10)', 'StartAge']) & (train_data.Age <= df_age_groups.loc['children (0-10)', 'EndAge']) & (train_data.Survived == 1), ['PassengerId', 'Pclass', 'Sex', 'Age']]

print(len(children_sur))



teenager_sur = train_data.loc[(train_data.Age >= df_age_groups.loc['teenager (11-18)', 'StartAge']) & (train_data.Age <= df_age_groups.loc['teenager (11-18)', 'EndAge']) & (train_data.Survived == 1), ['PassengerId', 'Pclass', 'Sex', 'Age']]

print(len(teenager_sur))



young_adult_sur = train_data.loc[(train_data.Age >= df_age_groups.loc['young adult (19-30)', 'StartAge']) & (train_data.Age <= df_age_groups.loc['young adult (19-30)', 'EndAge']) & (train_data.Survived == 1), ['PassengerId', 'Pclass', 'Sex', 'Age']]

print(len(young_adult_sur))



adult_sur = train_data.loc[(train_data.Age >= df_age_groups.loc['adult (31-65)', 'StartAge']) & (train_data.Age <= df_age_groups.loc['adult (31-65)', 'EndAge']) & (train_data.Survived == 1), ['PassengerId', 'Pclass', 'Sex', 'Age']]

print(len(adult_sur))



elder_sur = train_data.loc[(train_data.Age >= df_age_groups.loc['elder (65+)', 'StartAge']) & (train_data.Age <= df_age_groups.loc['elder (65+)', 'EndAge']) & (train_data.Survived == 1), ['PassengerId', 'Pclass', 'Sex', 'Age']]

print(len(elder_sur))



df_age_groups_sur = pd.DataFrame([['children (0-10)', 0, 10, len(children_sur)], ['teenager (11-18)', 11, 18, len(teenager_sur)], 

                                  ['young adult (19-30)', 19, 30,len(young_adult_sur)], ['adult (31-65)', 31,64,len(adult_sur)],

                                  ['elder (65+)', 65,100,len(elder_sur)]],

     index=[1,2,3,4,5],

     columns=['Category','StartAge', 'EndAge', 'Count'])

print(df_age_groups_sur)



children = train_data.loc[(train_data.Age >= df_age_groups.loc['children (0-10)', 'StartAge']) & (train_data.Age <= df_age_groups.loc['children (0-10)', 'EndAge']), ['PassengerId', 'Pclass', 'Sex', 'Age']]

print(len(children))



teenager = train_data.loc[(train_data.Age >= df_age_groups.loc['teenager (11-18)', 'StartAge']) & (train_data.Age <= df_age_groups.loc['teenager (11-18)', 'EndAge']) , ['PassengerId', 'Pclass', 'Sex', 'Age']]

print(len(teenager))



young_adult = train_data.loc[(train_data.Age >= df_age_groups.loc['young adult (19-30)', 'StartAge']) & (train_data.Age <= df_age_groups.loc['young adult (19-30)', 'EndAge']), ['PassengerId', 'Pclass', 'Sex', 'Age']]

print(len(young_adult))



adult = train_data.loc[(train_data.Age >= df_age_groups.loc['adult (31-65)', 'StartAge']) & (train_data.Age <= df_age_groups.loc['adult (31-65)', 'EndAge']), ['PassengerId', 'Pclass', 'Sex', 'Age']]

print(len(adult))



elder = train_data.loc[(train_data.Age >= df_age_groups.loc['elder (65+)', 'StartAge']) & (train_data.Age <= df_age_groups.loc['elder (65+)', 'EndAge']), ['PassengerId', 'Pclass', 'Sex', 'Age']]

print(len(elder))



df_age_groups_complett = pd.DataFrame([['children (0-10)', 0, 10, len(children)], ['teenager (11-18)', 11, 18, len(teenager)], 

                                  ['young adult (19-30)', 19, 30,len(young_adult)], ['adult (31-65)', 31,64,len(adult)],

                                  ['elder (65+)', 65,100,len(elder)]],

     index=[1,2,3,4,5],

     columns=['Category','StartAge', 'EndAge', 'Count'])

print(df_age_groups_complett)






# data to plot

n_groups = 5



# create plot

plt.figure(figsize=(15, 8))

bar_width = 0.35

opacity = 0.8



rects1 = plt.bar(df_age_groups_sur.index, df_age_groups_sur.Count, bar_width,

alpha=opacity,

color='g',

label='Survived')



rects2 = plt.bar(df_age_groups_complett.index+bar_width, df_age_groups_complett.Count, bar_width,

alpha=opacity,

color='r',

label='Died')



plt.title('Age Distribution of Survivors/Deaths')

plt.xlabel('Age-Groups', fontsize=14, color='black')

plt.ylabel('Count', fontsize=14, color='black')

plt.xticks(df_age_groups_complett.index + bar_width, df_age_groups_complett.Category)

plt.grid(True)

plt.legend()

plt.figure(figsize=(15, 8))

plt.tight_layout()

plt.show()
children.head()
children_male = children.loc[children.Sex == 'male']

children_female = children.loc[children.Sex == 'female'] 



teenager_male = teenager.loc[teenager.Sex == 'male']

teenager_female = teenager.loc[teenager.Sex == 'female'] 



youngadult_male = young_adult.loc[young_adult.Sex == 'male']

youngadult_female = young_adult.loc[young_adult.Sex == 'female'] 



adult_male = adult.loc[adult.Sex == 'male']

adult_female = adult.loc[adult.Sex == 'female'] 



elder_male = elder.loc[elder.Sex == 'male'] 

elder_female = elder.loc[elder.Sex == 'female'] 



children_male_sur = children_sur.loc[children_sur.Sex == 'male']

children_female_sur = children_sur.loc[children_sur.Sex == 'female'] 



teenager_male_sur = teenager_sur.loc[teenager_sur.Sex == 'male']

teenager_female_sur = teenager_sur.loc[teenager_sur.Sex == 'female'] 



youngadult_male_sur = young_adult_sur.loc[young_adult_sur.Sex == 'male']

youngadult_female_sur = young_adult_sur.loc[young_adult_sur.Sex == 'female'] 



adult_male_sur = adult_sur.loc[adult_sur.Sex == 'male']

adult_female_sur = adult_sur.loc[adult_sur.Sex == 'female'] 



elder_male_sur = elder_sur.loc[elder_sur.Sex == 'male'] 

elder_female_sur = elder_sur.loc[elder_sur.Sex == 'female'] 
# data to plot

n_groups = 5

menMeans = (len(children_male), len(teenager_male), len(youngadult_male), len(adult_male), len(elder_male))

womenMeans = (len(children_female), len(teenager_female), len(youngadult_female), len(adult_female), len(elder_female))



menMeans_sur = (len(children_male_sur), len(teenager_male_sur), len(youngadult_male_sur), len(adult_male_sur), len(elder_male_sur))

womenMeans_sur = (len(children_female_sur), len(teenager_female_sur), len(youngadult_female_sur), len(adult_female_sur), len(elder_female_sur))



# create plot

plt.figure(figsize=(15, 8))

bar_width = 0.35

opacity = 0.8



p1=plt.bar(df_age_groups_sur.index, menMeans, bar_width,

alpha=opacity,

color='g',

label='Men')



p2 = plt.bar(df_age_groups_complett.index, womenMeans, bar_width, bottom=menMeans,

alpha=opacity,

color='r',

label='Woman')



p3 = plt.bar(df_age_groups_sur.index, menMeans_sur, bar_width, 

alpha=opacity,

color='y',

label='Man_survived')





p4 = plt.bar(df_age_groups_sur.index, womenMeans_sur, bar_width, bottom=menMeans_sur,

alpha=opacity,

color='orange',

label='Women_survived')





plt.title('Age Distribution of Survivors/Deaths')

plt.xlabel('Age-Groups', fontsize=14, color='black')

plt.ylabel('Count', fontsize=14, color='black')

plt.xticks(df_age_groups_complett.index + bar_width, df_age_groups_complett.Category)

plt.grid(True)

plt.legend()

plt.figure(figsize=(15, 8))

plt.tight_layout()

plt.show()