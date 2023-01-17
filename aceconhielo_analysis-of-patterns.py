# Libraries

import pandas as pd

import datetime



# Visualization libraries

import matplotlib.pyplot as plt

import seaborn as sns



# Data

path_data = '../input/santa-workshop-tour-2019/'

family_data = pd.read_csv(path_data + 'family_data.csv')

sample_submission = pd.read_csv(path_data + 'sample_submission.csv')
# Aux libraries

# Get some useful and basic statistics from the dataset

def basic_stats(df, namedf):

    print('These are the next basic stats for the dataframe: ', namedf)

    print('==================================================')

    print('Rows x Columns: ', df.shape)

    print('Info output: ', df.info())

    print('Describe output: ', df.describe())

    

basic_stats(sample_submission, 'sample_submission.csv')

sample_submission.head()
basic_stats(family_data, 'family_data.csv')

family_data.head()
family_data_people = family_data[['family_id','n_people']].groupby('n_people', as_index=False).count().sort_values('n_people', ascending=True)

family_data_people = family_data_people.rename(columns={"family_id": "count"})



print(family_data_people)

sns.barplot(x='n_people', y='count', data=family_data_people, palette='rocket')
# choice 0

sns.set(rc={'figure.figsize':(25,7)})



def choice(n_choice):

    date_christmas = datetime.datetime.strptime('2019-12-24', "%Y-%m-%d") # Happy Christmas !!

    choicedf = family_data[['family_id',n_choice]].groupby(n_choice, as_index=False).count()

    choicedf['date'] = choicedf.apply(lambda x: (date_christmas + datetime.timedelta(days=-int(x[n_choice]))).strftime("%b %d - %a"),axis=1)

    ax0 = sns.barplot(x='date', y='family_id', data=choicedf, palette='rocket')

    for item in ax0.get_xticklabels():

        item.set_rotation(90)

        

choice('choice_0')
def transform_choice(n_choice):

    date_christmas = datetime.datetime.strptime('2019-12-24', "%Y-%m-%d") # Happy Christmas !!

    choicedf = family_data[['family_id',n_choice]].groupby(n_choice, as_index=False).count()

    choicedf['date'] = choicedf.apply(lambda x: (date_christmas + datetime.timedelta(days=-int(x[n_choice]))).strftime("%b %d - %a"),axis=1)

    choicedf = choicedf.rename(columns={"family_id": "amount_families"})

    

    return choicedf



choice_0 = transform_choice('choice_0')

choice_1 = transform_choice('choice_1')

choice_2 = transform_choice('choice_2')

choice_3 = transform_choice('choice_3')

choice_4 = transform_choice('choice_4')

choice_5 = transform_choice('choice_5')

choice_6 = transform_choice('choice_6')

choice_7 = transform_choice('choice_7')

choice_8 = transform_choice('choice_8')

choice_9 = transform_choice('choice_9')
f, axes = plt.subplots(5, 2, figsize=(30, 20), sharex=True)

for ax in f.axes:

    plt.sca(ax)

    plt.xticks(rotation=90)

sns.barplot(x='date', y='amount_families', data=choice_0, ax=axes[0,0]).set_title('Choice 0')

sns.barplot(x='date', y='amount_families', data=choice_1, ax=axes[0,1]).set_title('Choice 1')

sns.barplot(x='date', y='amount_families', data=choice_2, ax=axes[1,0]).set_title('Choice 2')

sns.barplot(x='date', y='amount_families', data=choice_3, ax=axes[1,1]).set_title('Choice 3')

sns.barplot(x='date', y='amount_families', data=choice_4, ax=axes[2,0]).set_title('Choice 4')

sns.barplot(x='date', y='amount_families', data=choice_5, ax=axes[2,1]).set_title('Choice 5')

sns.barplot(x='date', y='amount_families', data=choice_6, ax=axes[3,0]).set_title('Choice 6')

sns.barplot(x='date', y='amount_families', data=choice_7, ax=axes[3,1]).set_title('Choice 7')

sns.barplot(x='date', y='amount_families', data=choice_8, ax=axes[4,0]).set_title('Choice 8')

sns.barplot(x='date', y='amount_families', data=choice_9, ax=axes[4,1]).set_title('Choice 9')

# Check duplicates in the same row

# If all the user priorities are uniques dates => True



family_data['uniques_dates'] = family_data.apply(lambda x: len(list(set(x[1:11].values))) == len(x[1:11].values) , axis=1)

print('Families that have choose more than once the same date : ', family_data[family_data['uniques_dates']==False].shape[0])
