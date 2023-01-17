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
import pandas as pd



def extract_accuracy_group(df: pd.DataFrame) -> pd.DataFrame:

    # Regex strings for matching Assessment Types

    assessment_4100 = '|'.join(['Mushroom Sorter',

                                'Chest Sorter',

                                'Cauldron Filler',

                                'Cart Balancer'])

    assessment_4110 = 'Bird Measurer'

    

    # 1. Extract all assessment scoring events

    score_events = df[((df['title'].str.contains(assessment_4110)) & (df['event_code']==4110)) |\

                      ((df['title'].str.contains(assessment_4100)) & (df['event_code']==4100))]

    

    # 2. Count number of correct vs. attempts

    # 2.a. Create flags for correct vs incorrect

    score_events['correct'] = 1

    score_events['correct'] = score_events['correct'].where(score_events['event_data'].str.contains('"correct":true'),other=0)

    

    score_events['incorrect'] = 1

    score_events['incorrect'] = score_events['incorrect'].where(score_events['event_data'].str.contains('"correct":false'),other=0)

    

    # 2.b. Aggregate by `installation_id`,`game_session`,`title`

    score_events_sum = score_events.groupby(['installation_id','game_session','title'])['correct','incorrect'].sum()

    

    # 3. Apply heuristic to convert counts into accuracy group

    # 3.a. Define heuristic

    def acc_group(row: pd.Series) -> int:

        if row['correct'] == 0:

            return 0

        elif row['incorrect'] == 0:

            return 3

        elif row['incorrect'] == 1:

            return 2

        else:

            return 1

        

    # 3.b. Apply heuristic to count data

    score_events_sum['accuracy_group'] = score_events_sum.apply(acc_group,axis=1)

    

    return score_events_sum
import os



DATA_DIR = '/kaggle/input/data-science-bowl-2019'
# Read `train.csv`

train = pd.read_csv(os.path.join(DATA_DIR,'train.csv'))
# Run reconciliation

train_labels_extracted = extract_accuracy_group(train)
train_labels_extracted.head(20)
# Read `train_labels.csv`

train_labels = pd.read_csv(os.path.join(DATA_DIR,'train_labels.csv'))
train_labels.drop(['accuracy'], axis=1).head(20)
train_labels_extracted['accuracy_group'].value_counts()
train_labels['accuracy_group'].value_counts()
# Flatten multi-index

train_labels_extracted.reset_index(inplace=True)

train_labels_extracted.head()
extracted_train_sessions = set(train_labels_extracted['game_session'])

train_sessions = set(train_labels['game_session'])
extracted_train_sessions.symmetric_difference(train_sessions)
extracted_train_groups = list(train_labels_extracted['accuracy_group'])

train_groups = list(train_labels['accuracy_group'])
all_match = True

for extract, gold in zip(extracted_train_groups, train_groups):

    if extract != gold:

        all_match = False

        break



if(all_match):

    print(f"All {len(extracted_train_groups)} groups match")

else:

    print(f"Found at least one mismatched group")
test = pd.read_csv(os.path.join(DATA_DIR,'test.csv'))
test_labels = extract_accuracy_group(test)
test_labels['accuracy_group'].value_counts()
test_labels.to_csv('test_labels.csv')