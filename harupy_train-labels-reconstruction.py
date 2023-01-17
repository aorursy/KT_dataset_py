import pandas as pd

import numpy as np



pd.set_option('display.max_columns', None)
import IPython



def display(*dfs):

    for df in dfs:

        IPython.display.display(df)
train = pd.read_csv('../input/data-science-bowl-2019/train.csv')

test = pd.read_csv('../input/data-science-bowl-2019/test.csv')

labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')
display(train.head(), test.head(), labels.head())
def select_assessments(df):

    is_assessment = (

        (df['title'].eq('Bird Measurer (Assessment)') & df['event_code'].eq(4110)) |

        (~df['title'].eq('Bird Measurer (Assessment)') & df['event_code'].eq(4100)) &

        df['type'].eq('Assessment')

    )

    assessments = df[is_assessment].reset_index(drop=True)

    assessments['correct'] = assessments['event_data'].str.extract(r'"correct":([^,]+)')

    assessments['correct'] = assessments['correct'].map(lambda x: 1 if x == 'true' else 0)

    return assessments
assessments_train = select_assessments(train)

assessments_train.head()
def with_name(f, name):

    f.__name__ = name

    return f





def accuracy_group(acc):

    if acc == 0:

        return 0

    elif acc == 1:

        return 3

    elif 0.5 <= acc < 1.0:

        return 2

    else:

        return 1
aggs = {'correct': [

    with_name(lambda ser: (ser == 1).sum(), 'num_correct'),

    with_name(lambda ser: (ser == 0).sum(), 'num_incorrect'),

    with_name(lambda ser: ser.mean(), 'accuracy'),

]}



by = ['installation_id', 'title', 'game_session']

stats = assessments_train.groupby(by).agg(aggs).reset_index()

stats.columns = [col[1] if (col[1] != '') else col[0] for col in stats.columns] # flatten columns

stats = stats.assign(accuracy_group=stats['accuracy'].map(accuracy_group).astype(np.int64))  # add accuracy group
display(

    stats.sort_values(by).reset_index(drop=True).sort_index(axis=1),

    labels.sort_values(by).reset_index(drop=True).sort_index(axis=1)

)
from pandas.testing import assert_frame_equal



assert_frame_equal(

    stats.sort_values(by).reset_index(drop=True).sort_index(axis=1),

    labels.sort_values(by).reset_index(drop=True).sort_index(axis=1),

)