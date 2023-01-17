import pandas as pd



pd.set_option('display.max_colwidth', 500)

pd.set_option('display.max_rows', 200)
test = pd.read_csv('../input/data-science-bowl-2019/test.csv')

test.head()
is_assessment = (

    (test['title'].eq('Bird Measurer (Assessment)') & test['event_code'].eq(4110)) |

    (~test['title'].eq('Bird Measurer (Assessment)') & test['event_code'].eq(4100)) &

    test['type'].eq('Assessment')

)



test['correct'] = test['event_data'].str.extract(r'"correct":([^,]+)')

assessment = test[test['correct'].notnull() & is_assessment]



assessment
last_assessment = assessment.sort_values('timestamp').groupby(['installation_id', 'title']).last().reset_index()

last_assessment
def accuracy_group(acc):

    if acc == 1.0:

        return 3

    elif acc == 0:

        return 0

    elif 0.5 <= acc < 1.0:

        return 2

    else:

        return 1
history = last_assessment.groupby(['installation_id','title', 'correct']).size().unstack([2])

history = history.reset_index().fillna(0)

history['total'] = history['false'] + history['true']

history['accuracy'] = history['true'] / history['total']

history['accuracy_group'] = history['accuracy'].map(accuracy_group)

history
acc_stats = history.groupby('installation_id').agg({'accuracy_group': ['count', 'max', 'min', 'mean']}).reset_index()

acc_stats.columns = ['installation_id', 'count', 'max', 'min', 'mean']

acc_stats[acc_stats['count'] > 3]
last_event = test.sort_values(['installation_id', 'timestamp']).groupby('installation_id').last().reset_index()

last_event.head()
sbm_sample = pd.read_csv('../input/data-science-bowl-2019/sample_submission.csv')

sbm_sample = pd.merge(sbm_sample, last_event[['installation_id', 'title']], on='installation_id')

sbm_sample
sbm_history = pd.merge(

    sbm_sample.drop('accuracy_group', axis=1),

    history[['installation_id', 'title', 'accuracy_group']],

    on=['installation_id', 'title'],

    how='left'

)



sbm_history
sbm_history['accuracy_group'].value_counts(dropna=False)
def predict(accuracy):

    if accuracy > 0.5:

        return 3

    if accuracy > 0.4:

        return 2

    return 0



labels = pd.read_csv('../input/data-science-bowl-2019/train_labels.csv')

agg = labels.groupby('title').sum()[['num_correct', 'num_incorrect']].reset_index()

agg['accuracy'] = agg['num_correct'] / (agg['num_incorrect'] + agg['num_correct'])

agg['accuracy_group'] = agg['accuracy'].map(predict)

agg
import numpy as np



sbm = pd.merge(

    sbm_history,

    agg[['title', 'accuracy_group']],

    on='title',

    how='left',

    suffixes=('_history', '_labels'),

)



sbm['accuracy_group'] = np.where(

    sbm['accuracy_group_history'].notnull(),

    sbm['accuracy_group_history'],

    sbm['accuracy_group_labels'],

).astype(np.int8)
sbm.sample(30)
sbm[['installation_id', 'accuracy_group']].to_csv('submission.csv', index=False)
test[test['installation_id'] == '00abaee7']