import pandas as pd
train = pd.read_csv('../input/bdsc-classification-3-challenge/train.csv', index_col=0, parse_dates=[f'time{i+1}' for i in range(10)])

test = pd.read_csv('../input/bdsc-classification-3-challenge/test.csv', index_col=0, parse_dates=[f'time{i+1}' for i in range(10)])
target = train.target

train = train.drop(columns='target')
def preproc(df):

    return (

        df.assign(

            # difference between 10th page and 1st in seconds

            first_last_diff=lambda x: (x.time10 - x.time1).dt.seconds.fillna(0),

            # number of missing pages (from 10 max)

            nans_count=lambda x: x.isna().sum(axis=1).div(2),

            # number of unique pages in session

            n_unique_pages=lambda x: 

                x.filter(like='webpage').apply(lambda row: row.nunique(), axis=1) / (10 - x.nans_count),

            # avg hour of day in a session

            avg_hour=lambda x: x.filter(like='time').apply(lambda x: x.dt.hour).mean(axis=1),

            # avg day of week in a session

            avg_day=lambda x: x.filter(like='time').apply(lambda x: x.dt.dayofweek).mean(axis=1),

        )

        # drop time columns

        .drop(columns=[f'time{i+1}' for i in range(10)])

        # fill missing pages with zeros

        .fillna(0.)

    )

    

train = preproc(train)

test = preproc(test)
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(n_estimators=135)
rfc.fit(train, target)
from sklearn.metrics import roc_auc_score



print(roc_auc_score(target, rfc.predict_proba(train)[:, 1]))
def save_submission(pred):

    pd.Series(

        pred, name='target', index=pd.Index(range(len(pred)), name='session_id')

    ).to_csv('notebook_submission.csv')
pred = rfc.predict_proba(test)[:, 1]

save_submission(pred)