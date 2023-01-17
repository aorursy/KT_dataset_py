import pandas as pd

calls = pd.read_csv('/kaggle/input/montcoalert/911.csv')

calls.head()

calls.info()
calls['zip'].value_counts().head(5).keys()
calls['twp'].value_counts().head(5).keys()
calls['title'].apply(lambda title: title.split(':')[1]).unique()
calls['reason'] = calls['title'].apply(lambda title: title.split(':')[0])

calls.head()
calls['reason'].value_counts().head(1).keys()

calls.info()


calls['timeStamp'] = calls['timeStamp'].astype('datetime64')

calls.info()

calls
calls['hour'] = calls['timeStamp'].dt.hour

calls['month'] = calls['timeStamp'].dt.month

calls['day'] = calls['timeStamp'].dt.day

calls['week'] = calls['timeStamp'].dt.weekday

calls['year'] = calls['timeStamp'].dt.year

calls
def number_to_name_week(value):

    if value == 0:

        return 'Mon'

    elif value == 1:

        return 'Tue'

    elif value == 2:

        return 'Wed'

    elif value == 3:

        return 'Thu'

    elif value == 4:

        return 'Fri'

    elif value == 5:

        return 'Sat'

    elif value == 6:

        return 'Sun'



calls['week'] = calls['week'].apply(number_to_name_week)

calls
month_reason_df = calls.groupby(['year','month','reason'])['e'].count()

month_reason_df
month_day_max_call = calls.groupby(['year','month','day'])['e'].count().sort_values().tail(1)

month_day_max_call

fri_hour_max_call = calls[calls['week'] == 'Fri'].groupby('hour')['e'].count().sort_values().tail(1)

fri_hour_max_call
fire_sat_call = calls[(calls['reason'] == 'Fire') & (calls['week'] == 'Sat')]['e'].count()

fire_sat_call