import pandas as pd
df = pd.read_csv('../input/stack-overflow-developer-survey-results-2019/survey_results_public.csv')

schema_df = pd.read_csv('../input/stack-overflow-developer-survey-results-2019/survey_results_schema.csv')
person = {

    'first': 'Raghav',

    'last': 'Gupta',

    'mail': 'raghav_gupta@mail.com',

}



people = {

    'first': ['Raghav', 'Sachin', 'KP'],

    'last': ['Gupta', 'Patidar', 'Bro'],

    'mail': ['raghav_gupta@mail.com', 'sac_p@mail.com', 'KP_bro@mail.com'],

}
df.shape # (88883, 85)

df.info() # list columns
df.describe()
pd.set_option('display.max_columns', 85)
df
small_df = pd.DataFrame(people)
small_df
small_df['first']
type(small_df['first']) # pandas.core.series.Series

type(small_df) # pandas.core.frame.DataFrame
small_df[['last', 'mail']]
small_df.columns
small_df.iloc[0]
row_list = [0, 1]

col_int_list = [0, 2]

col_lable_list = ['first', 'mail']



small_df.iloc[row_list, col_int_list]