import pandas as pd

import os



pd.set_option('display.max_columns', None)
BASE_DIR = '../input/aipowered-literature-review-csvs/kaggle/working/'

dir_list = os.listdir(BASE_DIR)

dir_list
for d in dir_list:

    print(d)

    file_list = os.listdir(os.path.join(BASE_DIR, d))

    for f in file_list:

        print(f'\t{f}')

        df = pd.read_csv(os.path.join(BASE_DIR, d, f), index_col=0)

        for c in df.columns:

            print(f'\t\t- {c}')

        print()
risk_dir = os.path.join(BASE_DIR, 'Risk Factors')

df_list = []

for f in os.listdir(risk_dir):

    df = pd.read_csv(os.path.join(risk_dir, f), index_col=0)

    df['csv_source'] = f

    df_list.append(df)
df = pd.concat(df_list).reset_index(drop=True)

df['Date'] = pd.to_datetime(df.Date)
# Check what metrics have been used

df.Severe.str.split(' ', expand=True)[0].value_counts()
df.Fatality.str.split(' ', expand=True)[0].value_counts()
# Extract the various metrics used

for col in ['Severe', 'Fatality']:

    for metric in ['OR', 'AOR', 'HR', 'AHR', 'RR']:

        capture_string = metric + r'(?:\s|=)(\d+.\d+)'

        df[f'{col.lower()}_{metric.lower()}'] = df[col].str.extract(capture_string)
# Extract the upper and lower confidence intervals

lower_capture_string = r'95% CI: (\d+.\d+)'

upper_capture_string = r'95% CI: \d+.\d+\s?-\s?(\d+.\d+)'

for col in ['Severe', 'Fatality']:

    df[f'{col.lower()}_ci_lower'] = df[col].str.extract(lower_capture_string)

    df[f'{col.lower()}_ci_upper'] = df[col].str.extract(upper_capture_string)
# Extract the p values

p_value_capture_string = r'p=(0.\d+)'

for col in ['Severe', 'Fatality']:

    df[f'{col.lower()}_p_value'] = df[col].str.extract(p_value_capture_string)
df.head(20)
# Quick tidy to clarify the subject

df['risk_factor'] = df.csv_source.str.slice(0, -4)
df.to_csv('risk_factors_training_data.csv', index=False)