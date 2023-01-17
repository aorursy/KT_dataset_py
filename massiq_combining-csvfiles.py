import os

import pandas as pd

import warnings

warnings.filterwarnings("ignore")
combined = pd.DataFrame()

for i,file in enumerate(os.listdir('/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/')):

    if file.endswith('.csv'):

        path = f'/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/{file}'

        df = pd.read_csv(f'/kaggle/input/cord-19-eda-parse-json-and-generate-clean-csv/{file}')

        combined = pd.concat([combined, df], ignore_index=True)

        print(f'Total documents in {file}: ', len(df))

        df.head()



print('='*80)

combined['title'] = combined['title'].str.lower() 

combined['abstract'] = combined['abstract'].str.lower() 

before = len(combined)

print('Total documents in dataset: ', before)

combined.drop_duplicates(subset=['title', 'abstract'], inplace=True)

print('After removing duplicates based on title and abstract: ', len(combined))

combined.rename(columns={'title': 'combined_title'}, inplace=True)

print('Documents with same title and abstract: ', before - len(combined))


output_file = 'combined_dataset.csv'

combined.to_csv(output_file, index=False)