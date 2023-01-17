import pandas as pd

data = pd.read_csv('../input/scl-dummy/Dummy data.csv')
data.head(5)
data_copy = data.copy()
data_copy['number'] = data['id'].apply(lambda x: x + 2)
data_copy.rename(columns={'number':'new_number'}, inplace=True)
data_copy.head(5)
data_copy.to_csv('solution.csv', index = False)
