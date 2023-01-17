import numpy as np
import pandas as pd
df = pd.read_csv('../input/train.csv')
print('Train data:', df.shape)
df = df[df['parent_category_name'] == 'Услуги']
print('Train data for "services" parent catogory:', df.shape)
print('Distinct values for target variable in this category:', df['deal_probability'].value_counts().shape[0])
df['deal_probability'].value_counts()

dfg = df.groupby(['param_2','deal_probability'])['item_id'].count().reset_index()
dfg.columns = ['param_2','deal_probability','count']
dfg