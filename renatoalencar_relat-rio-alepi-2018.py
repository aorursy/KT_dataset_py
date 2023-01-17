import pandas as pd
import numpy as np
data = pd.read_csv('../input/year-report-2018.csv', delimiter=';')
data.head(10)
by_name = data[['name', 'value']].groupby('name').sum().sort_values('value', ascending=False)
by_name
by_name.plot(x='name', y='value', kind='pie')
by_type = data[['description', 'value']].groupby('description').sum().sort_values('value', ascending=False)
by_type
by_type.plot(
    x='description',
    y='value',
    kind='pie',
    title='Distribuicao dos gastos por tipo',
    autopct='%1.1f%%',
    shadow=True,
    startangle=90
)
data['value'].sum()
by_month = data[['month', 'value']].groupby('month').sum()
by_month
mean = np.mean(by_name['value'])
mean
deviation = np.std(by_name['value'])
deviation
by_name_copy = by_name[:]
by_name_copy['relative'] = by_name_copy['value'].apply(lambda i: '%.2f %%' % ((i - mean) * 100/mean))
by_name_copy