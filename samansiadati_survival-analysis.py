import pandas as pd
address = '../input/habermans-survival-data-set/haberman.csv'
df = pd.read_csv(address)
df
df.info()
df.rename(columns = {'30':'Age', '64':'Op_Year', '1':'axil_nodes', '1.1':'Status'}, inplace=True)
df['Op_Year'].value_counts().plot(kind='bar')
df['axil_nodes'].value_counts().plot(kind='bar')
df['Status'].value_counts().plot(kind='bar')
!pip install lifelines
from lifelines import KaplanMeierFitter
kapmei = KaplanMeierFitter() 
kapmei.fit(df['Age'],df['Status'], label='Kaplan Meier Estimation').plot(ci_show=False)
group1 = (df['axil_nodes'] >= 1) 
group2 = (df['axil_nodes'] < 1)  
Age = df.Age
Status = df.Status
kapmei.fit(Age[group1],Status[group1], label='Positive axillary detected')
kapmei1 = kapmei.plot()
kapmei.fit(Age[group2],Status[group2], label='No positive axillary nodes detected ')
kapmei.plot(ax = kapmei1)
from lifelines import CoxPHFitter
cox = CoxPHFitter()
cox.fit(df, 'Age', event_col='Status').plot()