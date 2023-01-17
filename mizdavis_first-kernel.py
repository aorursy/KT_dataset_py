import pandas as pd

import seaborn as sns



sns.set(rc={'figure.figsize':(14,4)})
firesdf=pd.read_csv('../input/forest-fires-in-brazil/amazon.csv', encoding='latin1')

firesdf.head()
years=firesdf.year.unique()

print(years)
states=firesdf.state.unique()

print(states)
firesdf.drop('date', axis=1, inplace=True)
firesdf.number.describe([.25, .5, .75])
sub1=pd.DataFrame(firesdf.groupby(['state']).sum())

sub1=sub1.reset_index()

sub1=sub1.sort_values(by=['number'], ascending=False)

sub1=sub1.head(10)
sns.barplot(x="state", y="number",data=sub1)