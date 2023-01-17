import pandas as pd
data = pd.read_csv('../input/pokemon_alopez247.csv', index_col='Number')
data.head()

data.columns
type(data.Name)
(data.Name == data['Name']).all()
data['Total'].max()
total_mas_grande = data['Total'].max()
data[ data.Total == data['Total'].max() ]
data['Speed'].mean()
!pip install matplotlib
%matplotlib inline
data['Attack'].hist()
data['Attack'].min()
data['Attack'].max()
data.Type_1.value_counts()
data.Type_2.value_counts()
data.groupby('Type_2')['Total'].mean().sort_values(ascending=False)
data.groupby('Type_2')['Total'].value_counts().sort_values(ascending=False)
data.groupby(['Type_1', 'Type_2'])['Attack'].max().sort_values(ascending=False)
data.groupby(['Type_1', 'Type_2'])['Attack'].value_counts().sort_values(ascending=False)
import seaborn as sns
sns.jointplot(x='Sp_Atk', y='Sp_Def', data=data, kind='reg')
data.head()
sns.boxplot(data = data.drop(['Name', 'Total'], axis=1).head())