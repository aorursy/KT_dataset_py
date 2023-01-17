%matplotlib inline
import pandas as pd
from subprocess import check_output
print(check_output(["ls", "../input/"]))
data = pd.read_csv('../input/train.tsv', delimiter='\t')
data.head()
data.shape
data.describe()
data['Sentiment'].unique()
data['Sentiment'].hist()
data['Sentiment'].value_counts()
data.isnull().sum(axis=0)
data[data['Sentiment'] == 4].head()
data['Phrase'].map(len).max()
data.groupby(['SentenceId']).filter(lambda x: x['Sentiment'].nunique() > 1)['SentenceId'].nunique()
data.groupby(['SentenceId']).filter(lambda x: x['Sentiment'].nunique() > 1)['PhraseId'].nunique()
test_data = pd.read_csv('../input/test.tsv', delimiter='\t')
test_data.head()