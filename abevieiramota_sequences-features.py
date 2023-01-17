import pandas as pd
import numpy as np
%matplotlib inline
train = pd.read_csv('../input/train.csv', index_col='Id')
seqs = {ix: pd.Series(x['Sequence'].split(',')) for ix, x in train.iterrows()}
train['SequenceSize'] = [len(seq) for seq in seqs.values()]
train['Mode'] = [seq.value_counts().idxmax() for seq in seqs.values()]
train['LastValue'] = [seq.iloc[-1] for seq in seqs.values()]
train['IsLastValueEqMode'] = train.apply(lambda x: x['LastValue'] == x['Mode'], axis=1)
train['NDifferentValues'] = [seq.value_counts().shape[0] for seq in seqs.values()]
train['SeqValuesSizeMean'] = [seq.apply(lambda x: len(x)).mean() for seq in seqs.values()]
train['SeqValuesSizeMax'] = [seq.apply(lambda x: len(x)).max() for seq in seqs.values()]
train['SeqValuesSizeMin'] = [seq.apply(lambda x: len(x)).min() for seq in seqs.values()]
ax = train[train.IsLastValueEqMode]\
.plot(kind='scatter', x='SequenceSize', y='NDifferentValues', color='Blue', figsize=(12, 12), \
      label='LastValue = Mode', xlim=(0, 400), ylim=(0, 200))
train[~train.IsLastValueEqMode]\
.plot(kind='scatter', x='SequenceSize', y='NDifferentValues', color='Red', ax=ax, \
      label='LastValue <> Mode', alpha=.5)
