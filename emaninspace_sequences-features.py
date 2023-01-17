import pandas as pd
import numpy as np
%matplotlib inline
train = pd.read_csv('../input/train.csv', index_col='Id')

seqs = {ix: pd.Series(x['Sequence'].split(',')) for ix, x in train.iterrows()}
train['SequenceSize'] = [len(seq) for seq in seqs.values()]
