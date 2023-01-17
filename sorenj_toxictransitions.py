import numpy as np

import pandas as pd
train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
joined = train.dropna(subset=['parent_id']).astype({'parent_id': np.int64}).set_index(

    ['parent_id','id']).join(train.set_index('id'), on='parent_id', rsuffix='_parent')
joined.head(5)
joined.dropna(subset=['target_parent']).plot.scatter('target_parent', 'target')
joined[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']].corr()
joined[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat',

        'target_parent', 'severe_toxicity_parent', 'obscene_parent', 'identity_attack_parent',

        'insult_parent', 'threat_parent']].corr().iloc[0:6,6:]