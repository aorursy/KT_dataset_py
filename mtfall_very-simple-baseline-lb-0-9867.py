import pandas as pd
train = pd.read_csv('../input/en_train.csv')

test = pd.read_csv('../input/en_test.csv')
d = train.groupby(['before', 'after']).size()

d = d.reset_index().sort_values(0, ascending=False)

d = d.loc[d['before'].drop_duplicates(keep='first').index]

d = d.loc[d['before'] != d['after']]

d = d.set_index('before')['after'].to_dict()
def mapping(x):

    if x in d.keys():

        return d[x]

    else:

        return x



test['after'] = test.before.apply(mapping)
test['id'] = test.sentence_id.astype(str) + '_' + test.token_id.astype(str)

test[['id', 'after']].to_csv('./output.csv', index=False)