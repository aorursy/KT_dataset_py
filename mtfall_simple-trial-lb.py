import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv('../input/en_train.csv')

test = pd.read_csv('../input/en_test.csv')
d = train[['before', 'after']].drop_duplicates()

d = d.loc[d.before != d.after]

d = d.set_index('before')['after'].to_dict()
def mapping(x):

    if x in d.keys():

        return d[x]

    else:

        return x



test['after'] = test.before.apply(mapping)
test['id'] = test.sentence_id.astype(str) + '_' + test.token_id.astype(str)
test[['id', 'after']].to_csv('./output.csv', index=False)