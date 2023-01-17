import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import h2o

from h2o.automl import H2OAutoML
h2o.init(max_mem_size='14G')
train = h2o.import_file('../input/learn-together/train.csv')

test = h2o.import_file('../input/learn-together/test.csv')
display(train.head(2))

print(f'Train shape: {train.shape}')



display(test.head(2))

print(f'Test shape: {test.shape}')
train = train.drop('Id')

display(train.head(1))

print(f"Train shape: {train.shape}")
df_id = test['Id']

display(test.head(1))

print(f'Test shape: {test.shape}')
test = test.drop('Id')

x = train.columns

x.remove('Cover_Type')

y = 'Cover_Type'



train[y] = train[y].asfactor()
# df = train.split_frame(ratios=0.8, seed=8)

# df_train = df[0]

# df_valid = df[1]
aml = H2OAutoML(max_runtime_secs=25000)

aml.train(x=x, y=y, training_frame=train)
lb = aml.leaderboard

lb.head(rows=lb.nrows)
aml.leader
preds = aml.predict(test)
preds.head(2)
submission = pd.DataFrame({'Id': df_id.as_data_frame().squeeze(),

                       'Cover_Type': preds['predict'].as_data_frame().squeeze()})



submission.to_csv('submission.csv', index=False)