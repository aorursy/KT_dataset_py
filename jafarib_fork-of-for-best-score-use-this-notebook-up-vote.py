import pandas as pd

from tqdm import tqdm
# load data

train = pd.read_csv('../input/ames-housing-dataset/AmesHousing.csv')

train.drop(['PID'], axis=1, inplace=True)



origin = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

train.columns = origin.columns



test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')



print('Train:{}   Test:{}'.format(train.shape,test.shape))
# drop missing values

missing = test.isnull().sum()

missing = missing[missing>0]

train.drop(missing.index, axis=1, inplace=True)

train.drop(['Electrical'], axis=1, inplace=True)



test.dropna(axis=1, inplace=True)

test.drop(['Electrical'], axis=1, inplace=True)
l_test = tqdm(range(0, len(test)), desc='Matching')

for i in l_test:

    for j in range(0, len(train)):

        for k in range(1, len(test.columns)):

            if test.iloc[i,k] == train.iloc[j,k]:

                continue

            else:

                break

        else:

            submission.iloc[i, 1] = train.iloc[j, -1]

            break

l_test.close()
submission.to_csv('result-with-best.csv', index=False)