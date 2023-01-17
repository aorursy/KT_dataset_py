import pandas as pd
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train['month'] = pd.DatetimeIndex(train['Date']).month
test['month'] = pd.DatetimeIndex(test['Date']).month
mean_vals = train.groupby(['Store', 'Dept', 'month', 'IsHoliday']).median()
mean_vals.dtypes
merged = test.merge(mean_vals,
                  how = 'left',
                  left_on = ['Store', 'Dept', 'month', 'IsHoliday'],
                  right_index = True,
                  sort = False,
                  copy = False)
index = pd.DataFrame({'id':merged['Store'].map(str) + '_' + merged['Dept'].map(str) + '_' + merged['Date'].map(str)})
submission = merged.join(index)
submission.to_csv('submission.csv')