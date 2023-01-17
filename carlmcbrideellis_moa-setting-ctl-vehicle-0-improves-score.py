train = pd.read_csv('../input/lish-moa/train_features.csv')
test  = pd.read_csv('../input/lish-moa/test_features.csv')
train.at[train['cp_type'].str.contains('ctl_vehicle'),train.filter(regex='-.*').columns] = 0.0
test.at[test['cp_type'].str.contains('ctl_vehicle'),test.filter(regex='-.*').columns] = 0.0