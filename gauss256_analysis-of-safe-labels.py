import pandas as pd
def check_new_files(t):

    t['patient'] = t['image'].apply(lambda x: x.split('_')[0])

    t['dataset'] = t['image'].apply(lambda x: 'old_train' if len(x.split('_'))==3 else 'old_test')

    stats = t.groupby(['patient', 'dataset', 'class', 'safe']).size().reset_index()

    stats.rename(columns={0: 'count'}, inplace=True)

    return stats
try:

    t = pd.read_csv('../input/train_and_test_data_labels_safe.csv')

    stats = check_new_files(t)

    stats

except OSError as e:

    print('Safe labels file not found')

    print(e)

    print('Screen shot of safe label analysis is below')
