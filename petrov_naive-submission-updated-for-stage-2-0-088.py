import pandas as pd

top_n = 3

tuning_labels = pd.read_csv('../input/tuning_labels.csv', names=['id', 'labels'], index_col=['id'])

# calculate top_n most popular labels
predicted = ' '.join(
    tuning_labels['labels']
    .str
    .split()
    .apply(pd.Series)
    .stack()
    .value_counts()
    .head(top_n)
    .index
    .tolist()
)

print(f'{top_n} most popular labels are: {predicted}')
submission = pd.read_csv('../input/stage_1_sample_submission.csv', index_col='image_id')

# tuning table is part of submission.csv
submission.index.isin(tuning_labels.index).sum()
# use most popular labels as a prediction unless the correct labels are provided
submission['labels'] = predicted
submission.update(tuning_labels)
submission.to_csv('naive.csv')
(
    pd.read_csv('https://storage.googleapis.com/inclusive-images-stage-2/stage_2_sample_submission.csv', index_col='image_id')
    .assign(labels=predicted)
    .to_csv('stage_2_submission.csv')
)