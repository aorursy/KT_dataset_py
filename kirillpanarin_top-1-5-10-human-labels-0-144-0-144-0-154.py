import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))
sample_submission_df = pd.read_csv('../input/stage_1_sample_submission.csv', index_col='image_id')
sample_submission_df.head()
labels = pd.read_csv('../input/train_human_labels.csv', index_col='ImageID')
labels.count()
top_5_labels = ' '.join(labels.groupby(['LabelName']).sum().sort_values(['Confidence'], ascending=False).head(5).index.values)
top_1_labels = ' '.join(labels.groupby(['LabelName']).sum().sort_values(['Confidence'], ascending=False).head(1).index.values)
top_10_labels = ' '.join(labels.groupby(['LabelName']).sum().sort_values(['Confidence'], ascending=False).head(10).index.values)
sample_submission_df['labels'] = top_1_labels
sample_submission_df.to_csv('top_1_human_labels.csv')
sample_submission_df['labels'] = top_5_labels
sample_submission_df.to_csv('top_5_human_labels.csv')
sample_submission_df['labels'] = top_10_labels
sample_submission_df.to_csv('top_10_human_labels.csv')