import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from lwlwrap import calculate_overall_lwlrap_sklearn



from sklearn.preprocessing import MultiLabelBinarizer
curated_df = pd.read_csv("../input/freesound-audio-tagging-2019/train_curated.csv")

#noisy_df = pd.read_csv("../input/freesound-audio-tagging-2019/train_noisy.csv")



sample_df = pd.read_csv("../input/freesound-audio-tagging-2019/sample_submission.csv")



df = pd.concat([curated_df])
mlb = MultiLabelBinarizer()

true_labels = mlb.fit_transform(df['labels'].str.split(","))

all_classes = mlb.classes_
calculate_overall_lwlrap_sklearn(true_labels, true_labels)
calculate_overall_lwlrap_sklearn(true_labels, np.zeros_like(true_labels))
label_means = np.mean(true_labels, axis=0)

predicted_labels = np.repeat([label_means], len(df), axis=0)
calculate_overall_lwlrap_sklearn(true_labels, predicted_labels)
submission_labels = np.repeat([label_means], len(sample_df), axis=0)

submission = pd.DataFrame(submission_labels)

submission.columns = mlb.classes_

submission.insert(0, 'fname', sample_df['fname'])
submission.to_csv("submission.csv", index=False)