# Internet ON.



# !pip install -U pip

# !pip install evaluations





# Internet OFF. 

# You can use evaluations dataset (see input folders) (the same as in pypi).



!pip install --no-deps '../input/evaluations/'
import random



from evaluations.kaggle_2020 import global_average_precision_score

import pandas as pd
df_train = pd.read_csv('../input/landmark-recognition-2020/train.csv', index_col=0)

df_train
correct_labels = df_train.to_dict()['landmark_id']
predicted_labels = {key: (item, random.random()) for key, item in correct_labels.items()}
global_average_precision_score(correct_labels, predicted_labels)
predicted_labels = {}

for ind, (key, item) in enumerate(correct_labels.items()):

    if ind % 2:

        predicted_labels[key] = (item, random.random())

    else:

        predicted_labels[key] = (1, random.random())
global_average_precision_score(correct_labels, predicted_labels)
labels = df_train['landmark_id'].unique().tolist()

predicted_labels = {key: (random.sample(labels, 1)[0], random.random()) for key, item in correct_labels.items()}
global_average_precision_score(correct_labels, predicted_labels)