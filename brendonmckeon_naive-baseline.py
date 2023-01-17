# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
human_labels = pd.read_csv('../input/train_human_labels.csv')
# human_labels.head()
topN = human_labels.LabelName.value_counts()[:5].index.tolist()
topN
submission = pd.read_csv('../input/stage_1_sample_submission.csv')
submission.head()
images = submission.image_id.tolist()
results = [(i, " ".join(topN)) for i in images]
results_df = pd.DataFrame(results, columns=['image_id', 'labels'])
results_df.to_csv('naive_baseline.csv', index=False)
