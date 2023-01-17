# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from tqdm import tqdm



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/landmark-recognition-2020/train.csv')

df
labels = np.unique(np.array(df['landmark_id'], dtype=int))

print(labels.size)

np.sort(labels)
"""Global Average Precision for Google Landmark Recognition 2020

https://www.kaggle.com/c/landmark-recognition-2020/overview"""

"""

Thanks to https://github.com/yisaienkov/evaluations/blob/master/evaluations/kaggle_2020/global_average_precision.py

"""



from typing import Dict, Tuple, Any





def global_average_precision_score(

        y_true: Dict[Any, Any],

        y_pred: Dict[Any, Tuple[Any, float]]

) -> float:

    """

    Compute Global Average Precision score (GAP)

    Parameters

    ----------

    y_true : Dict[Any, Any]

        Dictionary with query ids and true ids for query samples

    y_pred : Dict[Any, Tuple[Any, float]]

        Dictionary with query ids and predictions (predicted id, confidence

        level)

    Returns

    -------

    float

        GAP score

    Examples

    --------

    >>> from evaluations.kaggle_2020 import global_average_precision_score

    >>> y_true = {

    ...         'id_001': 123,

    ...         'id_002': None,

    ...         'id_003': 999,

    ...         'id_004': 123,

    ...         'id_005': 999,

    ...         'id_006': 888,

    ...         'id_007': 666,

    ...         'id_008': 666,

    ...         'id_009': None,

    ...         'id_010': 666,

    ...     }

    >>> y_pred = {

    ...         'id_001': (123, 0.15),

    ...         'id_002': (123, 0.10),

    ...         'id_003': (999, 0.30),

    ...         'id_005': (999, 0.40),

    ...         'id_007': (555, 0.60),

    ...         'id_008': (666, 0.70),

    ...         'id_010': (666, 0.99),

    ...     }

    >>> global_average_precision_score(y_true, y_pred)

    0.5479166666666666

    """

    indexes = list(y_pred.keys())

    indexes.sort(

        key=lambda x: -y_pred[x][1],

    )

    queries_with_target = len([i for i in y_true.values() if i is not None])

    correct_predictions = 0

    total_score = 0.

    for i, k in tqdm(enumerate(indexes, 1)):

        relevance_of_prediction_i = 0

        if y_true[k] == y_pred[k][0]:

            correct_predictions += 1

            relevance_of_prediction_i = 1

        precision_at_rank_i = correct_predictions / i

        total_score += precision_at_rank_i * relevance_of_prediction_i



    return 1 / queries_with_target * total_score
# Generate some random predictions on 3 classes

np.random.seed(2020)

ypred = np.random.choice([1,2,3], 10)

ytrue = np.random.choice([1,2,3], 10)

conf = np.random.random(10)





global_average_precision_score(y_true = {str(idx): ytrue[idx] for idx in range(10)}, y_pred = {str(idx): (ypred[idx], conf[idx]) for idx in range(10)})
train_labels = {df['id'].iloc[idx]: df['landmark_id'].iloc[idx] for idx in tqdm(range(len(df)))}

dummy_labels = {df['id'].iloc[idx]: (np.random.choice(labels, 1), np.random.random(1)) for idx in tqdm(range(len(df)))}

global_average_precision_score(train_labels, dummy_labels)