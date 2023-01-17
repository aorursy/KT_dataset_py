import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



paths = []

ids = []

for dirname, _, filenames in os.walk('/kaggle/input/atml2020-assignment-2/test'):

    for filename in filenames:

        paths.append(os.path.join(dirname, filename))

        ids.append(filename.split('.')[0])
def predict_random(path):

    """Returns a randomn prediction"""

    return np.random.randint(100)
predictions = []



for path in paths:

    prediction = predict_random(path)

    predictions.append(prediction)
# ids - list of ids

# predictions - list of corresponding predictions

# the following lines save the ids and corresponding predictions in a csv file



with open('test_results.csv', 'w') as f:

    f.write('id,label\n')

    for id, pred in zip(ids, predictions):

        f.write(f'{id},{pred}\n')

        

## File can be downloaded from output directory on the right