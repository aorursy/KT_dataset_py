import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
train_raw = pd.read_csv('../input/train.csv')
test_raw = pd.read_csv('../input/test.csv')
sample = pd.read_csv('../input/sample_submission.csv')
def toArray(s):
    return np.array(s.split(','))
test = test_raw
test['Sequence'] = test['Sequence'].apply(toArray)
1