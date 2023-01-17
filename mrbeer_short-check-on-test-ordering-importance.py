import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

dtypes = {
    'ip'            : 'uint32',
    'app'           : 'uint16',
    'device'        : 'uint16',
    'os'            : 'uint16',
    'channel'       : 'uint16',
    'is_attributed' : 'uint8',
    'click_id'      : 'uint32'
}

print('Loading the test data...')
test = pd.read_csv("../input/test.csv", dtype=dtypes)
print("The percentage of not ordered index is {0}".format(np.mean(test.index.values != test.click_id.values)))
test[test.index.values != test.click_id.values]
N_SIM = 2500000
results = np.zeros((N_SIM)).astype(np.int8)
THRESH = 0.0025
np.random.seed(42)
results = (np.random.uniform(size=N_SIM) < THRESH).astype(int)
predictions = results[test["click_id"].values[:N_SIM]] 
from sklearn.metrics import roc_auc_score
roc_auc_score(results, predictions)