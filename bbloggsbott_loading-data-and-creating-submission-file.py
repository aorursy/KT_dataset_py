import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))
data = pd.read_csv('../input/train.csv')
data.head()
test = pd.read_csv('../input/test.csv')
test.head()
import random
pred = [random.random() for i in range(test.shape[0])]
submission = pd.DataFrame()
submission['Id'] = test['Id']
submission['critical_temp'] = pred
submission.head()
submission.to_csv('random_sub.csv', index = False)
os.listdir()