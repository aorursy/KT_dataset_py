import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import pandas as pd
train = pd.read_csv('../input/application_train.csv')
test = pd.read_csv('../input/application_test.csv')
import matplotlib.pyplot as plt
plt.plot(train.SK_ID_CURR.diff(),'.')
plt.plot(test.SK_ID_CURR.diff(),'.')
train.SK_ID_CURR.head()
train.SK_ID_CURR.head().diff()
plt.plot(train.SK_ID_CURR.diff(),'.')
plt.plot(test.SK_ID_CURR.diff(),'.')
