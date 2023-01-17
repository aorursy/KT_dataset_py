# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train=pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020-v2/train_data.csv')
test=pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020-v2/test_data.csv')
sample_submission=pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020-v2/sample_submission.csv')

train.head()
test.head()
print(train.shape, test.shape)
train.info()
test.info()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
X=train.drop('price_range',axis=1).values
y=train['price_range']
data_test=pd.read_csv('../input/mobile-price-range-prediction-is2020-v2/test_data.csv')
predicted_price=kn.predict(data_test)
data_test['price_range']=predicted_price
data_test=pd.DataFrame({'id':id_test,'price_range':predicted_price})
data_test.to_csv('output.csv',index=False)
data_test