import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
from scipy.stats import ttest_ind
import os
# print(os.listdir("../input"))
df = pd.read_csv('../input/cereal.csv')
df.head()
hot = df['sugars'][df['type']== 'H']
cold = df['sugars'][df['type']== 'C']
ttest_ind(hot, cold, equal_var=False)
plt.hist(hot, label='Hot')
plt.hist(cold, label='Cold', alpha=0.5)
plt.title('Sugar content of hot and cold cereals')
