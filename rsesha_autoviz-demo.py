import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
!pip install autoviz
from autoviz.AutoViz_Class import AutoViz_Class
AV = AutoViz_Class()
url1 = '../input/titanic/train.csv'
url2 = '../input/titanic/test.csv'
reduce_train = pd.read_csv(url1,index_col=None)
reduce_test = pd.read_csv(url2,index_col=None)
print(reduce_train.shape,reduce_test.shape)
reduce_train.head()
target='Survived'
dft = AV.AutoViz(depVar=target, dfte=reduce_train, header=0, verbose=0,
                lowess=False,chart_format='svg',max_rows_analyzed=1500000,max_cols_analyzed=300,filename='', sep=',' )
