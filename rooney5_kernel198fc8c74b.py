# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
print(pd.read_csv('../input/digit-recognizer/train.csv').head())

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



print(pd.read_csv('../input/digit-recognizer/test.csv').head())
import pandas as pd

from sklearn.neural_network import MLPClassifier



df=pd.read_csv('../input/digit-recognizer/train.csv')

col = ['pixel%d'%i for i in range(784)]

plf = MLPClassifier((100,100,100),max_iter=5)

plf.fit(df[col],df['label'])



df = pd.read_csv('../input/digit-recognizer/test.csv')

res=plf.predict(df[col])

df = pd.read_csv('../input/digit-recognizer/sample_submission.csv')

df['Label']=res

df.to_csv('submission.csv',index=False)


