# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
X_train = pd.read_csv('../input/train.csv', index_col=0)
#groupby('ほにゃらら')で、ほにゃららの列でまとめた統計値が出せます。とりあえず平均値。

X_train.groupby('grade').mean()
#groupby(['ほにゃらら','なんとか'])で、’ほにゃらら'と'なんとか'の列でまとめた統計値が出せます。

X_train.groupby(['grade','application_type']).mean()
#count()に変えて該当行がどれくらいか見てみます。

X_train.groupby(['grade','application_type']).count()