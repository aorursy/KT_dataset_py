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
import pandas as pd

from sklearn import datasets

datadict=datasets.load_breast_cancer()

datadict.keys()

x=datadict['data']

y=datadict['target']

pd.DataFrame(x, columns=datadict['feature_names']).head()



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=0.3,random_state=42)

print('Traning data: ', x_train.shape,y_train.shape)

print('Testing data: ', x_test.shape,y_test.shape)



from sklearn import linear_model

model=linear_model.LogisticRegression()

model.fit(x_train,y_train)



import numpy as np

predictions=model.predict(x_test )

accuracy=np.mean(y_test==predictions)

accuracy
