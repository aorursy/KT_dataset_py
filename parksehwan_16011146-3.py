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

from sklearn.neighbors import KNeighborsClassifier
test = pd.read_csv("../input/logistic-classification-diabetes-knn/test_data.csv")

train = pd.read_csv("../input/logistic-classification-diabetes-knn/train.csv")

train_x = train.loc[:,train.keys()[1:-1]]

train_y = train.loc[:,train.keys()[-1]]

test_x = test.loc[:,test.keys()[1:-1]]
knn = KNeighborsClassifier(n_neighbors=20,p=2)

knn.fit(train_x,train_y)
testresult = knn.predict(test_x)
testresult = testresult.astype(np.int32)

id = np.array([i for i in range(0,50)])

sub = np.hstack([id.reshape(50,1),testresult.reshape(50,1)])

submission_form = pd.DataFrame(sub,columns=["ID","Label"])

submission_form.to_csv("submission_form.csv",mode='w',header=True,index=False)
submission_form