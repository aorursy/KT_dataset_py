# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")



target_label = "SalePrice"

features = [col for col in train_data.columns if col != target_label]



train_features = train_data[features]

train_output = train_data[target_label]



test_features = test_data[features]
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()



for i, c in enumerate(train_features.dtypes):

    if c.hasobject:

        print(enc.fit_transform(train_features[features[i]]))

        #train_features[features[i]] = enc.fit_transform(train_features[features[i]])

        #print(train_features[features[i]])

    #enc.fit_transform(train_features["SaleType"])