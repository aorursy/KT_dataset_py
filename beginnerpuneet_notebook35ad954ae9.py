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

# read data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



features = [x for x in train.columns if x not in ['id','loss']]

#print(features)



cat_features = [x for x in train.select_dtypes(include=['object']).columns if x not in ['id','loss']]

num_features = [x for x in train.select_dtypes(exclude=['object']).columns if x not in ['id','loss']]
test.head()
tr