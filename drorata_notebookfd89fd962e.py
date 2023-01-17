import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



%matplotlib inline

from matplotlib import pyplot as plt
train = pd.read_csv('../input/train.csv')

test  = pd.read_csv('../input/test.csv')
train.head()
train.Age.hist();
train.Embarked.value_counts().plot(kind='bar');
train.Fare.hist();
train[train.Fare > 200].Survived.value_counts()
train[train.Fare <= 200].Survived.value_counts()