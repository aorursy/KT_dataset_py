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
import pandas as pd

import numpy as np

cover = pd.read_csv("../input/train.csv")

cover
cover.info()
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
import pandas as pd

import numpy as np

cover = pd.read_csv("../input/train.csv")

cover
cover.describe()

cover.info()
X = cover[cover.columns[0:11]]

Y = cover["Survived"]

cover.isnull().sum()

cover['Cabin']

cover['Cabin'].mode()
cover['Cabin'].describe()
cover['Cabin'].fillna("B96B98",inplace=True)

cover.isnull().sum()
cover['Embarked'].describe()
cover['Embarked'].fillna("S",inplace=True)

cover.isnull().sum()
cover
cover["Embarked"].map({"S":"1", "C":"2","Q":"3"})
cover
cover