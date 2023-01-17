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
#packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas_profiling
#Data upload
test = pd.read_csv("../input/slashing-prices/Test.csv")
train = pd.read_csv("../input/slashing-prices/Train.csv")
train.head(5)
train.shape
train.describe()
sns.heatmap(train.corr(),annot=True)
sns.lineplot(x="Grade",y="High_Cap_Price",data=train)
sns.scatterplot(x="Grade",y="Market_Category",data=train)
train.columns
X = train[['State_of_Country', 'Market_Category','Product_Category', 'Grade', 'Demand','High_Cap_Price']]
Y = train[['Low_Cap_Price']]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.3, random_state = 30)
pro = train.profile_report(title="NEW")
pro
#test data set
test.head()
sns.heatmap(test.corr(),annot=True)
x = test[['State_of_Country', 'Market_Category','Product_Category', 'Grade', 'Demand']]
y = test[['High_Cap_Price']]
sns.lineplot(x="Grade",y="High_Cap_Price",data=test)
x_train,x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 30)
pro1 = test.profile_report(title="Test report")
pro1
