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
data=pd.read_csv("../input/titanic/train_and_test2.csv")
df=pd.DataFrame(data,columns=['Age','Sex'])
print (df.head(3))
import pandas as pd
data=pd.read_csv("../input/titanic/train_and_test2.csv")
df=pd.DataFrame(data,columns=['Age','Fare','Sex'])
df[df.isnull().any(axis=1)]
import pandas as pd
data=pd.read_csv("../input/titanic/train_and_test2.csv")
df=pd.DataFrame(data,columns=['Age','Fare','Sex','sibsp'])
df.iloc[1:4,[0,1,2,3]]
import pandas as pd
data=pd.read_csv("../input/titanic/train_and_test2.csv")
df=pd.DataFrame(data,columns=['Age','Fare','Sex','sibsp'])
data=data[data.isnull().any(axis=1)].head()
print(data)

exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df=pd.DataFrame(exam_data , index=labels)
print(df)
import pandas as pd
import numpy as np
sample=np.array([1,2,3,4,5])
s=pd.Series(sample)
print("Original array is" )
print(s)
power=s.pow(s)
print("Element wise power")
print(power)