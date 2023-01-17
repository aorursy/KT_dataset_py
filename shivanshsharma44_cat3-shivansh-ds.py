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
print("CAT 3 DATA SCIENCE 9TH OCTOBER \n Group Number 10 Activity 5 \n Shivansh Sharma 18scse1180037/18021180036 \n Kaggle link-\n https://www.kaggle.com/shivanshsharma44/cat3-shivansh-ds ")
 # 1. Read the dataset.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv('/kaggle/input/world-happiness/2018.csv')
df
# 2. Describe the dataset. 
df.describe(include = 'all')
# 3. Find mean,median and mode of columns.
df.mean()
# 3. Find mean,median and mode of columns. 
df.median()
# 3. Find mean,median and mode of columns. 
df.mode ()
# 4. Find the distribution of columns with numerical data. Evaluate if they are normal or not.
# Numeric Dataset and these features are normal
numeric = list(df._get_numeric_data().columns)
numeric
# 4. Find the distribution of columns with numerical data. Evaluate if they are normal or not.
# Ordinal columns
# columns which are not normal are ordinal
categorical = list(set(df.columns) - set(df._get_numeric_data().columns))
categorical
# 5. Draw different types of graphs to represent information hidden in the dataset.
df.plot.hist()
# 5. Draw different types of graphs to represent information hidden in the dataset.
df.plot.bar()
# 6. Find columns which are correlated.
# 7. Find columns which are not correlated.
corr=df.corr()
corr
# 8. Compare different columns of dataset
comparison_column = np.where(df['Freedom to make life choices'] == df['Generosity'], True, False)
df["equal"] = comparison_column
df
print("Is any supervised machine learning possible? if yes explain.\n Answer  Supervised machine learning is possible as Supervised learning is where you have input variables (x) and an output variable (Y) and you use an algorithm to learn the mapping function from the input to the output. Y = f(X) The goal is to approximate the mapping function so well that when you have new input data (x) that you can predict the output variables (Y) for that data. It is called supervised learning because the process of an algorithm learning from the training dataset can be thought of as a teacher supervising the learning process. We know the correct answers, the algorithm iteratively makes predictions on the training data and is corrected by the teacher. Learning stops when the algorithm achieves an acceptable level of performance.")