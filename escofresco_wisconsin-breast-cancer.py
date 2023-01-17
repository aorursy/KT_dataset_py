# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

from matplotlib import pyplot
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# a4_dims = (11.7, 8.27)
# fig, ax = pyplot.subplots(figsize=a4_dims)
sns.set(rc={'figure.figsize':(11.7,8.27)})

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = load_breast_cancer()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df.head()
sns.heatmap(data=df.corrwith(pd.Series(dataset.target, index=df.columns), axis=1))
sns.heatmap(data=df.corr())
X_train, X_test, y_train, y_test = train_test_split(df, dataset.target, test_size=.2, random_state=42)
pipe = make_pipeline(StandardScaler(), SVC(gamma='auto'))
pipe.fit(X_train, y_train)
pipe.score(X_test, y_test)
