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
from sklearn import preprocessing
gender_age_train = pd.read_csv('../input/gender_age_train.csv')
gender_age_train.head()
gender_age_train.describe()
gender_age_train.info()
gender_age_train['gender_bin'] = gender_age_train['gender'].apply(lambda x: 1 if x == 'F' else 0)
gender_age_train['gender_bin'].value_counts()
gender_age_train['gender_bin'].hist()
gender_age_train['gender'].value_counts()
gender_age_train.hist()
le = preprocessing.LabelEncoder()
gender_age_train['group_encoded'] = le.fit_transform(gender_age_train['group'])
gender_age_train['group_encoded'].hist()
gender_age_train.groupby('group_encoded')[['age','gender_bin']].hist()
