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
data = pd.read_csv("../input/password-strength-classifier-dataset/data.csv",',',error_bad_lines=False)
data
data.isnull().sum()
data.dropna(inplace = True)
password_tuple = np.array(data)
password_tuple
import random

random.shuffle(password_tuple)
y = [labels[1] for labels in password_tuple ]
x = [labels[0] for labels in password_tuple ]
import seaborn as sns
sns.set_style('whitegrid')

sns.countplot(x = 'strength' , data = data)
def word_char(inputs):

    a= []

    for i in inputs:

        a.append(i)

    return a
from sklearn.feature_extraction.text import TfidfVectorizer

vect = TfidfVectorizer(tokenizer = word_char)

x =vect.fit_transform(x)
x.shape
import xgboost as xgb
from sklearn.model_selection import train_test_split
X_train,X_test ,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)
xg = xgb.XGBClassifier()
xg.fit(X_train,y_train)
xg.score(X_test,y_test)
y_pred=xg.predict(X_test)