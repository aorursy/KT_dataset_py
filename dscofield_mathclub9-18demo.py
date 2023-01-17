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
df = pd.read_csv("/kaggle/input/titanic/train.csv")
df
df.groupby('Pclass')['Survived'].value_counts()
df.groupby('Pclass')['Survived'].value_counts()/df.groupby('Pclass')['Survived'].count()
df['Predicted_survival'] = np.where(df['Pclass']==1, 1, 0)
df
df["Correct?"] = np.where(df["Survived"] == df["Predicted_survival"], 1, 0)
df["Correct?"].value_counts()/df["Correct?"].count()
Kaggle_test_submit = pd.read_csv("/kaggle/input/titanic/test.csv")
Kaggle_test_submit
Kaggle_test_submit['Predicted_survival'] = np.where(Kaggle_test_submit["Pclass"] ==1, 1, 0)
Kaggle_test_submit
my_submission = pd.DataFrame({'PassengerId':Kaggle_test_submit.PassengerId, 'Survived':Kaggle_test_submit.Predicted_survival})
my_submission.to_csv('Titanic_submission.csv', index=False)
