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
!pip install pycaret
data = pd.read_csv("/kaggle/input/titanic/train.csv")
data.head()
from pycaret.classification import *
clf1 = setup(data, target="Survived", ignore_features=["Name", "Ticket", "PassengerId"])
compare_models()
tuned_gbc = tune_model('gbc', optimize='AUC')
evaluate_model(tuned_gbc)
final_gbc = finalize_model(tuned_gbc)
print(final_gbc)
test = pd.read_csv('/kaggle/input/titanic/test.csv')
test.head()
predictions = predict_model(final_gbc, data=test)
predictions.head()
submissions = predictions[['PassengerId', 'Score']]
submissions['Survived'] = submissions['Score'].apply(lambda x: 1 if x>0.41 else 0)
submissions = submissions.drop(columns="Score")
submissions.head()
submissions.to_csv('submissions_pycaret.csv', index=False)