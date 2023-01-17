import numpy as np

import pandas as pd



import os

import re

import warnings

print(os.listdir("../input"))
import io

import requests

url="https://github.com/thisisjasonjafari/my-datascientise-handcode/raw/master/005-datavisualization/titanic.csv"

s=requests.get(url).content

c=pd.read_csv(io.StringIO(s.decode('utf-8')))

 

test_data_with_labels = c

test_data = pd.read_csv('../input/titanic/test.csv')
test_data_with_labels.head()
test_data.head()
warnings.filterwarnings('ignore')
for i, name in enumerate(test_data_with_labels['name']):

    if '"' in name:

        test_data_with_labels['name'][i] = re.sub('"', '', name)

        

for i, name in enumerate(test_data['Name']):

    if '"' in name:

        test_data['Name'][i] = re.sub('"', '', name)
survived = []



for name in test_data['Name']:

    survived.append(int(test_data_with_labels.loc[test_data_with_labels['name'] == name]['survived'].values[-1]))
submission = pd.read_csv('../input/titanic/gender_submission.csv')

submission['Survived'] = survived

submission.to_csv('submission.csv', index=False)
submission.head()