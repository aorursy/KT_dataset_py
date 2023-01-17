import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import json



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
base_path = '/kaggle/input/tensorflow2-question-answering/'

train_file_name = 'simplified-nq-train.jsonl'
train_records = []

max_records = 10

current_record = 1



with open(os.path.join(base_path, train_file_name)) as file:

    line = file.readline()

    while(line):

        train_records.append(json.loads(line))

        line = file.readline()

        if current_record > max_records :

            break

        current_record = current_record + 1



df_train = pd.DataFrame(train_records)
df_train.head()