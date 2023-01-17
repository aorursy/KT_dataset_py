import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Grabs training and testing data

training_data_file_path = "/kaggle/input/train.csv"

testing_data_file_path = "/kaggle/input/test.csv"

#train_data = pd.read_csv(training_data_file_path)

test_data = pd.read_csv(testing_data_file_path)



test_data.describe()

#train_data.describe()


