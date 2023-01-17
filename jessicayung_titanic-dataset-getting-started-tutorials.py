# Load data

import csv as csv

import numpy as np



# Open file in text mode (default 'rt', not 'rb' which is binary mode,

# i.e. bytes)

csv_file_object = csv.reader(open('train.csv'))

header = next(csv_file_object)

data=[]



for row in csv_file_object:

    data.append(row)

data = np.array(data)
# Import test data

test_df = pd.read_csv('test.csv')