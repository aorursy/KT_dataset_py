# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

def read_data(filname, limit=None):
    data = []
    labels = []

    csv_reader = csv.reader(open(filname, "r"), delimiter=",")
    index = 0
    for row in csv_reader:
        index += 1
        if index == 1:
            continue

        labels.append(int(row[0]))
        row = row[1:]

        data.append(np.array(np.int64(row)))

        if limit != None and index == limit + 1:
            break

    return (data, labels)

print("Reading train data")
train, target = read_data("../input/train.csv")

# Any results you write to the current directory are saved as output.
print(train)