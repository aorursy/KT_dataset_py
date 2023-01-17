# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from tqdm import tqdm



def save_solution(fname, result):

    with open(fname, 'w') as f:

        f.write('id,nextvisit\n')

        for person_id in sorted(list(result.keys())):

            f.write(str(person_id) + ', ' + str(result[person_id]) + '\n')



def solve(train):

    result = {}

    for person_id in train.keys():

        weights = [0] * 7

        for day in train[person_id]:

            weights[(day - 1) % 7] += day / 1099

        for i in range(len(weights)):

            if weights[i] == max(weights):

                result[person_id] = i + 1

                break

    return result

            

data = {}

with open('../input/train.csv', 'r') as f:

    lines = f.readlines()[1:]

    for line in tqdm(lines):

        person_id, days = line.strip().split(',')

        data[int(person_id)] = list(map(int, days.split()))



save_solution('result.csv', solve(data))