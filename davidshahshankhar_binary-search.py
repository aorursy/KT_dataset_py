# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
def search_binary(sorted_array, target):

    left = 0

    right = len(sorted_array) - 1

    while left <= right:

        midpoint = left + (right - left) // 2

        current = sorted_array[midpoint]

        if current == target:

            return midpoint

        else:

            if target < current:

                right = midpoint - 1

            else:

                left = midpoint + 1

    return None

 

target = 5

sorted_array = [0, 1, 2, 3, 4, 5]

result = search_binary(sorted_array, target)

if result is not None:

    print('Value {} found at position {} using binary search'.format(target, result+1))

else:

    print('Not found')