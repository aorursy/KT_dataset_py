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
sample = pd.read_csv("/kaggle/input/gd-code-battle/sampleSubmission.csv")
sample.head()
sample.value = 0.0
def gen(min_first, plus_first, res=[]):

    print(len(res))

    x = (min_first - plus_first) / 2.0 * 25

    e = plus_first*25 - (100.0 - x)

    data = [x]

    errors = [e]

    for el in res:

        x = (e - 25*el + 1000)/2

        e = e - x

        errors.append(e)

        data.append(x)

    return data + [1000.0], errors

min_first, plus_first, res = 49.83239, 46.09239, []

rows, errors = gen(min_first, plus_first, res)

sample.value[:len(rows)] = rows

sample.head()

sample.to_csv("cur.csv", index=None)