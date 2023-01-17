# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')

train_data.head()
def extract_title(n):

    import re

    # print('name:%s' %n)

    e = re.compile('^[^,]*, ([^ ]+)')

    m = e.search(n)

    if m:

        #print(m.group(1))

        return m.group(1)

    return None

train_data['Title'] = train_data.Name.map(extract_title)
set(train_data['Title'])
train_data.Title.value_counts()