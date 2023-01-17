# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Email metadata fields

# {From:, To:, Cc:, timestamp}
emails = pd.read_csv("../input/emails.csv")

print(type(emails))
emails.head()
emails['message'][1]
fromRegex = re.compile(r'From:\s(\w)+(\.)?(\w)*@(\w)+.com')

toRegex = re.compile(r'To:\s(\w)+(\.)?(\w)*@(\w)+.com')

m_f = fromRegex.search(emails['message'][1])

m_e = toRegex.search(emails['message'][1])

if m_f:

    print(m_f.group())

if m_e:

    print(m_e.group())
len(emails)
from_set = set()

for i in range(len(emails)):

    mf = fromRegex.search(emails['message'][i])

    if mf:

        if mf.group() not in from_set:

            from_set.add(mf.group())

to_set = set()

for i in range(len(emails)):

    me = toRegex.search(emails['message'][i])

    if me:

        if me.group() not in to_set:

            to_set.add(me.group())
print("From emails:", len(from_set))

print("To emails:", len(to_set))
email_set = from_set.union(to_set)

print(len(email_set))