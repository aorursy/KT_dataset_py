# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import datetime

fmt = '%Y-%m-%d %H %m'
t1 = datetime.datetime(2012, 1, 25, 24, 1)
print(t1.strftime(fmt))
df = pd.DataFrame({"t": ["2020-01-01 24:01", "2020-01-01 23:59"]})
def parse_date(s):
    try:
        d = datetime.datetime.strptime(s, "%Y-%m-%d %H:%M")
    except ValueError:
        match = re.findall(r'(\d+)',s)
        year = int(match[0])
        month = int(match[1])
        day = int(match[2])
        hour = int(match[3])
        minute = int(match[4])
        if hour >= 24:
            hour = hour - 24
            day = day + 1
        d = datetime.datetime(year, month, day, hour, minute)
    return d

df.t.apply(parse_date)
