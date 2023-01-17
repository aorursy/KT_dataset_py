# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# find size of groups by their powerball (PB) number

df = pd.read_csv('../input/pb_winning_numbers_03-18-2017.csv')

out_pb = df.groupby(['PB'])

print(out_pb.size())
# print the group where PB is 11

print(out_pb.get_group(11))
# has the number 11 been the Powerball number in the last 90 days in any drawings

import datetime

import dateutil

from dateutil import parser

from datetime import date

group_DD = out_pb.get_group(13).groupby(['DrawDate'])

for name,group in group_DD:

	d1 = parser.parse(name)

	d2 = datetime.datetime.now()

	delta = (d2-d1).days

	if (delta < 90):

		print(name+' was ' + str(delta) + ' days ago')