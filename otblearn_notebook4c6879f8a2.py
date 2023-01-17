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
import csv



users_act = dict()

with open('../input/train_ver2.csv',"rt", encoding="utf8") as f:

    reader = csv.reader(f)

    for row in reader:          

        if row[1] in users_act.keys():

            users_act[row[1]].append([row[0]] + row[24:])

        else:

            users_act[row[1]]=[[row[0]] + row[24:]]

            

print(len(users_act.keys()))
for user in users_act.values():

    if len(user)>1:

        print(user)

        break