# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))

# print(check_output(["ls", "../input/resultcsv"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

sub = pd.read_csv('../input/commit-1/result2.csv')

sub[sub.duplicated('test_id')].sort_values('test_id')

# sub=sub.fillna(0)

# sub[pd.isnull(sub['price'])]

sub.to_csv("submition2.csv",index=False)