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
clicks_train = pd.read_csv("../input/clicks_train.csv")

clicks_train.head()
clicks_train.count()
clicks_train['display_id'].value_counts()
clicks_train['ad_id'].value_counts()
clicked = clicks_train['clicked']
import seaborn as sns
clicks_test = pd.read_csv("../input/clicks_test.csv")