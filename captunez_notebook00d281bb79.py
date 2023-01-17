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
reader = pd.read_csv('../input/page_views_sample.csv',iterator=True)

page_df = reader.get_chunk(50)

page_df.head()
reader = pd.read_csv('../input/clicks_train.csv',iterator=True)

click_df = reader.get_chunk(50)

click_df.head()