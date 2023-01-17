#from pandas_profiling import ProfileReport



#prof = ProfileReport(df)



#prof.to_file(output_file='output.html')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas_profiling import ProfileReport



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

train = pd.read_csv("../input/titanic/train.csv")



test = pd.read_csv("../input/titanic/test.csv")



submission = pd.read_csv("../input/titanic/gender_submission.csv")
profile_train = ProfileReport(train, title='Pandas Profiling Report')
profile_train
profile_train.to_widgets()
profile_train.to_file("train_output1.html")
profile_test = ProfileReport(test, title='Pandas Profiling Report')
profile_test
profile_test.to_widgets()
profile_test.to_file("output1.html")
submission = ProfileReport(submission, title='Pandas Profiling Report')
submission