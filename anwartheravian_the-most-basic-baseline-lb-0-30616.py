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
train = pd.read_csv('../input/train.csv')
train.groupby('is_churn').count()
63471/float(929460)
submission = pd.read_csv('../input/sample_submission_zero.csv')
submission['is_churn'] = 0.06828803821573817

submission.head()
submission.to_csv('most_basic_baseline_prediction.csv',index=False)