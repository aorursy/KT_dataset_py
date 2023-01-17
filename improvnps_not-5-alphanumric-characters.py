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
df_train = train = pd.read_csv('../input/train.tsv', sep='\t')

df_test = test = pd.read_csv('../input/test.tsv', sep='\t')



train['target'] = np.log1p(train['price'])
train.describe()

test
submit=pd.DataFrame(test["test_id"])
submit.shape
submit.loc[:,"price"]=26.73
submit
submit.to_csv('sample_submission.csv',index=False)