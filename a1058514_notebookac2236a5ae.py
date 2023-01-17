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





train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

# combine = [train_df, test_df]



train_df['color'] = train_df['color'].map({'clear':0, 'white':1, 'green':2, 'blood':3, 'blue':4, 'black':5})

import seaborn as sns

sns.set()

sns.pairplot(train_df, hue="type")



#print(combine)

#print(test_df.columns.values)

# train_df.head()



# train_df.info()

# print('_'*40)

# test_df.info()



#What is the distribution of numerical feature values across the samples?

# train_df.describe()



#What is the distribution of categorical features?

# train_df.describe(include=['O'])





# submission = pd.DataFrame({

#         "id": test_df["id"],

#         "type": "Ghoul"

#     })

#print(submission)

#submission.to_csv('../output/submission.csv', index=False)

# submission.to_csv('submission.csv', index=False)




