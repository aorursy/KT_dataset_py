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
import numpy as np

import pandas as pd



#Print you can execute arbitrary python code

train_df = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )

test_df = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )
#Print to standard output, and see the results in the "log" section below after running your script

print("Top of the training data:")

print(train_df.head(5))
print("\n\nSummary statistics of training data")

print(train_df.describe())
train_df.info()
#Any files you save will be available in the output tab below

train.to_csv('copy_of_the_training_data.csv', index=False)