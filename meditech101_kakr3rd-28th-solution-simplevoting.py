# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
def get_winner(cand_main, cand_sub1, cand_sub2):

    if(cand_sub1==cand_sub2):

        return cand_sub1

    else:

        return cand_main
df3 = pd.read_csv('../input/submissions/submission_LB94835.csv')

df2 = pd.read_csv('../input/submissions/submission_LB95026.csv')

df1 = pd.read_csv('../input/submissions/submission_LB95456.csv')
subm_df=pd.DataFrame(columns=['img_file', 'class', 'class1', 'class2', 'class3'])
subm_df['img_file']=df1['img_file']
subm_df['class1']=df1['class']

subm_df['class2']=df2['class']

subm_df['class3']=df3['class']
subm_df['class']=subm_df.apply(lambda row : get_winner(row['class1'], row['class2'], row['class3']), axis=1)
subm_df.drop(['class1','class2', 'class3'], axis=1, inplace=True)

subm_df.to_csv('./submission.csv')