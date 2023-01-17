# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

gt_df = pd.read_csv("../input/disasters-on-social-media/socialmedia-disaster-tweets-DFE.csv")

test_df = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
gt_df = gt_df[['choose_one', 'text']]

gt_df['target'] = (gt_df['choose_one']=='Relevant').astype(int)

gt_df['id'] = gt_df.index

gt_df
merged_df = pd.merge(test_df, gt_df, on='id')

merged_df
subm_df = merged_df[['id', 'target']]

subm_df
subm_df.to_csv('submission.csv', index=False)