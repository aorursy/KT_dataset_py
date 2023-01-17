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
im_df=pd.read_csv('../input/digit-recognizer/train.csv')

im_df.head()
target=im_df['label']
im_df.drop(['label'],axis=1,inplace=True)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
x_train,x_test,y_train,y_test=train_test_split(im_df,target,test_size=0.3,random_state=43)
lr=LogisticRegression(multi_class='multinomial',solver='newton-cg')
lr.fit(x_train,y_train)
y_pre=lr.predict(x_test)
te_df=pd.read_csv('../input/digit-recognizer/test.csv')
y_te=lr.predict(te_df)
ss_df=pd.read_csv('../input/digit-recognizer/sample_submission.csv')
ss_df.head()
ss_df['Label']=y_te
ss_df.to_csv('/kaggle/working/submission.csv',index=False)