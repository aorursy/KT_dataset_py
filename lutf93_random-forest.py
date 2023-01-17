# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestClassifier 

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

X_submission=pd.read_csv('../input/test.csv')

X_train,X_test,y_train,y_test=train_test_split(train.iloc[:,1:],train['label'],random_state=0)
model=RandomForestClassifier().fit(X_train,y_train)

model.score(X_test,y_test)
submission=pd.DataFrame(model.predict(X_submission),index=range(1,28001),columns=['Label'])

submission.index.name='ImageId'

submission.to_csv('submission11.csv')
print(check_output(["ls", "../output"]).decode("utf8"))