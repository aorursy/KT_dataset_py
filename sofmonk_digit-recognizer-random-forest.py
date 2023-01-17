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
dataset = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



target = dataset[[0]].values.ravel()

train = dataset.iloc[:,1:].values

from sklearn.ensemble import RandomForestClassifier

##RANDOM FOREST

rf = RandomForestClassifier(n_estimators=100)

rf.fit(train, target)

pred = rf.predict(test)

rf.score(train,target)



submission = pd.DataFrame({

        "ImageId": list(range(1,len(pred)+1)),

        "Label": pred

    })

submission.to_csv("submit.csv", index=False, header=True)