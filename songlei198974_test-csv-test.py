%%time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn import linear_model



from numpy import genfromtxt,savetxt



dataset = pd.read_csv("../input/train.csv")

target = dataset.iloc[:,0]

train = dataset.iloc[:,1:]



test = pd.read_csv("../input/test.csv")



clf = linear_model.SGDClassifier()

clf.fit(train,target) # 784

rs = clf.predict(test)

sm = pd.DataFrame({'ImageId': range(1,len(rs)+1), 'Label':rs})



sm.to_csv('mine_submission.csv',index=False)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))