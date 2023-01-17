# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

from sklearn.decomposition import PCA

from sklearn.svm import SVC

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

test_data = test.values



train_data=train.ix[:,1:]

train_target=train.ix[:,0]



#test_data=test.values



pca=PCA(n_components=0.8, whiten=True)

train_data=pca.fit_transform(train_data)

test_data=pca.transform(test_data)





print('fit')

svc=SVC(kernel='rbf',C=2)

svc.fit(train_data,train_target)

print('fit over')

predict=svc.predict(test_data)



pd.DataFrame({"ImageId":range(1,len(predict)+1),'Label':predict}).to_csv('output.csv', index=False, header=True)

print('done')