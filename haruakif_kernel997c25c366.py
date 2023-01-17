# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/ai-introduction-hit-u/pokemon_train.csv")

#kaggleとかGoogle Colabとか使うときは、ファイルのパスは上に黒い四角に出るのでそれをコピーして使おう
df.head()
data=df.iloc[:,6:11].values

legend=df.iloc[:,13].values
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split



X_train, X_test,y_train,y_test=train_test_split(data,legend,random_state=0)



model=SVC(kernel="rbf",C=5,gamma="auto")

model.fit(X_train,y_train)



print(model.score(X_train,y_train))

print(model.score(X_test,y_test))
dtest=pd.read_csv("/kaggle/input/ai-introduction-hit-u/pokemon_test.csv")

dtest.head()
tdata=dtest.iloc[:,6:11].values
l=model.predict(tdata)
print(l) # これが予測結果
id=np.arange(1,201)#1~200の配列を作る

df = pd.DataFrame(data=np.transpose([id, l]), columns=['id', 'LABEL'])

df.to_csv("submit.csv",index=False) #submit.csvとして保存。index=Falseは1列目にindex(0~199)が入らないように