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
df_original =pd.read_csv("../input/train.csv")
df_test =  pd.read_csv("../input/test.csv")
X = df_original.drop(['label'],axis=1)
y = df_original['label']


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
df_scaled=scaler.transform(X)


from sklearn.decomposition import PCA
pca = PCA(svd_solver = 'auto')
pca.fit(df_scaled)
principalComponents = pca.transform(df_scaled)

from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(random_state=1211)
model_rf.fit(X ,y)
y_pred = model_rf.predict(df_test)
import numpy as np

ImageId = np.arange(1,28001)
Label = y_pred
ImageId = pd.Series(ImageId)
Label = pd.Series(Label)
submit = pd.concat([ImageId,Label],axis=1, ignore_index=True)
submit.columns=['ImageId','Label']
submit.to_csv("RFandPCA.csv",index=False)