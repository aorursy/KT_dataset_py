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
import numpy as np
import pandas as pd
import os
data=pd.read_csv("../input/uci-semcom/uci-secom.csv")
raw_data=data.copy()

data.describe(include='all')
nas=data.isnull().sum()
nas.values
data2=data[data.columns[nas < 500]]

data2=data2.fillna(data2.mean())
data2.head()
new_data=data2.drop(['Time'],axis=1)
new_data.loc[(new_data['Pass/Fail'] == -1),'Pass/Fail']=0
new_data.head()
x=new_data.drop(['Pass/Fail'],axis=1)
y=new_data['Pass/Fail']

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_scaled=sc_x.fit_transform(x)
x_scaled
from sklearn.decomposition import PCA
pca = PCA(n_components = 0.95)
pca.fit(x_scaled)
x_reduced = pca.transform(x_scaled)
np.shape(x_reduced)
y.value_counts()
indices_to_remove = []
zero_counter=0
for i in range(x_reduced.shape[0]):
    if y[i] ==0:
        zero_counter += 1
        if zero_counter > 104:
            indices_to_remove.append(i)

np.shape(indices_to_remove)
y1=np.array(y)
x_new= np.delete(x_reduced,indices_to_remove, axis=0)
y_new= np.delete(y1,indices_to_remove, axis=0)
np.shape(y_new)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_new,y_new,test_size=0.2,random_state=0)
import sklearn
from sklearn.linear_model import LogisticRegression
scikit_log_reg = LogisticRegression()
scikit_log_reg.fit(x_train,y_train)
y_pred=scikit_log_reg.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm
accuracy=31/42
accuracy