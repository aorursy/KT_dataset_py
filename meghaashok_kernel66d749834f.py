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
        
import warnings
warnings.filterwarnings('ignore')
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
train_data=pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020/train_data.csv')
train_data.head()
X=train_data.drop('price_range',axis=1).values
Y=train_data['price_range']
from sklearn.model_selection import train_test_split
df_test=pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020/test_data.csv')
id_test=df_test['id']
X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=0.5,random_state=101)
from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier(n_neighbors=10)
import numpy as np
error_rate=[]
for i in range(1,20):
    kn=KNeighborsClassifier(n_neighbors=i)
    kn.fit(X_train,Y_train)
    pred_i=kn.predict(X_test)
    error_rate.append(np.mean(pred_i!=Y_test))
from sklearn.linear_model import LogisticRegression
logmodel=LogisticRegression()
logmodel.fit(X_train,Y_train)
Y_pred=logmodel.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
pred=kn.predict(X_test)
data_test=pd.read_csv('/kaggle/input/mobile-price-range-prediction-is2020/test_data.csv')
predicted_price=kn.predict(data_test)
data_test['price_range']=predicted_price
data_test=pd.DataFrame({'id':id_test,'price_range':predicted_price})
data_test.to_csv('output.csv',index=False)
data={'id': sample_submission['id'],
      'price_range':y_pred}
res=pd.DataFrame(data)
res.to_csv('/kaggle/working/result_assign.csv',index=False)
output=pd.read_csv('/kaggle/working/result_assign.csv')