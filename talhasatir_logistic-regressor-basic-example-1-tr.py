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
data =pd.read_csv('../input/voicegender/voice.csv')
data.head(500)
data.label.value_counts()
data.label=[1 if each == 'male' else 0 for each in data.label]

#data.diagnosis =[1 if each =='M' else 0 for each in data.diagnosis]
data.head()
data.info()
x=data.drop(['label'],axis=1)#x =label haric diger kolonlar
y =data.iloc[:,-1:]#y= sadece son kolon =label kolonum
x =(x -np.min(x))/(np.max(x)-np.min(x)).values 

#verimi normalize ediyorum böylece kolonlar arası baglantı daha yakın oldugu için ögrenmem daha iyi oluyor ve sonucuda 

#test accuracy'im daha iyi sonuc çıkartıyor.
from sklearn.model_selection import train_test_split #verimi test,egitim diye bölüyorum

x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=42)
#Logistic Regressor

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(x_train,y_train)

print('test accuracy: {}'.format(lr.score(x_test,y_test)))

#test accuracy = y_head larımdan kaç tanesi y ile aynı =bunun oranının verir bana