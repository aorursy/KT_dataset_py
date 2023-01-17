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
#%% veri kümesini içeri aktarma/import dataset

#ilk olarak verimizi 'data' isimli değişkene atıyoruz.

#before of all we read the data that is we will use

data=pd.read_csv("/kaggle/input/breast-cancer-wisconsin-data/data.csv")



#sütunlar hakkında genel bilgi almak için 'info' komutunu kullanıyoruz.

#we use the 'info' command to get general information about columns/features.

print(data.info())

data.head()
#etkisiz sütunları silip bütün verileri sayısal formata dönüştürüyoruz.

#we delete ineffective columns and convert all data to numeric format. eg. int

data.drop(["Unnamed: 32","id"],axis=1,inplace=True) # 'Unnamed: 32' ve 'id' isimli sütunları siliyoruz.

data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis] #M=1 B=0
#sütun seçme ve normalizasyon

#column selection and normalization

y=data.diagnosis.values #adı geçen sütunu y değişkenine atadık. 0 ve 1'lerden oluşan hedef matrisimizi oluşturduk.

x_data=data.drop(["diagnosis"],axis=1) #adı geçen sütun haricini x_data değişkenine atadık.

x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values #normalizasyon yaptık. 0-1 arasına oranladık
#%%train test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42) 

#x ve y'yi rastgele(random) olarak eğitim ve test veriyeri ayırdık. 

#We randomly divided x and y into training and test data.

#'test_size=0.2' kodunun manası %80 eğitim %20 test verisi olarak ayırmaktır.

#The code 'test_size = 0.2' means separating the data as 80% training and 20% testing.

print("x_train: ", x_train.shape)

print("x_test: ", x_test.shape)

print("y_train: ", y_train.shape)

print("y_test: ", y_test.shape)
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression() 

lr.fit(x_train,y_train) #lr isimli YSA eğitildi.

print("test accuracy: {}".format(lr.score(x_test,y_test)))

print("test doğruluğu: {}".format(lr.score(x_test,y_test))) #lr isimli YSA test edildi.