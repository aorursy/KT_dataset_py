# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/heart.csv")
data.head()
data.info()
data.isnull().sum()
y=data.target   #y değerini belirledim. Yani predict edeceğimiz 1 veya 0 değeri
x_data =data.drop(["target"],axis=1)  #Y değerim dışındaki tüm değerleri train için al dedik.
# Yüksek değerlerin büyük değerlere baskın gelmemesi için Normalizasyon yaptık.
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
# Test ve Train olarak datamızı 2 ye böldük
# Train datası ile eğiteceğiz, test datası ile tahminler yapacağız.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
# Logistic Regression Modelimizi Oluşturduk
from sklearn.linear_model import LogisticRegression 
lr=LogisticRegression()
# Modelimizi train değerleri ile eğittik.
lr.fit(x_train,y_train)
# Ve test datasından gelen sonuçlar ile gerçek sonuçları karşılaştırıp bir accuracy(doğruluk) skoru bulduk.
print("Accuracy Score:",(lr.score(x_test,y_test))) 