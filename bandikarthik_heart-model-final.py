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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler as MM

from sklearn.linear_model import LogisticRegression as LR

from sklearn.model_selection import train_test_split as tts

from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier as NC

df=pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")

df.rename(columns={"cp":"CHESTPAIN_TYPE"},inplace=True)

df.rename(columns={"chol":"CHOLESTORAL_MG/DL"},inplace=True)

df.rename(columns={"fbs":"FASTING_BLOOD_SUGAR"},inplace=True)

df.rename(columns={"restecg":"REST_ECG"},inplace=True)

df.rename(columns={"exang":"EXCERSICE_INDUCED_ANGINA"},inplace=True)

df.rename(columns={"oldpeak":"OLDPEAK"},inplace=True)

df.rename(columns={"slope":"SLOPE"},inplace=True)

df.rename(columns={"ca":"NO_VESSELS"},inplace=True)

df.rename(columns={"thalach":"MAX_HEARTRATE"},inplace=True)

df.rename(columns={"thal":"THAL"},inplace=True)

df.rename(columns={"age":"AGE"},inplace=True)

df.rename(columns={"sex":"GENDER"},inplace=True)

df.rename(columns={"trestbps":"REST_BP"},inplace=True)

df.rename(columns={"target":"TARGET"},inplace=True)

df.head()
df.isnull().sum()
df.describe()
x=df["TARGET"]

plt.figure(figsize=(15,10))

x.value_counts().plot(kind="bar")
df=df.drop("TARGET",axis=1)
df.head()
x.head()
normalization=MM()

df_norm=normalization.fit_transform(df)

df_norm
df.info()
M=NC(n_neighbors=5)

M.fit(x_train,y_train)

P1=M.predict(x_test)

score_accuracy1=accuracy_score(y_test,P1)*100

print("score_accuracy using neighborclassifier = ",score_accuracy1)
x_train,x_test,y_train,y_test=tts(df_norm,x)

lm=LR()

lm.fit(x_train,y_train)

P=lm.predict(x_test)

score_accuracy=accuracy_score(y_test,P)*100

print("score_accuracy using LogisticRegression = ",score_accuracy)
##lOGISTICREGRESSION GOT THE HEIGHEST SCORE