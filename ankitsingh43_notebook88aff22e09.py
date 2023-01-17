# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



df=pd.read_csv("../input/abalone.data.csv",header=None)

s=pd.get_dummies(df[0])

df=df.drop(0,axis=1)

df=pd.concat([s,df],axis=1)

y=df[8]



x=df.iloc[:,:-1]

from sklearn.model_selection import train_test_split as tts

xt,xe,yt,ye=tts(x,y)

from sklearn.svm import SVC as svc

clf=svc()

clf.fit(xt,yt)

prd=clf.predict(xe)

from sklearn.metrics import confusion_matrix  as cm, classification_report as cr,accuracy_score as ac 

res3=ac(ye,prd)
