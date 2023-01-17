

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np


########### first one ###########



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np



df=pd.read_csv("train.csv")

df_test=pd.read_csv("test.csv")



X=df.drop(["target"],axis=1)



X['1']



Y=df['target'].values



from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)









from sklearn.linear_model import LogisticRegression

clf=LogisticRegression()

clf.fit(X_train,Y_train)



pred=clf.predict(X_test)



from sklearn.metrics import accuracy_score,confusion_matrix



ac=accuracy_score(Y_test,pred).round(2)



cm=confusion_matrix(pred,Y_test)





plt.hist(pred)

plt.hist(Y)



plt.plot(pred,Y_test)



sns.stripplot(x=pred,y=Y_test)

sns.stripplot(x=X['1'],y=X['2'])



sns.boxplot(x=X['1'],y=X['2'])



sns.distplot(pred,hist=True)



sns.distplot(X['1'],hist=True)


