import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

import sklearn.metrics as mt

from sklearn.model_selection import train_test_split

from scipy.stats import zscore



df = pd.read_csv("../input/Admission_Predict.csv",sep = ",",index_col='Serial No.')

df.head()
columnMap = {key: key.split()[0] for key in df.head(0)}

df.rename(columns = columnMap, inplace = True)

df.head(1)
df.isna().any()
df.hist(layout=(2,4),figsize=(16,6))

df.corr()
X= df.drop("Chance",axis=1).apply(zscore)

X.Research = df.Research

Y = df.Chance

x_train, x_test,y_train, y_test = train_test_split(X,Y,test_size = 0.20,random_state = 0)



P=.9

yc1_train = np.where(y_train>=P, 1, 0)

yc1_test = np.where(y_test>=P, 1, 0)



P =.6

yc2_train = np.where(y_train>=P, 1, 0)

yc2_test = np.where(y_test>=P, 1, 0)
rf1 = RandomForestClassifier(n_estimators=100,random_state=0)

rf1.fit(x_train, yc1_train)



rf2 = RandomForestClassifier(n_estimators=100,random_state=0)

rf2.fit(x_train, yc2_train)



yc1_out = rf1.predict(x_test)

yc2_out = rf2.predict(x_test)



fig = plt.figure(figsize=(15,5))

ax1 = fig.add_subplot(121)

ax2 = fig.add_subplot(122)



importances1 = pd.DataFrame(rf1.feature_importances_,index = x_train.columns,columns=['importance']).sort_values('importance',ascending=False)

importances1.plot(kind='bar',title='Selective University',ax=ax1,ylim=(.05,.35))



importances2 = pd.DataFrame(rf2.feature_importances_,index = x_train.columns,columns=['importance']).sort_values('importance',ascending=False)

importances2.plot(kind='bar',title='Sane University',ax=ax2,ylim=(.05,.35))



print("Selective University's Report")

print(mt.classification_report(yc1_test, yc1_out,target_names=['Reject','Admit']))



print("Sane University's Report")

print(mt.classification_report(yc2_test, yc2_out,target_names=['Reject','Admit']))
