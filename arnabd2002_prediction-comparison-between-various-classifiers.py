import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

import seaborn as sns
df=pd.read_csv('../input/mushrooms.csv',sep=',')
df.columns
del classifierScores
classifierScores={}
X_data=df[df.columns[1:]]

y_data=df['class']
np.shape(X_data)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
X_EncodedData=X_data.apply(le.fit_transform)
np.shape(X_EncodedData),np.shape(y_data)
X_train,X_test,y_train,y_test=train_test_split(X_EncodedData,y_data,train_size=0.70,random_state=42)
from sklearn.neural_network import MLPClassifier
mlpClf=MLPClassifier(random_state=43,verbose=False)
mlpClf.fit(X_train,y_train)
mlpClf.score(X_test,y_test)*100
classifierScores['NN']=mlpClf.score(X_test,y_test)*100
from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()
mnb.fit(X_train,y_train)
mnb.score(X_test,y_test)*100
classifierScores['MNB']=mnb.score(X_test,y_test)*100
from sklearn.linear_model import LogisticRegression
logR=LogisticRegression(random_state=43,solver='lbfgs')
logR.fit(X_train,y_train)
logR.score(X_test,y_test)*100
classifierScores['LR']=logR.score(X_test,y_test)*100
from sklearn.ensemble import RandomForestClassifier
rfClf=RandomForestClassifier(random_state=43)
rfClf.fit(X_train,y_train)
rfClf.score(X_test,y_test)*100
classifierScores['RFC']=rfClf.score(X_test,y_test)*100
from matplotlib import pyplot as plt
plt.close()
plt.bar(classifierScores.keys(),classifierScores.values())

plt.show()
from sklearn.decomposition import PCA
pca=PCA(n_components=22,random_state=43)
pca.fit_transform(X_EncodedData)
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
plt.plot(var1)

plt.show()
finalPCA=PCA(n_components=15)
components=finalPCA.fit_transform(X_EncodedData)
pcaColumns=['PCA-'+str(a) for a in range(np.shape(components)[1])]
pcaDF=pd.DataFrame(components,columns=pcaColumns)
pcaDF[:1]
plt.scatter(pcaDF[:1

                 ],range(15),c=['r','g','b'])

plt.show()
X_pca_train,X_pca_test,y_pca_train,y_pca_test=train_test_split(pcaDF,y_data,train_size=0.70,random_state=43)
pcaLogR=LogisticRegression(random_state=43)
pcaLogR.fit(X_pca_train,y_pca_train)
pcaLogR.score(X_pca_test,y_pca_test)*100
pcaMNB=MultinomialNB()
pcaMNB.fit(abs(X_pca_train),y_pca_train)
pcaMNB.score(X_pca_test,y_pca_test)*100
pcaMLP=MLPClassifier()
pcaMLP.fit(X_pca_train,y_pca_train)
pcaMLP.score(X_pca_test,y_pca_test)*100