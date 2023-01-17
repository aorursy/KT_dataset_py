import numpy as np 

import pandas as pd

from sklearn import metrics

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier

from sklearn.cross_validation import train_test_split

from sklearn import linear_model, datasets

from sklearn.linear_model import RandomizedLogisticRegression

from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from sklearn.decomposition import PCA

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

import seaborn
df=pd.read_csv("../input/globalterrorismdb_0616dist.csv", encoding='ISO-8859-1',low_memory=False)

df.head()
df1=df[['iyear','imonth','iday','country','region','latitude','longitude','specificity'

        ,'vicinity','crit1','crit2','crit3','doubtterr','multiple','success','suicide'

        ,'attacktype1','targtype1','targsubtype1','ingroup','guncertain1','weaptype1']]
dfs=df1[df1['success']==1].dropna()

dff=df1[df1['success']==0].dropna()

dfs=dfs.sample(len(dff))





yf=dff['success']

xf=dff.drop(['success'],axis=1)





ys=dfs['success']

xs=dfs.drop(['success'],axis=1)
train1,test1=train_test_split(dfs,test_size=0.3)

train2,test2=train_test_split(dff,test_size=0.3)



train=train1.append(train2)

test=test1.append(test2)
Y=train['success'].values

X = train.drop(['success'],axis=1).values



y=test['success'].values

x = test.drop(['success'],axis=1).values



#X=MinMaxScaler().fit_transform(X)

#x=MinMaxScaler().fit_transform(x)



#pca=PCA(n_components=5)

#Xd=pca.fit_transform(X)

#xd=pca.fit_transform(x)
#transformer=SelectKBest(score_func=chi2,k=5)

#Xt=transformer.fit_transform(abs(X),Y)

#xt=transformer.fit_transform(abs(x),y)
reg_model = RandomForestClassifier()

reg_model = reg_model.fit(X,Y)



pred = reg_model.predict(x)



print(len(pred[pred==y])/float(len(y)))

X = train.drop(['success'],axis=1)



w11=pd.Series(np.sort(reg_model.feature_importances_),X.columns[np.argsort(reg_model.feature_importances_)])

w11.sort_values(inplace=True,ascending=False)

print (w11)
print(metrics.classification_report(y, pred))

print(metrics.confusion_matrix(y, pred))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y, pred)

roc_auc = auc(false_positive_rate, true_positive_rate)

print('AUC = %0.4f'% roc_auc)

plt.title('Receiver Operating Characteristic')

plt.plot(false_positive_rate, true_positive_rate, 'b',label='AUC = %0.2f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
chi=df[df['country_txt']=='China']

bj=chi[chi['provstate']=='Beijing']

bj1=bj[['iyear','imonth','iday','country','region','latitude','longitude','specificity'

        ,'vicinity','crit1','crit2','crit3','doubtterr','multiple','success','suicide'

        ,'attacktype1','targtype1','targsubtype1','ingroup','guncertain1','weaptype1']]



bj1['iyear']=2017

bj1=bj1.dropna()

bjx = bj1.drop(['success'],axis=1).values



bjy=reg_model.predict_proba(bjx)

pd.DataFrame(bjy,columns=['fail_prob','success_prob'])
bj1[-2:]
#bj[bj['targtype1']==14]['targtype1_txt']

#bj[bj['weaptype1']==8]['weaptype1_txt']

#bj[bj['targsubtype1']==74]['targsubtype1_txt']

#bj[bj['attacktype1']==2]['attacktype1_txt']