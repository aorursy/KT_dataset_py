import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib as mpl

%matplotlib inline
df_iris=pd.read_csv('../input/iris/Iris.csv')

df_iris.head()
df_iris.drop('Id',axis=1,inplace=True)

df_split=df_iris['Species'].str.rsplit('-',expand=True)

df_iris.drop('Species',axis=1,inplace=True)
df_iris['Species']=df_split.iloc[:,-1]

df_iris.head()
fig1=plt.figure(figsize=(10,5))

ax1=fig1.add_subplot(121)



g=sns.distplot(df_iris['SepalLengthCm'],ax=ax1)

ax1.set_xlabel('Sepal Length (cm)',size=15)

ax1.set_title('Sepal length distribution  \n \n Median length (cm): {0:.2f}'.format(df_iris['SepalLengthCm'].median()),size=18)

g.axvline(df_iris['SepalLengthCm'].median(),color='indianred',label='Median length')

ax1.legend()





ax2=fig1.add_subplot(122)

h=sns.distplot(df_iris['SepalWidthCm'],ax=ax2,color='green')

ax2.set_xlabel('Sepal Width (cm)',size=15)

ax2.set_title('Sepal length distribution  \n \n Median width (cm): {0:.2f}'.format(df_iris['SepalWidthCm'].median()),size=18)

h.axvline(df_iris['SepalWidthCm'].median(),color='black',label='Median Width')

ax2.legend()
fig2=plt.figure(figsize=(10,5))

ax3=fig2.add_subplot(121)



g=sns.distplot(df_iris['PetalLengthCm'],ax=ax3)

ax3.set_xlabel('Petal Length (cm)',size=15)

ax3.set_title('Petal length distribution  \n \n Median length (cm): {0:.2f}'.format(df_iris['PetalLengthCm'].median()),size=18)

g.axvline(df_iris['PetalLengthCm'].median(),color='indianred',label='Median length')

ax3.legend()





ax4=fig2.add_subplot(122)

h=sns.distplot(df_iris['PetalWidthCm'],ax=ax4,color='green')

ax4.set_xlabel('Petal Width (cm)',size=15)

ax4.set_title('Petal length distribution  \n \n Petal width (cm): {0:.2f}'.format(df_iris['PetalWidthCm'].median()),size=18)

h.axvline(df_iris['PetalWidthCm'].median(),color='black',label='Median Width')

ax4.legend()
df_vir=df_iris[df_iris['Species']=='virginica']

df_set=df_iris[df_iris['Species']=='setosa']

df_ver=df_iris[df_iris['Species']=='versicolor']





fig3=plt.figure(figsize=(13,5))



ax5=fig3.add_subplot(121)



a=sns.violinplot(x='Species',y='SepalLengthCm',data=df_iris,ax=ax5,orient='v',inner='quartile')



ax5.set_title('Sepal Length distribution of each species',size=13)





ax6=fig3.add_subplot(122)

b=sns.violinplot(x='Species',y='SepalWidthCm',data=df_iris,ax=ax6,orient='v',inner='quartile',palette='summer')

ax6.set_title('Sepal Width distribution of each species',size=13)

fig4=plt.figure(figsize=(15,6))

ax6=fig4.add_subplot(121)



a=sns.violinplot(x='Species',y='PetalLengthCm',data=df_iris,ax=ax6,orient='v',inner='quartile')



ax6.set_title('Petal Length distribution of each species',size=13)





ax7=fig4.add_subplot(122)

b=sns.violinplot(x='Species',y='PetalWidthCm',data=df_iris,ax=ax7,orient='v',inner='quartile',palette='summer')

ax7.set_title('Petal Width distribution of each species',size=13)



sns.set()

fig5=plt.figure(figsize=(20,10))

ax8=fig5.add_subplot(121)

ax8.set_title('Labelled clusters based on Sepal width and length',size=20)



ax8.scatter(df_set.iloc[:,0],df_set.iloc[:,1],c='red',s=100,alpha=0.6,label='Setosa')

ax8.scatter(df_vir.iloc[:,0],df_vir.iloc[:,1],c='blue',s=100,alpha=0.6,label='Virginica')

ax8.scatter(df_ver.iloc[:,0],df_ver.iloc[:,1],c='green',s=100,alpha=0.6,label='Versicolor')



ax8.set_xlabel('Sepal Length (cm)',size=15)

ax8.set_ylabel('Sepal Width (cm)',size=15)



ax8.legend(fontsize=15)





ax8=fig5.add_subplot(122)

ax8.set_title('Labelled clusters based on Petal width and length',size=20)



ax8.scatter(df_set.iloc[:,2],df_set.iloc[:,3],c='red',s=100,alpha=0.6,label='Setosa')

ax8.scatter(df_vir.iloc[:,2],df_vir.iloc[:,3],c='blue',s=100,alpha=0.6,label='Virginica')

ax8.scatter(df_ver.iloc[:,2],df_ver.iloc[:,3],c='green',s=100,alpha=0.6,label='Versicolor')



ax8.set_xlabel('Petal Length (cm)',size=15)

ax8.set_ylabel('Petal Width (cm)',size=15)



ax8.legend(fontsize=15)
df_iris.plot.area(y=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],

                  alpha=0.6,figsize=(10,8),stacked=False)

plt.xlabel('Number of flowers',size=15)

plt.ylabel('Dimensions (cm)',size=15)
df_iris.plot.area(y=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],

                  alpha=0.6,figsize=(10,8),stacked=True)

plt.xlabel('Number of flowers',size=15)

plt.ylabel('Cumulative dimensions (cm)',size=15)




sns.jointplot(df_iris['SepalLengthCm'],df_iris['SepalWidthCm'],kind='kde',color='green')

sns.jointplot(df_iris['PetalLengthCm'],df_iris['PetalWidthCm'],kind='kde',color='red')

df_iris['Species']=df_iris['Species'].replace('setosa',1)

df_iris['Species']=df_iris['Species'].replace('virginica',2)

df_iris['Species']=df_iris['Species'].replace('versicolor',3)
targets=df_iris['Species']

df_iris.drop('Species',axis=1,inplace=True)
X=df_iris

y=targets
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,shuffle=True)
dtc=DecisionTreeClassifier(max_depth=6)

dtc.fit(X_train,y_train)
y_preds=dtc.predict(X_test)

dtc.score(X_test,y_test)
conf_mat=confusion_matrix(y_preds,y_test)

fig6=plt.figure(figsize=(10,8))

ax9=fig6.add_subplot(111)



sns.heatmap(conf_mat,annot=True,cmap='viridis',ax=ax9)

ax9.xaxis.set_ticklabels(['Setosa','Virginica','Versicolor'])

ax9.yaxis.set_ticklabels(['Setosa','Virginica','Versicolor'])

plt.xlabel('Predictions',size=15)

plt.ylabel('Actual',size=15)
dtc.fit(X_train,y_train)

cols=df_iris.columns

col_arr=np.array(cols).reshape(-1,1)

df_imp=pd.DataFrame(col_arr)

df_imp.rename(columns={0:'Feature'},inplace=True)

imp=(100*dtc.feature_importances_)

df_imp['Importance']=imp

df_imp
sns.catplot('Importance','Feature',data=df_imp,kind='bar',height=8,aspect=2)

plt.title('Feature importance',size=20)


fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300 ,facecolor='b')

mpl.rcParams['text.color'] = 'black'

fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']

cn=['setosa', 'versicolor', 'virginica']





from sklearn import tree

tree.plot_tree(dtc,feature_names = fn, 

               class_names=cn,filled=True,rotate=True,ax=axes)