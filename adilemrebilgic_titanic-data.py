# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Loading data
trainData=pd.read_csv('../input/train.csv', index_col=0)
testData=pd.read_csv('../input/test.csv', index_col=0)
#Check the training data
trainData.head()
#About data size
print('Number of obs: ',trainData.shape[0])
print('Number of features: ',trainData.shape[1])
#about data
trainData.info()
# some features have missing value issue. Age, Cabin, Embarked, SibSp, Parch
#numeric features: Age, SibSp, Fare, Parch
#categoric features: pClass, name, Sex, Ticket,Cabin , Embarked
#target: Survived
#visualize all
pd.plotting.scatter_matrix(trainData,figsize=(15,15));
#Numeric Data Summary
trainData[['Age','Fare','SibSp', 'Parch']].describe()
#Visualize Numeric Vals
import matplotlib.pyplot as plt
##Age
fig=plt.figure()

###
ax1=fig.add_subplot(4,3,1)
ax1=trainData['Age'].plot(kind='hist',
         title='Age',
        
         grid=True,
        figsize=(25,20))

###
ax2=fig.add_subplot(4,3,2)
ax2=trainData['Age'].plot(kind='line',
         title='Age',
         style='o',
         grid=False,
         alpha=0.5,figsize=(25,20)
         )
ax2=plt.plot([trainData['Age'].mean()]*trainData.shape[0],color='red',linewidth=2,alpha=0.5)
ax2=plt.plot([trainData['Age'].quantile(q=0.01)]*trainData.shape[0],
          linestyle='--',
          color='red',
          alpha=0.5)
ax2=plt.plot([trainData['Age'].quantile(q=0.99)]*trainData.shape[0],
          linestyle='--',
          color='red',
          alpha=0.5)

###
ax3=fig.add_subplot(4,3,3)
ax3=trainData['Age'].plot(kind='box',figsize=(26,18))

##Fare
###
ax4=fig.add_subplot(4,3,4)
ax4=trainData['Fare'].plot(kind='hist',
         title='Fare',
         grid=True,
        figsize=(25,20))
###
ax5=fig.add_subplot(4,3,5)
ax5=trainData['Fare'].plot(kind='line',
         title='Fare',
         style='o',
         grid=False,
         alpha=0.5,figsize=(25,20)
         )
ax5=plt.plot([trainData['Fare'].mean()]*trainData.shape[0],color='red',linewidth=2,alpha=0.5)
ax5=plt.plot([trainData['Fare'].quantile(q=0.01)]*trainData.shape[0],
          linestyle='--',
          color='red',
          alpha=0.5)
ax5=plt.plot([trainData['Fare'].quantile(q=0.99)]*trainData.shape[0],
          linestyle='--',
          color='red',
          alpha=0.5)

###
ax6=fig.add_subplot(4,3,6)
ax6=trainData['Fare'].plot(kind='box')

##Sibsp
###

ax7=fig.add_subplot(4,3,7)
ax7=trainData['SibSp'].plot(kind='hist',
         title='SibSp',
         grid=True,
        figsize=(16,8))
###
ax8=fig.add_subplot(4,3,8)
ax8=trainData['SibSp'].plot(kind='line',
         title='SibSp',
         style='o',
         grid=False,
         alpha=0.5,figsize=(25,20)
         )
ax8=plt.plot([trainData['SibSp'].mean()]*trainData.shape[0],color='red',linewidth=2,alpha=0.5)
ax8=plt.plot([trainData['SibSp'].quantile(q=0.01)]*trainData.shape[0],
          linestyle='--',
          color='red',
          alpha=0.5)


ax8=plt.plot([trainData['SibSp'].quantile(q=0.99)]*trainData.shape[0],
          linestyle='--',
          color='red',
          alpha=0.5)

###
ax9=fig.add_subplot(4,3,9)
ax9=trainData['SibSp'].plot(kind='box')


##Parch
###

ax10=fig.add_subplot(4,3,10)
ax10=trainData['Parch'].plot(kind='hist',
         title='Parch',
         grid=True,
        figsize=(25,20))
###
ax11=fig.add_subplot(4,3,11)
ax11=trainData['Parch'].plot(kind='line',
         title='Parch',
         style='o',
         grid=False,
         alpha=0.5,figsize=(25,20)
         )
ax11=plt.plot([trainData['Parch'].mean()]*trainData.shape[0],color='red',linewidth=2,alpha=0.5)
ax11=plt.plot([trainData['Parch'].quantile(q=0.01)]*trainData.shape[0],
          linestyle='--',
          color='red',
          alpha=0.5)


ax11=plt.plot([trainData['Parch'].quantile(q=0.99)]*trainData.shape[0],
          linestyle='--',
          color='red',
          alpha=0.5)

###
ax12=fig.add_subplot(4,3,12)
ax12=trainData['Parch'].plot(kind='box')

plt.tight_layout()


#Visualize Non-Numeric Vals
feature_list=['Survived','Pclass','Sex','Embarked']
fig_row_size=len(feature_list)
fig_col_size=1
fig=plt.figure()
## Survived
for i in range(len(feature_list)):
    feature=feature_list[i]
    fig_name=('ax'+str(i))
    fig_name=fig.add_subplot(fig_row_size,fig_col_size,int(i+1))
    trainData[feature].value_counts().plot.pie(startangle=90,
                                         autopct='%1.0f%%',
                                         figsize=(4,14),
                                         colormap='Pastel1',
                                         ax=fig_name
                                              )
    fig_name.set_ylabel(feature)
    
plt.tight_layout()
#Features and survival rates
feature_list=['Pclass','Age','Sex','SibSp','Parch','Embarked']
fig_row_size=len(feature_list)
fig_col_size=1
fig=plt.figure()
for i in range(len(feature_list)):
    feature=feature_list[i]
    fig_name=fig.add_subplot(fig_row_size,fig_col_size,int(i+1))
    df_for_plotting=trainData.groupby(feature).agg(['sum','count'])['Survived']
    df_for_plotting['SurvivalRate']=df_for_plotting['sum']/df_for_plotting['count']
    plot_title=('Survival Rate - '+feature)
    df_for_plotting.SurvivalRate.plot(kind='bar',title=plot_title,ax=fig_name,figsize=(16,28))
    fig_name.set_ylabel('Survival Rate')

plt.tight_layout()
