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
import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns

from dateutil import parser

import time
df=pd.read_csv('../input/appdata10/appdata10.csv')

df.head()
df.describe().T
df['hour']=df.hour.str.slice(1,3).astype(int)
df.head()
df1=df.copy().drop(columns=['user','screen_list','enrolled_date','first_open','enrolled'])
df1.head()
plt.figure(figsize=(20,10))

plt.suptitle('Histogram of Numerical Columns',fontsize=20)

for i in range(1,df1.shape[1]+1):

    plt.subplot(3,3,i)

    f=plt.gca()

    f.set_title(df1.columns.values[i-1])

    vals=np.size(df1.iloc[:,i-1].unique())

    plt.hist(df1.iloc[:i-1],bins=vals)

    
df1.shape[1]
df1.corrwith(df.enrolled).plot.bar(figsize=(20,10),title='Correlation with Response Variable',fontsize=15,rot=45,grid=True)

plt.ioff()
#Set up plot style

sns.set(style='white',font_scale=2)



#Compute Correlation Matrix

corr=df1.corr()



#Generate mask for upper traingle

mask=np.zeros_like(corr,dtype=np.bool)

mask[np.triu_indices_from(mask)]=True



#Set up the matplotlib figure

f,ax=plt.subplots(figsize=(18,15))

f.suptitle("Correlation Matrix",fontsize=40)



#Generate a custom diverging Colormap 

cmap=sns.diverging_palette(220,10,as_cmap=True)



#Draw the heat map with the mask and correct aspect ratio

sns.heatmap(corr,mask=mask,cmap=cmap,vmax=.3,center=0,square=True,linewidths=0.5,cbar_kws={'shrink':.5})

plt.ioff()
df.dtypes
df.head()
df['first_open']=[parser.parse(row_data) for row_data in df['first_open']]

df['enrolled_date']=[parser.parse(row_data) if isinstance(row_data,str) else row_data for row_data in df['enrolled_date']]
df.info()
df['Difference']=(df.enrolled_date-df.first_open).astype('timedelta64[h]')
df.head()
plt.hist(df['Difference'].dropna(),color='r')

plt.title('Distribution of Time-Since-Enrolled')

plt.ioff()
plt.hist(df['Difference'].dropna(),color='r',range=[0,100])

plt.title('Distribution of Time-Since-Enrolled')

plt.ioff()
df.shape
df.loc[df.Difference>48,'enrolled']=0
df=df.drop(columns=['Difference','enrolled_date','first_open'])
top_screens=pd.read_csv('../input/appdata10/top_screens.csv')

top_screens.head()
top_screens=top_screens.top_screens.values
for sc in top_screens:

    df[sc]=df.screen_list.str.contains(sc).astype(int)

    df['screen_list']=df.screen_list.str.replace(sc+",","")
df['Other']=df.screen_list.str.count(",")

df=df.drop(columns=['screen_list'])
df.head()
savings_screens=['Saving1','Saving2','Saving2Amount','Saving4','Saving5','Saving6','Saving7','Saving8','Saving9','Saving10']

df['SavingsCount']=df[savings_screens].sum(axis=1)
df=df.drop(columns=savings_screens)
cm_screens=['Credit1','Credit2','Credit3','Credit3Container','Credit3Dashboard']
df['CMCOunt']=df[cm_screens].sum(axis=1)

df=df.drop(columns=cm_screens)
loan_screens=['Loan','Loan2','Loan3','Loan4']
df['LoansCount']=df[loan_screens].sum(axis=1)

df=df.drop(columns=loan_screens)
df.columns
df.to_csv('new_appdata10.csv',index=False)
response=df['enrolled']
df=df.drop(columns='enrolled')
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(df,response,test_size=0.2,random_state=0)
train_identifier=X_train['user']

X_train=X_train.drop(columns='user')

test_identifier=X_test['user']

X_test=X_test.drop(columns='user')
from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()

X_train2=pd.DataFrame(sc_X.fit_transform(X_train))

X_test2=pd.DataFrame(sc_X.transform(X_test))

X_train2.columns=X_test.columns.values

X_train2.index=X_train.index.values

X_test2.index=X_test.index.values

X_train=X_train2

X_test=X_test2
from sklearn.linear_model import LogisticRegression 

classifier=LogisticRegression(random_state=0,penalty='l1')

classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score

cm=confusion_matrix(y_test,y_pred)

cm
accuracy_score=(y_test,y_pred)
precision_score=(y_test,y_pred)
recall_score(y_test,y_pred)
f1_score(y_test,y_pred)
df_cm=pd.DataFrame(cm,index=(0,1),columns=(0,1))

plt.figure(figsize=(10,7))

sns.set(font_scale=1.4)

sns.heatmap(df_cm,annot=True,fmt='g')

#print('Test Data Accuracy:%0.4f' % accuracy_score(y_test,y_pred))

plt.ioff()
from sklearn.model_selection import cross_val_score

accuracies=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)

print('Logistic Accuracy:%0.3f (+/- %0.3f)' % (accuracies.mean(),accuracies.std()*2))
final_results=pd.concat([y_test,test_identifier],axis=1).dropna()

final_results['predicted_results']=y_pred

final_results[['user','enrolled','predicted_results']].reset_index(drop=True)