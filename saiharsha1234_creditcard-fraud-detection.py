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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
creditcard=pd.read_csv('../input/creditcardfraud/creditcard.csv')
creditcard.shape
creditcard.dtypes
creditcard['Class'].value_counts()
creditcard.columns
a=creditcard[['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']]
a.shape
from sklearn.decomposition import PCA
pca=PCA(n_components=19)
pca.fit(a)
pca.explained_variance_ratio_
pca.explained_variance_ratio_.sum()
creditcard_pca_19=pca.transform(a)
creditcard_pca_19_df=pd.DataFrame(data=creditcard_pca_19)
creditcard_pca_19_df.shape
creditcard_modified_19_column_table=pd.concat([creditcard_pca_19_df,creditcard['Amount']],axis=1)
creditcard_modified_19_column_table.shape
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(creditcard_modified_19_column_table)
x11=pd.DataFrame(data=x)
x11.head()
X_train=x11
Y_train=creditcard['Class']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X_train,Y_train,test_size=0.3,random_state=1,shuffle=True)
x_train.shape,y_train.shape,x_test.shape,y_test.shape
mu=x_train.mean()
sigma=x_train.std()


def multivariateGaussian(x_train, mu, sigma):
    k = len(mu)
    sigma=np.diag(sigma)
    x_train = x_train - mu.T
    p = 1/((2*np.pi)**(k/2)*(np.linalg.det(sigma)**0.5))* np.exp(-0.5* np.sum(x_train @ np.linalg.pinv(sigma) * x_train,axis=1))
    return p
p = multivariateGaussian(x_train, mu, sigma)
p_mean=p.mean()
p_std=p.std()
a11=(p-p_mean)/(p_std)
a11.plot(kind='hist')
a11.max()
a11.min()
dependent=creditcard['Class']
dependent_df=pd.DataFrame(data=dependent)
dependent_df['index']=dependent_df.index
dependent_df.reset_index(drop=True,inplace=True)
dependent_df
a11_df_3=pd.DataFrame(data=a11)
a11_df_3_5=pd.DataFrame(data=a11)
a11_df_4=pd.DataFrame(data=a11)
a11_df_3['index']=a11_df_3.index
a11_df_3.reset_index(drop=True,inplace=True)
a11_df_3_5['index']=a11_df_3_5.index
a11_df_3_5.reset_index(drop=True,inplace=True)
a11_df_4['index']=a11_df_4.index
a11_df_4.reset_index(drop=True,inplace=True)
a11_df_3['Pre'] = [1 if x >= 3 else 0 for x in a11_df_3[0]]  ##threshold 3
a11_df_3_5['Pre'] = [1 if x >= 3.5 else 0 for x in a11_df_3_5[0]]  ##threshold 3.5
a11_df_4['Pre'] = [1 if x >= 4 else 0 for x in a11_df_4[0]]  ##threshold 4
complete_3=pd.merge(a11_df_3,dependent_df,how='inner',on='index')
complete_3_5=pd.merge(a11_df_3_5,dependent_df,how='inner',on='index')
complete_4=pd.merge(a11_df_4,dependent_df,how='inner',on='index')
complete_3=complete_3.rename(columns={0:'datapoints','Class':'Actual'})
complete_3_5=complete_3_5.rename(columns={0:'datapoints','Class':'Actual'})
complete_4=complete_4.rename(columns={0:'datapoints','Class':'Actual'})
complete_3['Pre'].value_counts()
complete_3_5['Pre'].value_counts()
complete_4['Pre'].value_counts()
complete_3_5['Actual'].value_counts()
# Modeling:

#Import svm model
from sklearn.svm import SVC

#Create a svm Classifier
clf = SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(x_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(x_test)
# Model evaluation:

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix,classification_report
cm_3=confusion_matrix(complete_3['Actual'],complete_3['Pre'])
cm_3
cm_3_5=confusion_matrix(complete_3_5['Actual'],complete_3_5['Pre'])
cm_3_5
cm_4=confusion_matrix(complete_4['Actual'],complete_4['Pre'])
cm_4
CR_3=classification_report(complete_3['Actual'],complete_3['Pre'])
print(CR_3)
CR_3_5=classification_report(complete_3_5['Actual'],complete_3_5['Pre'])
print(CR_3_5)
CR_4=classification_report(complete_4['Actual'],complete_4['Pre'])
print(CR_4)
threshold_3_FNR=5222/(5222+193785)
threshold_3_FNR
threshold_3_5_FNR=3185/(3185+195822)
threshold_3_5_FNR
threshold_4_FNR=1795/(1795+197212)
threshold_4_FNR
a11_df=pd.DataFrame(data=a11)
y_train_df=pd.DataFrame(data=y_train)
a11_df['index']=a11_df.index
y_train_df['index']=y_train_df.index
a11_df.reset_index(drop=True,inplace=True)
y_train_df.reset_index(drop=True,inplace=True)
a11_y_train_df=pd.merge(a11_df,y_train_df,how='inner',on='index')
a11_y_train_df=a11_y_train_df.rename(columns={0:'DataPoints'})
a11_y_train_df[a11_y_train_df['Class']==1]['DataPoints'].plot(kind='hist')
a11_y_train_df[a11_y_train_df['Class']==1]['DataPoints'].min(),a11_y_train_df[a11_y_train_df['Class']==1]['DataPoints'].max()
