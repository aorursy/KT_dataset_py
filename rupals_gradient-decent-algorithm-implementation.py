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
#import statements

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

import random
df = pd.read_csv('/kaggle/input/gpu-runtime/sgemm_product.csv')

df.shape
#creating Runtime, target variable by taking average of Run1, Run2, Run3, Run4

df['Runtime']=df[['Run1 (ms)','Run2 (ms)','Run3 (ms)','Run4 (ms)']].mean(axis=1)
#viewing data

df.head()
#drop other Run time variables

df1=df.drop(columns =['Run1 (ms)','Run2 (ms)','Run3 (ms)','Run4 (ms)'], axis = 1)

df1.info()
#checking descriptive stats

df1.describe().T
#checking for NULL values

df1.isnull().sum() #no NULL values
#checking for outliers

plt.figure(figsize=(10,6))

sns.boxplot(df1['Runtime']);
#removing outliers

Q1=df1['Runtime'].quantile(0.25)

Q2=df1['Runtime'].quantile(0.75)

IQR = Q2 - Q1

LL=Q1-1.5*IQR

UL=Q2+1.5*IQR

df2 = df1[(df1.Runtime>LL) & (df1.Runtime<UL)]

df2.describe().T
plt.figure(figsize=(10,6))

sns.boxplot(df2['Runtime']);
#checking variable distribution

for index in range(10):

   df2.iloc[:,index] = (df2.iloc[:,index]-df2.iloc[:,index].mean()) / df2.iloc[:,index].std();

df2.hist(figsize= (14,16));
#plotting the distribution of Runtime

sns.distplot(df2['Runtime'])
df2['target']=np.log(df2.Runtime)

sns.distplot(df2['target'])
plt.figure(figsize=(14,14))

sns.set(font_scale=1)

sns.heatmap(df2.corr(),cmap='GnBu_r',annot=True, square = True ,linewidths=.5);

plt.title('Variable Correlation')
#creating an intercept varible during martix dot product

df2.insert(0,'intercept',1)

df2
#define cost function

def linear_costfunc(dfile,targetvar,coefmat):

  loss=np.dot(dfile,coefmat.T)-targetvar

  cost=np.sum(np.power(loss,2)/(2*len(dfile)))

  return cost
#define gradient decent considering fixed iterations

def linear_gdesc(dfile,targetvar,coefmat,alpha,iterations,threshold):

  cost_ls=[linear_costfunc(dfile,targetvar,coefmat)]

  gddf=pd.DataFrame(coefmat)

  for i in range(1,iterations):

    loss=np.dot(dfile,coefmat.T)-targetvar

    dep=np.dot(loss.T,dfile)

    coefmat=coefmat-dep*alpha/len(dfile)   

    gddf=gddf.append(pd.DataFrame(coefmat),ignore_index=True)

    cost_ls+=[linear_costfunc(dfile,targetvar,coefmat)]

    if (abs(cost_ls[i]-cost_ls[i-1]))<=threshold:

      break

  gddf['cost']=cost_ls

  #print(gddf)

  print("Iterations needed to converge: ", i+1)

  min_cost=gddf[gddf.cost==min(gddf.cost)]

  print('Cost at convergance: ', cost_ls[i])

  min_cost=min_cost.drop(columns='cost',axis=1)

  #print(min_cost)

  return min_cost
#predicting target variable

def predict(cost_mat,xtest):

  predic_target=xtest.dot(cost_mat.T)

  return predic_target
#RMSE

def linear_rmse(ypredict,ytest):

  sum_sq=np.sum((ytest-ypredict)**2)

  mse=sum_sq/len(ytest)

  rmse=(mse)**(1/2)

  return rmse
#Linear Regression fucntion

def LinearReg(x1_train,x1_test,y1_train,y1_test,alpha,iterations,coefmat,threshold):

  if len(alpha)>1:

    coef_ls=[0]*len(alpha)

    ypredict=[0]*len(alpha)

    rmse=[0]*len(alpha)

    for i, a in enumerate(alpha, start=0):

      coef_ls[i]=linear_gdesc(x1_train,y1_train,coefmat,a,iterations,threshold)

      ypredict[i]=predict(coef_ls[i],x1_test)

      rmse[i]=linear_rmse(ypredict[i],y1_test)

      print("For learning rate=", a, " RMSE is: ", rmse[i])

      print("Coeffients: ",coef_ls[i])

    plt.plot(alpha,rmse)

    plt.xlabel('Learning Rate')

    plt.ylabel('RMSE')

    plt.show()

  elif len(threshold)>1:

    coef_ls=[0]*len(threshold)

    ypredict=[0]*len(threshold)

    rmse=[0]*len(threshold)

    for i, t in enumerate(threshold, start=0):

      coef_ls[i]=linear_gdesc(x1_train,y1_train,coefmat,alpha,iterations,t)

      ypredict[i]=predict(coef_ls[i],x1_test)

      rmse[i]=linear_rmse(ypredict[i],y1_test)

      print("For threshold=", t, " RMSE is: ", rmse[i])

      print("Coeffients: ",coef_ls[i])

    plt.plot(threshold,rmse)

    plt.xlabel('Threshold')

    plt.ylabel('RMSE')

    plt.show()

  else:

    coef_ls=[0]

    ypredict=[0]

    rmse=[0]

    for i in range(1):

      coef_ls[i]=linear_gdesc(x1_train,y1_train,coefmat,alpha,iterations,threshold)

      ypredict[i]=predict(coef_ls[i],x1_test)

      rmse[i]=linear_rmse(ypredict[i],y1_test)

      print("For threshold=", threshold," and learning rate: ",alpha, " RMSE is: ", rmse[i])

      print("Coeffients: ",coef_ls[i])

    return rmse[i]
#test and train data

iterations=1000

df_target=df2[['target']].values

df_features=df2.drop(columns=['target','Runtime'],axis=1).values

x1_train, x1_test, y1_train, y1_test = train_test_split(df_features, df_target, test_size = 0.2, random_state = 0)

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)
#Part 1 for minimum rmse with train and test data

threshold=[0.000001]

alpha=[0.09,0.095,0.1,0.2,0.3]

iterations=1000

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

LinearReg(x1_train,x1_test,y1_train,y1_test,alpha,iterations,coefmat,threshold)
threshold=[0.000001]

alpha=[0.001,0.01,0.1,0.2,0.5]

iterations=1000

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

LinearReg(x1_train,x1_test,y1_train,y1_test,alpha,iterations,coefmat,threshold)
#Part 2 for minimum rmse within training data

threshold=[0.000001]

alpha=[0.7,0.75,0.8,0.85]

iterations=1000

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

LinearReg(x1_train,x1_train,y1_train,y1_train,alpha,iterations,coefmat,threshold)
threshold=[0.000001]

alpha=[0.001,0.01,0.1,0.8,0.9]

iterations=1000

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

LinearReg(x1_train,x1_train,y1_train,y1_train,alpha,iterations,coefmat,threshold)
#Part 1 for minimum rmse with train and test data

threshold=[0.00000001,0.0000001,0.0000002,0.0000003,0.0000004,0.0000005,0.0000006,0.0000007,0.0000008,0.0000009,0.000001,0.000002,0.000003,0.000004,0.000005]

alpha=[0.2]

iterations=1000

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

LinearReg(x1_train,x1_test,y1_train,y1_test,alpha,iterations,coefmat,threshold)
#Error as a function of number of gradient descent iterations for test and train

threshold=[0.000001]

alpha=[0.2]

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

iterations=1000

itera=[]

rmse=[]

cost_ls=[linear_costfunc(x1_train,y1_train,coefmat)]

gddf=pd.DataFrame(coefmat)

for i in range(1,iterations):

  loss=np.dot(x1_train,coefmat.T)-y1_train

  dep=np.dot(loss.T,x1_train)

  coefmat=coefmat-dep*alpha/len(x1_train)   

  ypredict=predict(coefmat,x1_test)

  rmse+=[linear_rmse(ypredict,y1_test)]

  itera+=[i]

  cost_ls+=[linear_costfunc(x1_train,y1_train,coefmat)]

  if (abs(cost_ls[i]-cost_ls[i-1]))<=threshold:

    break

  print("For iteration: ",i," RMSE is: ", linear_rmse(ypredict,y1_test))

#print("Coeffients: ",min_cost)

plt.plot(itera,rmse)

plt.xlabel('Iteration')

plt.ylabel('RMSE')

plt.show()
#Part 2 for minimum rmse within training data

#threshold=[0.000000000000001,0.00000000000001,0.0000000000001]

threshold=[0.0000000000000001,0.000000000000001,0.000000000000002,0.000000000000003,0.000000000000004,0.000000000000005,0.000000000000006,0.000000000000007,0.000000000000008,0.000000000000009,0.00000000000001]

alpha=[0.8]

iterations=1000

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

LinearReg(x1_train,x1_train,y1_train,y1_train,alpha,iterations,coefmat,threshold)
#Error as a function of number of gradient descent iterations within train

threshold=[0.000001]

alpha=[0.8]

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

iterations=1000

itera=[]

rmse=[]

cost_ls=[linear_costfunc(x1_train,y1_train,coefmat)]

gddf=pd.DataFrame(coefmat)

for i in range(1,iterations):

  loss=np.dot(x1_train,coefmat.T)-y1_train

  dep=np.dot(loss.T,x1_train)

  coefmat=coefmat-dep*alpha/len(x1_train)   

  ypredict=predict(coefmat,x1_train)

  rmse+=[linear_rmse(ypredict,y1_train)]

  itera+=[i]

  cost_ls+=[linear_costfunc(x1_train,y1_train,coefmat)]

  if (abs(cost_ls[i]-cost_ls[i-1]))<=threshold:

    break

  print("For iteration: ",i," RMSE is: ", linear_rmse(ypredict,y1_train))

#print("Coeffients: ",min_cost)

plt.plot(itera,rmse)

plt.xlabel('Iteration')

plt.ylabel('RMSE')

plt.show()
iterations=1000

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

df_target=df2[['target']].values

features=df2.drop(columns=['target','Runtime','intercept'])

df_feat=features.sample(axis = 1,random_state=0,n=8) 

df_feat.insert(0,'intercept',1)

df_features=df_feat.values

x1_train, x1_test, y1_train, y1_test = train_test_split(df_features, df_target, test_size = 0.2, random_state = 0)
#Random 8 features for test and train

threshold=[0.000001]

alpha=[0.2]

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

iterations=1000

print('For features: ', df_feat.columns)

LinearReg(x1_train,x1_test,y1_train,y1_test,alpha,iterations,coefmat,threshold)
#Random 8 features within train

threshold=[0.000001]

alpha=[0.8]

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

iterations=1000

print('For features: ', df_feat.columns)

LinearReg(x1_train,x1_train,y1_train,y1_train,alpha,iterations,coefmat,threshold)
#Fixed 8 features for test and train

iterations=1000

df_target=df2[['target']].values

features=df2.drop(columns=['target','Runtime','intercept'])

ls=['MWG', 'NWG', 'KWG', 'MDIMC', 'NDIMC', 'STRM', 'SA', 'SB']

df_feat=features[ls] 

df_feat.insert(0,'intercept',1)

df_features=df_feat.values

x1_train, x1_test, y1_train, y1_test = train_test_split(df_features, df_target, test_size = 0.2, random_state = 0)

threshold=[0.000001]

alpha=[0.2]

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

print('For features: ', df_feat.columns)

LinearReg(x1_train,x1_test,y1_train,y1_test,alpha,iterations,coefmat,threshold)
#Random 8 features for test and train loop to validate

iterations=1000

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

df_target=df2[['target']].values

features=df2.drop(columns=['target','Runtime','intercept'])

itera=[]

lsrmse=[]

for i in range(50):

  print('For seed=', i)

  df_feat=features.sample(axis = 1,random_state=i,n=8) 

  df_feat.insert(0,'intercept',1)

  df_features=df_feat.values

  x1_train, x1_test, y1_train, y1_test = train_test_split(df_features, df_target, test_size = 0.2, random_state = 0)

  threshold=[0.000001]

  alpha=[0.2]

  coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

  iterations=1000

  print('For features: ', df_feat.columns)

  r=LinearReg(x1_train,x1_test,y1_train,y1_test,alpha,iterations,coefmat,threshold)

  itera+=[i]

  lsrmse+=[r]

plt.plot(itera,lsrmse)
#Random 8 features within train

iterations=1000

df_target=df2[['target']].values

features=df2.drop(columns=['target','Runtime','intercept'])

ls=['MWG', 'NWG', 'KWG', 'MDIMC', 'NDIMC', 'STRM', 'SA', 'SB']

df_feat=features[ls] 

df_feat.insert(0,'intercept',1)

df_features=df_feat.values

x1_train, x1_test, y1_train, y1_test = train_test_split(df_features, df_target, test_size = 0.2, random_state = 0)

threshold=[0.000001]

alpha=[0.8]

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

print('For features: ', df_feat.columns)

LinearReg(x1_train,x1_test,y1_train,y1_test,alpha,iterations,coefmat,threshold)
#Random 8 features within train loop to validate

iterations=1000

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

df_target=df2[['target']].values

features=df2.drop(columns=['target','Runtime','intercept'])

itera=[]

lsrmse=[]

for i in range(50):

  print('For seed=', i)

  df_feat=features.sample(axis = 1,random_state=i,n=8) 

  df_feat.insert(0,'intercept',1)

  df_features=df_feat.values

  x1_train, x1_test, y1_train, y1_test = train_test_split(df_features, df_target, test_size = 0.2, random_state = 0)

  threshold=[0.000001]

  alpha=[0.2]

  coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

  iterations=1000

  print('For features: ', df_feat.columns)

  r=LinearReg(x1_train,x1_train,y1_train,y1_train,alpha,iterations,coefmat,threshold)

  itera+=[i]

  lsrmse+=[r]

plt.plot(itera,lsrmse)
iterations=1000

mean = df2['target'].mean()

df2.loc[df2['target'] <= mean, 'target'] = 0

df2.loc[df2['target'] > mean, 'target'] = 1

df_target=df2[['target']].values

df_features=df2.drop(columns=['target','Runtime'],axis=1).values

x1_train, x1_test, y1_train, y1_test = train_test_split(df_features, df_target, test_size = 0.2, random_state = 0)

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)
def sigmoid(dfile,coefmat):

  y_hat=np.dot(dfile,coefmat.T)

  z=1/(1+np.exp(-y_hat))

  return z
#define cost function

def logit_costfunc(dfile,targetvar,coefmat):

  fterm=np.sum(np.dot(targetvar.T,np.log(sigmoid(dfile,coefmat))))

  sterm=np.sum(np.dot((1-targetvar).T,np.log(1-sigmoid(dfile,coefmat))))

  cost=-(fterm+sterm)/len(dfile)

  return cost
#define gradient decent considering fixed iterations

def logit_gdesc(dfile,targetvar,coefmat,alpha,iterations,threshold):

  cost_ls=[logit_costfunc(dfile,targetvar,coefmat)]

  gddf=pd.DataFrame(coefmat)

  for i in range(1,iterations):

    loss=sigmoid(dfile,coefmat)-targetvar

    dep=np.dot(loss.T,dfile)

    coefmat=coefmat-dep*alpha/len(dfile)   

    #print('matrix: ',coefmat)

    gddf=gddf.append(pd.DataFrame(coefmat),ignore_index=True)

    cost_ls+=[logit_costfunc(dfile,targetvar,coefmat)]

    #print('iteration=',i,' cost_ls:',cost_ls)

    if (abs(cost_ls[i]-cost_ls[i-1]))<=threshold:

      break

  gddf['cost']=cost_ls

  #print(gddf)

  print("Iterations needed to converge: ", i+1)

  min_cost=gddf[gddf.cost==min(gddf.cost)]

  print('Cost at convergance: ', cost_ls[i])

  min_cost=min_cost.drop(columns='cost')

  return min_cost
def log_predict(cost_mat,xtest):

  predic_target=xtest.dot(cost_mat.T)

  target= np.where(predic_target >= 0.5 , 1, 0)

  return target
def accuracy(ypredict,ytest):

  df = pd.DataFrame({'actual': ytest.flatten(), 'predicted': ypredict.flatten()})

  correct= df.loc[df['actual'] == df['predicted']]

  rate=len(correct)/len(ytest)   

  return rate
def LogisticReg(x1_train,x1_test,y1_train,y1_test,alpha,iterations,coefmat,threshold):

  if len(alpha)>1:

    coef_ls=[0]*len(alpha)

    ypredict=[0]*len(alpha)

    accu=[0]*len(alpha)

    for i, a in enumerate(alpha, start=0):

      coef_ls[i]=logit_gdesc(x1_train,y1_train,coefmat,a,iterations,threshold)

      ypredict[i]=log_predict(coef_ls[i],x1_test)

      accu[i]=accuracy(ypredict[i],y1_test)

      print("For learning rate=", a, " Accuracy is: ", accu[i])

      print("Coeffients: ",coef_ls[i])

    plt.plot(alpha,accu)

    plt.xlabel('Learning Rate')

    plt.ylabel('Accuracy')

    plt.show()

  elif len(threshold)>1:

    coef_ls=[0]*len(threshold)

    ypredict=[0]*len(threshold)

    accu=[0]*len(threshold)

    for i, t in enumerate(threshold, start=0):

      #print('t',t)

      coef_ls[i]=logit_gdesc(x1_train,y1_train,coefmat,alpha,iterations,t)

      ypredict[i]=log_predict(coef_ls[i],x1_test)

      #print('y_hat:',ypredict)

      accu[i]=accuracy(ypredict[i],y1_test)

      print("For threshold=", t, " Accuracy is: ", accu[i])

      print("Coeffients: ",coef_ls[i])

    plt.plot(threshold,accu)

    plt.xlabel('Threshold')

    plt.ylabel('Accuracy')

    plt.show()

  else:

    coef_ls=[0]

    ypredict=[0]

    accu=[0]

    for i in range(1):

      coef_ls[i]=logit_gdesc(x1_train,y1_train,coefmat,alpha,iterations,threshold)

      ypredict[i]=log_predict(coef_ls[i],x1_test)

      accu[i]=accuracy(ypredict[i],y1_test)

      print("For threshold=", threshold," and learning rate: ",alpha, " Accuracy is: ", accu[i])

      print("Coeffients: ",coef_ls[i])

    return accu[i]
#Part 1 for minimum rmse with train and test data

threshold=[0.000001]

alpha=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3]

LogisticReg(x1_train,x1_test,y1_train,y1_test,alpha,iterations,coefmat,threshold)
threshold=[0.000001]

alpha=[0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3]

LogisticReg(x1_train,x1_test,y1_train,y1_test,alpha,iterations,coefmat,threshold)
#Part 2 for minimum rmse within training data

threshold=[0.000001]

alpha=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5]

LogisticReg(x1_train,x1_train,y1_train,y1_train,alpha,iterations,coefmat,threshold)
threshold=[0.000001]

alpha=[0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5]

LogisticReg(x1_train,x1_train,y1_train,y1_train,alpha,iterations,coefmat,threshold)
#Part 1 for minimum rmse with train and test data

threshold=[0.0000005,0.000001,0.000002,0.000003]

alpha=[0.4]

iterations=1000

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

LogisticReg(x1_train,x1_test,y1_train,y1_test,alpha,iterations,coefmat,threshold)
#Error as a function of number of gradient descent iterations for test and train

threshold=[0.000001]

alpha=[0.6]

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

iterations=1000

itera=[]

accu=[]

cost_ls=[logit_costfunc(x1_train,y1_train,coefmat)]

gddf=pd.DataFrame(coefmat)

for i in range(1,iterations):

  loss=sigmoid(x1_train,coefmat)-y1_train

  dep=np.dot(loss.T,x1_train)

  coefmat=coefmat-dep*alpha/len(x1_train)   

  ypredict=log_predict(coefmat,x1_test)

  accu+=[accuracy(ypredict,y1_test)]

  itera+=[i]

  cost_ls+=[logit_costfunc(x1_train,y1_train,coefmat)]

  if (abs(cost_ls[i]-cost_ls[i-1]))<=threshold:

    break

  print("For iteration: ",i," accuracy is: ", accuracy(ypredict,y1_test))

#print("Coeffients: ",min_cost)

plt.plot(itera,accu)

plt.xlabel('Iteration')

plt.ylabel('Accuracy')

plt.show()
#Part 2 for minimum rmse within training data

threshold=[0.0000001,0.0000005,0.000001,0.000002,0.000003,0.000004,0.000005]

alpha=[1]

iterations=1000

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

LogisticReg(x1_train,x1_test,y1_train,y1_test,alpha,iterations,coefmat,threshold)
#Error as a function of number of gradient descent iterations within train

threshold=[0.000001]

alpha=[1]

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

iterations=1000

itera=[]

accu=[]

cost_ls=[logit_costfunc(x1_train,y1_train,coefmat)]

gddf=pd.DataFrame(coefmat)

for i in range(1,iterations):

  loss=sigmoid(x1_train,coefmat)-y1_train

  dep=np.dot(loss.T,x1_train)

  coefmat=coefmat-dep*alpha/len(x1_train)   

  ypredict=log_predict(coefmat,x1_train)

  accu+=[accuracy(ypredict,y1_train)]

  itera+=[i]

  cost_ls+=[logit_costfunc(x1_train,y1_train,coefmat)]

  if (abs(cost_ls[i]-cost_ls[i-1]))<=threshold:

    break

  print("For iteration: ",i," accuracy is: ", accuracy(ypredict,y1_train))

#print("Coeffients: ",min_cost)

plt.plot(itera,accu)

plt.xlabel('Iteration')

plt.ylabel('Accuracy')

plt.show()
iterations=1000

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

mean = df2['target'].mean()

df2.loc[df2['target'] <= mean, 'target'] = 0

df2.loc[df2['target'] > mean, 'target'] = 1

df_target=df2[['target']].values

features=df2.drop(columns=['target','Runtime','intercept'])

df_feat=features.sample(axis = 1,random_state=0,n=8) 

df_feat.insert(0,'intercept',1)

df_features=df_feat.values

x1_train, x1_test, y1_train, y1_test = train_test_split(df_features, df_target, test_size = 0.2, random_state = 0)
#Random 8 features for test and train

threshold=[0.000001]

alpha=[0.6]

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

iterations=1000

print('For features: ', df_feat.columns)

LogisticReg(x1_train,x1_test,y1_train,y1_test,alpha,iterations,coefmat,threshold)
#Random 8 features within train

threshold=[0.000001]

alpha=[1]

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

iterations=1000

print('For features: ', df_feat.columns)

LogisticReg(x1_train,x1_train,y1_train,y1_train,alpha,iterations,coefmat,threshold)
iterations=1000

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

mean = df2['target'].mean()

df2.loc[df2['target'] <= mean, 'target'] = 0

df2.loc[df2['target'] > mean, 'target'] = 1

df_target=df2[['target']].values

features=df2.drop(columns=['target','Runtime','intercept'])

ls=['MWG', 'NWG', 'KWG', 'MDIMC', 'NDIMC', 'STRM', 'SA', 'SB']

df_feat=features[ls] 

df_feat.insert(0,'intercept',1)

df_features=df_feat.values

x1_train, x1_test, y1_train, y1_test = train_test_split(df_features, df_target, test_size = 0.2, random_state = 0)

threshold=[0.000001]

alpha=[0.6]

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

print('For features: ', df_feat.columns)

LogisticReg(x1_train,x1_test,y1_train,y1_test,alpha,iterations,coefmat,threshold)
#Random 8 features for test and train for loop to validate

iterations=1000

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

mean = df2['target'].mean()

df2.loc[df2['target'] <= mean, 'target'] = 0

df2.loc[df2['target'] > mean, 'target'] = 1

df_target=df2[['target']].values

features=df2.drop(columns=['target','Runtime','intercept'])

itera=[]

lsaccu=[]

for i in range(50):

  print('For seed=', i)

  df_feat=features.sample(axis = 1,random_state=i,n=8) 

  df_feat.insert(0,'intercept',1)

  df_features=df_feat.values

  x1_train, x1_test, y1_train, y1_test = train_test_split(df_features, df_target, test_size = 0.2, random_state = 0)

  threshold=[0.000001]

  alpha=[0.6]

  coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

  iterations=1000

  print('For features: ', df_feat.columns)

  a=LogisticReg(x1_train,x1_test,y1_train,y1_test,alpha,iterations,coefmat,threshold)

  itera+=[i]

  lsaccu+=[a]

plt.plot(itera,lsaccu)
iterations=1000

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

mean = df2['target'].mean()

df2.loc[df2['target'] <= mean, 'target'] = 0

df2.loc[df2['target'] > mean, 'target'] = 1

df_target=df2[['target']].values

features=df2.drop(columns=['target','Runtime','intercept'])

ls=['MWG', 'NWG', 'KWG', 'MDIMC', 'NDIMC', 'STRM', 'SA', 'SB']

df_feat=features[ls] 

df_feat.insert(0,'intercept',1)

df_features=df_feat.values

x1_train, x1_test, y1_train, y1_test = train_test_split(df_features, df_target, test_size = 0.2, random_state = 0)

threshold=[0.000001]

alpha=[1]

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

print('For features: ', df_feat.columns)

LogisticReg(x1_train,x1_test,y1_train,y1_test,alpha,iterations,coefmat,threshold)
#Random 8 features within train to validate

iterations=1000

coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

mean = df2['target'].mean()

df2.loc[df2['target'] <= mean, 'target'] = 0

df2.loc[df2['target'] > mean, 'target'] = 1

df_target=df2[['target']].values

features=df2.drop(columns=['target','Runtime','intercept'])

itera=[]

lsaccu=[]

for i in range(50):

  print('For seed=', i)

  df_feat=features.sample(axis = 1,random_state=i,n=8) 

  df_feat.insert(0,'intercept',1)

  df_features=df_feat.values

  x1_train, x1_test, y1_train, y1_test = train_test_split(df_features, df_target, test_size = 0.2, random_state = 0)

  threshold=[0.000001]

  alpha=[1]

  coefmat=np.zeros((1,len(x1_train[0])),dtype=int)

  iterations=1000

  print('For features: ', df_feat.columns)

  a=LogisticReg(x1_train,x1_train,y1_train,y1_train,alpha,iterations,coefmat,threshold)

  itera+=[i]

  lsaccu+=[a]

plt.plot(itera,lsaccu)