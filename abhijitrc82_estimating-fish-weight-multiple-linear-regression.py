import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv('/kaggle/input/fish-market/Fish.csv')

print(df.shape)

df.sample(10)
print(df.Species.unique())

print(df.info())
print(df.describe())
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb

%matplotlib inline
for specie in df.Species.unique():

    df_spe=df.query('Species==@specie')

    print(specie)

    for col in df_spe.columns:

        if(col =='Species'):continue

        df_spe.boxplot(col)

        plt.show()
df.query('Species=="Roach" & (Weight ==0 | Weight>350)')
df= df.drop([54])

df.query('Species =="Roach"').describe().T
df.iloc[40,1]=df.query('Species =="Roach"').describe().T['25%'].Weight
df.query('Species =="Roach"').describe().T
for col in df.columns:

    if(col =='Species'):continue

    df.boxplot(col)

    plt.show()
df.query('Weight> 1500')
df= df.query('Weight<= 1500')
df.query('Weight> 1500')
sb.pairplot(df, kind='scatter', hue='Species');
from sklearn import preprocessing

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn import metrics
# label_encoder object knows how to understand word labels. 

label_encoder = preprocessing.LabelEncoder() 

  

# Encode labels in column 'species'. 

df['Species']= label_encoder.fit_transform(df['Species']) 



df['Species'].unique()
X=df.drop(['Weight'] , axis=1, inplace=False)

X.head()
y= df[df.columns[1:2]]
lg = LinearRegression()

lstSeed=[]

lstRMSQ=[]

lstRSq=[]

for seed in range(0,150,10):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

    lg.fit(X_train, y_train) #training the algorithm

    pred = lg.predict(X_test)

    root_mean_sq = np.sqrt(metrics.mean_squared_error(y_test,pred))

    r_sq = metrics.r2_score(y_test,pred)

    lstRSq.append(r_sq)

    lstSeed.append(seed)

    lstRMSQ.append(root_mean_sq)
df_metric=pd.DataFrame({

    'Seed': lstSeed, 

    'RMSQ': lstRMSQ,

    'RSQ': lstRSq})

df_metric.head()
ax=df_metric.plot('Seed', 'RMSQ',legend=False)

ax2 = ax.twinx()

df_metric.plot('Seed', 'RSQ', ax=ax2,color="r",legend=False)

ax.figure.legend()

plt.show()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

lg.fit(X_train, y_train) #training the algorithm

pred = lg.predict(X_test)

print('root mean sq:',np.sqrt(metrics.mean_squared_error(y_test,pred)))

print('r squared:',metrics.r2_score(y_test,pred))
X=df.drop(['Weight'] , axis=1, inplace=False)

y= df[df.columns[1:2]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

X_train_inter = np.ones((X_train.shape[0],1))

X_train = np.concatenate((X_train_inter, X_train), axis = 1)



X_test_inter = np.ones((X_test.shape[0],1))

X_test = np.concatenate((X_test_inter, X_test), axis = 1)

print(X_train.shape)

print(X_test.shape)
def computeCost(X,y,theta):

    #number of training examples

    m= len(y)

    hypothesis= X.dot(theta)

    #Take a summation of the squared values

    delta=np.sum(np.square(hypothesis-y))

    J=(1/(2*m))*delta

    return J



def gradientDescent(X, y, theta, alpha, num_iters):

    #number of training examples

    m, n = np.shape(X)

    x_t = X.transpose()

    J_history = np.zeros((num_iters, 1))

    for i in range(num_iters):

        hypothesis = np.dot(X,theta)-y

        gradient = np.dot(x_t, hypothesis) / m

        #update the theta

        theta = theta- alpha*gradient

        J_history[i]=np.sum(hypothesis**2) / (2*m)

    return theta,J_history



def predict(x_test,theta):

    n = len(x_test)

    predicted_vals=[]

    for i in range(0,n):

        predicted_vals.append(np.matmul(theta.T,x_test[i,:]))

    return predicted_vals



def runEpoch(X,y,theta,alpha,iterations,epochs):

    dicGradient={}

    dicRMSQ={}

    dicRSQ={}

    dicJ_Hist={}

    J_hist=[]

    X_t_act, X_valid, y_t_act, y_valid = train_test_split(X, y, test_size=0.2, random_state=10)

    for epoch in range(epochs):

        print('Running Epoch {}'.format(epoch))

        theta,J_History=gradientDescent(X_t_act,y_t_act,theta,alpha,iterations)

        dicGradient[epoch]=(theta,J_History)

        J_hist.extend(J_History)

        pred_vals=predict(X_valid,theta)

        root_mean_sq = np.sqrt(metrics.mean_squared_error(y_valid,pred_vals))

        r_sq = metrics.r2_score(y_valid,pred_vals)

        dicRMSQ[epoch]=root_mean_sq

        print('Epoch {0}: RMSQ {1}'.format(epoch,root_mean_sq))

        dicRSQ[epoch]=r_sq

    key_min = min(dicRMSQ.keys(), key=(lambda k: dicRMSQ[k]))

    return dicGradient[key_min][0],J_hist
n=X_train.shape[1]

theta=np.zeros((n, 1))

theta,J_History=runEpoch(X_train,y_train,theta,0.00065,4000,25)

print(theta)

plt.plot(J_History);

plt.show();
pred_vals=predict(X_test,theta)

preds=[]

for pred in pred_vals:

    preds.append(abs(pred[0]))
root_mean_sq = np.sqrt(metrics.mean_squared_error(y_test,preds))

r_sq = metrics.r2_score(y_test,preds)

print('root mean sq:',root_mean_sq)

print('r squared:',r_sq)
def normalEquation(X,y):

    x_trans=X.T

    inv=np.linalg.pinv(np.dot(x_trans,X))

    theta=np.dot(np.dot(inv,x_trans),y)

    return theta
theta_ne= normalEquation(X_train,y_train)

print(theta_ne)
pred_vals=predict(X_test,theta_ne)

preds=[]

for pred in pred_vals:

    preds.append(abs(pred[0]))

root_mean_sq = np.sqrt(metrics.mean_squared_error(y_test,preds))

r_sq = metrics.r2_score(y_test,preds)

print('root mean sq:',root_mean_sq)

print('r squared:',r_sq)