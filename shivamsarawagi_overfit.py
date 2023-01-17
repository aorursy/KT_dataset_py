# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pystan as ps

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

train.head()
train.pop('id')                                                                                      

target = train.pop('target').astype(int)                                                             

                                                                                                     

test = pd.read_csv('../input/test.csv')                                                              

ids = test.pop('id')  
code = """                                                                                           

data {                                                                                               

  int N; //the number of training observations                                                       

  int N2; //the number of test observations                                                          

  int K; //the number of features                                                                    

  int y[N]; //the response                                                                           

  matrix[N,K] X; //the model matrix                                                                  

  matrix[N2,K] new_X; //the matrix for the predicted values                                          

}                                                                                                    

parameters {                                                                                         

  real alpha;                                                                                        

  vector[K] beta; //the regression parameters                                                        

}                                                                                                    

transformed parameters {                                                                             

  vector[N] linpred;                                                                                 

  linpred = alpha+X*beta;                                                                            

}                                                                                                    

model {                                                                                              

  alpha ~ cauchy(0,10); //prior for the intercept following Gelman 2008                              

                                                                                                     

  for(i in 1:K)                                                                                      

    beta[i] ~ student_t(1, 0, 0.03);                                                                 

                                                                                                     

  y ~ bernoulli_logit(linpred);                                                                      

}                                                                                                    

generated quantities {                                                                               

  vector[N2] y_pred;                                                                                 

  y_pred = alpha+new_X*beta; //the y values predicted by the model                                   

}                                                                                                    

"""               
data = {                                                                                             

    'N': 250,                                                                                        

    'N2': 19750,                                                                                     

    'K': 300,                                                                                        

    'y': target,                                                                                     

    'X': train,                                                                                      

    'new_X': test,                                                                                   

}                                                                                                    

                                                                                                     

sm = ps.StanModel(model_code=code)                                                               

fit = sm.sampling(data=data, seed=1234)                                                              

ex = fit.extract(permuted=True)                                                                      

target = np.mean(ex['y_pred'], axis=0)                                                               

df = pd.DataFrame({'id': ids, 'target': target})                                                     

df[['id', 'target']].to_csv('submission.csv', index=False) 