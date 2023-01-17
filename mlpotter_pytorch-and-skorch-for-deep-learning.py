import os
print(os.listdir("../input"))
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
import torch

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
# display first 5 rows of dataset to see what we are looking at
dataset.head()
# show the distributions of data 
dataset.describe()
# The serial No. adds nothing to predicting the chance of admittance
dataset = dataset.drop('Serial No.',axis=1)
sns.pairplot(dataset,diag_kind='kde',plot_kws={'alpha': .2});
sns.factorplot(y="Chance of Admit ",x='Research',data=dataset,kind='box');
dataset.loc[:,['Research','Chance of Admit ']].groupby('Research').describe()
sns.distplot(dataset[dataset.loc[:,'Research'] == 1].loc[:,['Chance of Admit ']],kde=True);
sns.distplot(dataset[dataset.loc[:,'Research'] == 0].loc[:,['Chance of Admit ']],kde=True);
plt.xlabel('Chance of Admittance')
plt.ylabel('Count')
plt.title('Research vs No Research');
plt.legend(['Research','No Research']);
sns.factorplot(x='University Rating',y='Chance of Admit ',kind='box',data=dataset);
sns.factorplot(x='SOP',y='Chance of Admit ',kind='box',data=dataset);
sns.factorplot(x='LOR ',y='Chance of Admit ',kind='box',data=dataset);
target = dataset.pop('Chance of Admit ')
# split data into train test 
X_train,X_test,y_train,y_test = train_test_split(dataset.values.astype(np.float32),
                                                 target.values.reshape(-1,1).astype(np.float32),
                                                 test_size=.2,
                                                random_state=42)
# normalize data to 0 mean and unit std
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
import skorch
from skorch import NeuralNetRegressor

from sklearn.model_selection import RandomizedSearchCV

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class MyModule(nn.Module):
    def __init__(self,num_units=10,nonlin=F.relu,drop=.5):
        super(MyModule,self).__init__()
        
        self.module = nn.Sequential(
            nn.Linear(7,num_units),
            nn.LeakyReLU(),
            nn.Dropout(p=drop),
            nn.Linear(num_units,1),
        )
        
    def forward(self,X):
        X = self.module(X)
        return X
net = NeuralNetRegressor(
    MyModule,
    criterion=nn.MSELoss,
    max_epochs=10,
    optimizer=optim.Adam,
    optimizer__lr = .005
)
lr = (10**np.random.uniform(-5,-2.5,1000)).tolist()
params = {
    'optimizer__lr': lr,
    'max_epochs':[300,400,500],
    'module__num_units': [14,20,28,36,42],
    'module__drop' : [0,.1,.2,.3,.4]
}

gs = RandomizedSearchCV(net,params,refit=True,cv=3,scoring='neg_mean_squared_error',n_iter=100)
%%capture
gs.fit(X_train_scaled,y_train);
# Utility function to report best scores (found online)
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
# review top 10 results and parameters associated
report(gs.cv_results_,10)
# get training and validation loss
epochs = [i for i in range(len(gs.best_estimator_.history))]
train_loss = gs.best_estimator_.history[:,'train_loss']
valid_loss = gs.best_estimator_.history[:,'valid_loss']
plt.plot(epochs,train_loss,'g-');
plt.plot(epochs,valid_loss,'r-');
plt.title('Training Loss Curves');
plt.xlabel('Epochs');
plt.ylabel('Mean Squared Error');
plt.legend(['Train','Validation']);
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score
# predict on test data
y_pred = gs.best_estimator_.predict(X_test_scaled.astype(np.float32))
# get RMSE
MSE(y_test,y_pred)**(1/2)
sns.kdeplot(y_pred.squeeze(), label='estimate', shade=True)
sns.kdeplot(y_test.squeeze(), label='true', shade=True)
plt.xlabel('Admission');
sns.distplot(y_test.squeeze()-y_pred.squeeze(),label='error');
plt.xlabel('Admission Error');
# show R^2 plot
print(r2_score(y_test,y_pred))
plt.plot(y_pred,y_test,'g*')
plt.xlabel('predicted')
plt.ylabel('actual')
plt.title('$R^{2}$ visual');

# show where the big errors were
errors = np.where(abs(y_test-y_pred)>.2)
for tup in zip(y_test[errors],y_pred[errors]):
    print(tup)