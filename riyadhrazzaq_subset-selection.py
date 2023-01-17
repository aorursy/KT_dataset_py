import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from itertools import permutations
import pandas as pd
class BestSubset:
    def __init__(self, metric='Cp'):
        self.metric = metric
        self.modelContainer = []
    
    def fit(self, X, y):
        nSamples, nFeats = X.shape
        Mo = LinearRegression().fit(np.zeros((nSamples,1)),y) # this will return mean always
        self.modelContainer.append([]) #empty predictors
        
        for k in range(1,nFeats+1):
            print(f'Training for {k} predictors')
            tmp = list(permutations(range(nFeats),r=k))
            tmp = [tuple(sorted(x)) for x in tmp]
            feature_combinations = list(set(tmp))
            tmpModelContainer = []
            
            for t in range(len(feature_combinations)):
                p = np.array(feature_combinations[t])
#                 print(p.shape, p)
                tmpX = X[:,p]
                model = LinearRegression().fit(tmpX, y)
                r2 = model.score(tmpX, y)
                tmpModelContainer.append((p,r2))
            
            tmpModelContainer = sorted(tmpModelContainer, key=lambda x: x[1], reverse=True)
            self.modelContainer.append(tmpModelContainer[0][0]) # only best model's predictors
        
        print("Comparing final K models", X.shape)
        Cp = []
        for k in range(nFeats):
            predictors = self.modelContainer[k]
            print(f"{k}th model: ",len(predictors))
            
            if len(predictors) == 0:
                model = LinearRegression().fit(np.zeros((nSamples, 1)), y)
                yhat = model.predict(np.zeros((nSamples, 1)))
            else:
                model = LinearRegression().fit(X[:, predictors], y)
                yhat = model.predict(X[:, predictors])
                
            rss = np.sum((y - yhat)**2)
            est_var = np.var( (y-yhat) )
            _Cp = ((rss + (2 * len(predictors) * est_var)) / nSamples)
            Cp.append(_Cp)
            
        idxmin = np.argmin(Cp)
        return self.modelContainer[idxmin], Cp
from sklearn.preprocessing import OrdinalEncoder
df = pd.read_csv("../input/ISLR-Auto/Credit.csv")
df.drop(['Unnamed: 0'], axis=1, inplace=True)
X = df.loc[:,df.columns!='Rating'].values
y = df['Rating'].values
predictiors = df.columns[df.columns!='Rating']
oe = OrdinalEncoder().fit(X)
X = oe.transform(X)
%%time
bs = BestSubset()
best_predictors, scores = bs.fit(X,y)
print(f"Best predictors are {df.columns[best_predictors].values}")
class ForwardStepwise:
    def __init__(self):
        pass
    
    def fit(self, X, y):
        nSamples, nFeats = X.shape
        predictors = [] # predictors[0:i] will have best model for i number of predictors.
        for k in range(0, nFeats):
            unused_indices = [i for i in range(nFeats) if i not in predictors]
            tmp_scores = []
            for j in unused_indices:
                p = predictors + [j]
                model = LinearRegression().fit(X[:, p], y)
                score = model.score(X[:, p], y)
                tmp_scores.append(score)
            mx = np.argmax(tmp_scores)
            predictors.append(unused_indices[mx])
            
        Cp = []
        
        for k in range(nFeats):
            if k == 0:
                p = []
                model = LinearRegression().fit(np.zeros((nSamples,1)), y)
                yhat = model.predict(np.zeros((nSamples,1)))
            else:
                p = predictors[0:k]
                model = LinearRegression().fit(X[:, p], y)
                yhat = model.predict(X[:, p])
                
            rss = np.sum((y - yhat)**2)
            est_var = np.var( (y-yhat) )
            _Cp = ((rss + (2 * len(p) * est_var)) / nSamples)
            Cp.append(_Cp)
            
        mn = np.argmin(Cp)
        best_predictors = predictors[:mn]
        return best_predictors
%%time
fs = ForwardStepwise()
best = fs.fit(X, y)
print(f"Best predictors are {df.columns[best].values}")