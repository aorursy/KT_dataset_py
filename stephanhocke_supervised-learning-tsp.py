import numpy as np 

import pandas as pd 

import math

from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import roc_auc_score,plot_roc_curve





# constants

n = 15



def convertSol(solStr):

    y = np.zeros( (1,int(n*(n-1)/2)) )

    nlist = solStr.split("-")

    nlist = [int(val) for i,val in enumerate(nlist)]

    for i in range(0,n):

        start = min(nlist[i],nlist[i+1])

        end = max(nlist[i],nlist[i+1])



        aux = [n-(1+i) for i in range(0,start)]

        midx = sum(aux)+(end-start-1)

        y[0][midx] = 1

    

    return y







data = pd.read_csv('/kaggle/input/15k-tsps-with-optimal-tours/tspData.csv')



nnInput = np.zeros( (len(data),n*n) )

nnOutput = np.zeros( (len(data),int(n*(n-1)/2)) )





for idx,val in data.iterrows():

    xc = [v for i,v in enumerate(val) if i>0 and i < 16]

    yc =  [v for i,v in enumerate(val) if i >= 16]

    cc = np.zeros( (n,2) )

    cc[:,0] = xc

    cc[:,1] = yc



    dists = [math.sqrt(pow(p1[0]-p2[0],2)+pow(p1[1]-p2[1],2)) for i,p1 in enumerate(cc) for j,p2 in enumerate(cc)]

    np.reshape(dists, (1,n*n) )

    nnInput[idx] = dists

    nnOutput[idx] = convertSol(val[0])









x = nnInput

y = nnOutput



print(nnOutput)



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2)



# multi layer perceptron MLP

nn = MLPClassifier(random_state=1234)

# parameter grid

paras = {'hidden_layer_sizes':[ (int(n/3),), (n,), (n,n),(25,n,n,25)],

         'activation':['logistic'],

         'max_iter':[200,500,2000],

         'alpha':[0.0001,0.05]

}



gs_nn = GridSearchCV(nn,paras,cv=5,scoring='neg_mean_squared_error',verbose=1,n_jobs=6)

gs_nn.fit(x_train,y_train)



print("Best neural network paras: {}".format(gs_nn.best_params_))

print("Best neural network score: {}".format(gs_nn.best_score_))





probas = gs_nn.predict_proba(x_test)

print(probas)


