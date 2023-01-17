import torch

from torch import log,from_numpy,no_grad,nn,optim



import numpy as np

import pandas as pd



from sklearn.linear_model import LinearRegression





df=pd.read_csv('/kaggle/input/weka-scale/weka_scale.csv')



#Embaralha o dataframe, apartir de um estado predefindo

df=df.sample(frac=1.0, random_state=1)





x=df.iloc[100000:,:8].values

y=df.iloc[100000:,-1:].values

reg=LinearRegression().fit(x, y)





X=from_numpy(x).float()

Y=from_numpy(y).float()



#modelo=nn.Linear(8,1)

modelo=nn.Sequential(

    nn.Linear(8, 1)

)





#Otimiza para que o pred_y fique próximo do PL observado

loss_func = torch.nn.MSELoss() 

otimizador = torch.optim.Adam(modelo.parameters(), lr = 0.1) 



for _ in range(500):

    pred_y=modelo(X)



    # Compute and print loss 

    loss = loss_func(pred_y, Y) 



    otimizador.zero_grad() 

    loss.backward() 

    otimizador.step()



    

    

    

#Otimiza para maximizar o crescimento da banca    

def loss_somalog(y_pred,y):

    return -log(1+y*y_pred.relu()).sum()



otimizador = torch.optim.Adam(modelo.parameters(), lr = 0.01) 

for _ in range(300):

    pred_y=modelo(X)



    # Compute and print loss 

    loss = loss_somalog(pred_y, Y) 



    otimizador.zero_grad() 

    loss.backward() 

    otimizador.step()    

    



    

    

#Realiza o teste comparando a lucrativida da regressão linear com a rede neural    

Y=df.iloc[:100000,-1].values   

Y_nn=modelo(from_numpy(df.iloc[:100000,:8].values).float()).cpu().detach().numpy()

Y_reg=reg.predict(df.iloc[:100000,:8].values)





print('NN:', sum(np.log(1+y*y_pred) for y_pred,y in zip(Y_nn,Y) if y_pred>0) )

print('LR:', sum(np.log(1+y*y_pred) for y_pred,y in zip(Y_reg,Y) if y_pred>0) )