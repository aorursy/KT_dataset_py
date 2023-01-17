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
train_data=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/train.csv")

test_data=pd.read_csv("/kaggle/input/covid19-global-forecasting-week-1/test.csv")
#Splitting Data and labels 

X_train=train_data[["Lat","Long","Date"]]

Y_train=train_data[["ConfirmedCases","Fatalities"]]

X_test=test_data[["Lat","Long","Date"]]
#Splitting date into seperate months and days attributes

new=X_train["Date"].str.split("-", n = 2, expand = True) 

X_train["Month"]=new[1].astype(np.int)

X_train["Day"]=new[2].astype(np.int)

X_train.drop(["Date"], axis=1,inplace=True)

new=X_test["Date"].str.split("-", n = 2, expand = True) 

X_test["Month"]=new[1].astype(np.int)

X_test["Day"]=new[2].astype(np.int)

X_test.drop(["Date"], axis=1,inplace=True)
#creating a Model class

class Model:

    @classmethod

    def relu(self,x):

        return np.maximum(0,x)

    def initialize_params(self,layers):

        params={}

        for i in range(1,len(layers)):

            w=np.random.randn(layers[i-1],layers[i])*0.01

            b=np.zeros((layers[i],1))

            params.update({"W"+str(i):w})

            params.update({"b"+str(i):b})

        return (params)

    def forward(self,params, nof_layers, activations):

        for i in range(1,nof_layers):

            a=self.relu(np.dot(params["W"+str(i)].T,activations["A"+str(i-1)])+params["b"+str(i)])

            activations.update({"A"+str(i):a})

        return (activations)

    def cost(self,activations, Y, nof_layers):

        m=Y.shape[1]

        lhs=(activations["A"+str(nof_layers-1)]-Y)**2

        cost=np.sum(lhs,axis=0,keepdims=True)

        cost=(1/m)*(np.sum(cost,axis=1)**0.5)

        return (cost)

    def calc_gradient(self,Y, params, activations, nof_layers):

        grads={}

        m=Y.shape[1]

        Alast=activations["A"+str(nof_layers-1)]

        dAlast= (2/m)*(Alast-Y)

        dAlastT=dAlast.T

        for i in range(nof_layers-1,0,-1):

            if i==nof_layers-1:

                dWl=np.dot(activations["A"+str(i-1)],dAlastT)

                grads.update({"dW"+str(i):dWl})

                grads.update({"db"+str(i):dAlast})

                W=params["W"+str(i)]

            else:

                dWl=np.dot(np.dot(activations["A"+str(i-1)],dAlastT),W.T)

                grads.update({"dW"+str(i):dWl})

                dbl=np.dot(W,dAlast)

                grads.update({"db"+str(i):dbl})

                W=np.dot(params["W"+str(i)],W)

        return (grads)

    def backpropagate(self,params, grads, lr, nof_layers):

        for i in range(1,nof_layers):

            tW=params["W"+str(i)]-(lr*grads["dW"+str(i)])

            params.update({"W"+str(i):tW})

            tb=params["b"+str(i)]-(lr*np.sum(grads["db"+str(i)],axis=1, keepdims=True))

            params.update({"b"+str(i):tb})

        return (params)

    def fit(self,X, Y, params, layers, noi=1000, lr=0.001):

        activations={"A0":X}

        nof_layers=len(layers)

        for i in range(noi):

            activations=self.forward(params, nof_layers, activations)

            cost= self.cost(activations, Y, nof_layers)

            print("Cost after iteration "+str(i+1)+" : "+str(cost))

            grads=self.calc_gradient(Y, params, activations, nof_layers)

            params=self.backpropagate(params, grads, lr, nof_layers)

        return (params)

    def predict(self,X, params, nof_layers):

        activations={"A0":X}

        activations=self.forward(params, nof_layers, activations)

        return (activations["A"+str(nof_layers-1)].astype(np.int))
layers=[4,2]

model=Model()

params=model.initialize_params(layers)
params=model.fit((X_train.values).T, (Y_train.values).T, params,layers, noi=500000, lr=0.0002)
pred=model.predict((X_test.values).T, params, len(layers))

pred
forecast_ids=test_data[["ForecastId"]].values
submit=zip(forecast_ids, (pred[0].T).reshape(12212,1), (pred[1].T).reshape(12212,1))

submission_df = pd.DataFrame(columns=["ForecastId","ConfirmedCases","Fatalities"])
for i in submit:

    submission_df = submission_df.append({"ForecastId":i[0][0],

                                          "ConfirmedCases":i[1][0],

                                          "Fatalities":i[2][0]},

                                         ignore_index=True)
submission_df.to_csv("submission.csv",index=False)