import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap
dfTrain = pd.read_csv('../input/ClassificationDataTraining.csv')   #Training Dataset

dfTrain.head()
DistinctClasses=np.array(dfTrain['Y'].unique())

print(DistinctClasses)
pd.plotting.scatter_matrix(dfTrain, alpha=1, diagonal='kde',color='r')

plt.show()
cmap = ListedColormap(['blue', 'red']) 







plt.scatter(dfTrain.loc[:,['X1']].values,dfTrain.loc[:,['X2']].values, c=dfTrain.loc[:,['Y']].values, cmap=cmap)

plt.show()
def extractFeatures(df):

    df_Features=df.iloc[:,0:2]

    df_Label=df.iloc[:,2:3]

    X=df_Features.values

    Y=df_Label.values

    return X,Y
X,Y=extractFeatures(dfTrain)
def predict(inputX,trainX,trainY,K):

    trainBatchSize=trainX.shape[0]

    predictBatchSize=inputX.shape[0]

    pY=np.zeros((inputX.shape[0],1))

    distanceList=np.zeros(trainY.shape)

    for i in range(predictBatchSize): 

        

        #Step1:Calculate Distances

        distanceValues=np.linalg.norm(inputX[i,:]-trainX,axis=1)

        distanceList=np.column_stack((distanceValues,trainY))

          

        #Step2: Sort Distances

        sortedList=distanceList[distanceList[:,0].argsort()]

        

       

        #Step3: Pick top K

        topKList=sortedList[:K,:]

                

        

        

        #Step4: GetMost voted class top K    

        DistinctClassesWithCount=np.column_stack((DistinctClasses,np.zeros(DistinctClasses.shape)))

        

        for cls in DistinctClasses:           

            DistinctClassesWithCount[cls,1]= len(topKList[np.where(topKList[:,1]==cls)])

            

    

        

        mostVoted=np.argmax(np.max(DistinctClassesWithCount, axis=1))

        

        pY[i]=mostVoted

            

                    

    

    return pY
def accurracy(Y1,Y2):

    m=np.mean(np.where(Y1==Y2,1,0))    

    return m*100
K=25

pY=predict(X,X,Y,K) 

print(accurracy(Y, pY))
plt.scatter(X[:,0],X[:,1], c=Y[:,0], cmap=cmap) 

###########################################################################

#Predict for each X1 and X2 in Grid 

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

u = np.linspace(x_min, x_max, 20) 

v = np.linspace(y_min, y_max, 20) 



U,V=np.meshgrid(u,v)

UV=np.column_stack((U.flatten(),V.flatten())) 

W=predict(UV, X,Y,K) 

plt.scatter(U.flatten(), V.flatten(),  c=W.flatten(), cmap=cmap,marker='.', alpha=0.1)



###########################################################################

#Exact Decision Boundry can be plot with contour

z = np.zeros(( len(u), len(v) )) 

for i in range(len(u)): 

    for j in range(len(v)): 

        uv= np.column_stack((np.array([[u[i]]]),np.array([[v[j]]])))               

        z[i,j] =predict(uv, X,Y,K) 

z = np.transpose(z) 

plt.contour(u, v, z)

###########################################################################

plt.show()