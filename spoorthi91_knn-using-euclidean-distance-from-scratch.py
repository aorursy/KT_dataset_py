import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
def Euclidean_dist(pt1,pt2):
    distance=0.0
    for i in range(len(pt1)):
        distance += (pt1[i]-pt2[i])**2
    return math.sqrt(distance)
def Nearest_neighbors(train,test_obs,n):
    neighbor_distance= []
    for i in range(len(train)):
        l1=list(train.iloc[i,:])+[Euclidean_dist(train.iloc[i,:-1],test_obs)]
        neighbor_distance= neighbor_distance+[l1]
    neighbor_distance.sort(key=lambda x: x[-1])
    nearest_neighbors= [neighbor_distance[i] for i in range(0,n)]
    y_pred= [i[-2] for i in nearest_neighbors]
    return(int(max(y_pred,key=y_pred.count)))
def Prediction(train,test_obs,n):
    
    NN=Nearest_neighbors(train,test_obs,3)
    M= [i[n-1] for i in NN]
    
    return(test_obs+[max(M)])
def Normalize(data):
    df1=[]
    for i in range(len(data.columns)):
        z=[]
        z= [(k-np.mean(df.iloc[:,i]))/np.std(df.iloc[:,i]) for k in df.iloc[:,i]]
        df1.append(z)
    df1=pd.DataFrame(df1)
    df1=df1.T
    df1.columns=data.columns
    return(df1)
def F_score(Act,Pred):
    ConfusionMatrix= confusion_matrix(Act,Pred)
    
    return((2*ConfusionMatrix[1,1])/(2*ConfusionMatrix[1,1]+ConfusionMatrix[1,0]+ConfusionMatrix[0,1]))
def Accuracy(Act,Pred):
    ConfusionMatrix= confusion_matrix(Act,Pred)
    #return(ConfusionMatrix)
    return((ConfusionMatrix[0,0]+ConfusionMatrix[1,1])/(len(Act)))
df= pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.describe()
df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]= df[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
df.fillna(df.mean(),inplace=True)
df.describe()
#Distribution plots of predictors

df.hist(bins=10,figsize=(15,10))
plt.figure(figsize=(15,10))
p=sns.heatmap(df.corr(),annot=True)
sns.pairplot(data=df,hue='Outcome')
X=df.drop(columns='Outcome')
Y=df['Outcome']
X= Normalize(X)
#A sneakpeak of Normalized data
X.head()
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.3,random_state=5)
X_train=X_train.join(Y_train)
print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape,sep='\n')
Acc=[]
for j in range(1,20):
    pred=[]
    for i in range(len(X_test)):
        pred.append([Nearest_neighbors(X_train,X_test.iloc[i,:],j)])
    Acc= Acc+([Accuracy(Y_test,pred)])
Acc
pred=[]
for i in range(len(X_test)):
    pred.append(Nearest_neighbors(X_train,X_test.iloc[i,:],Acc.index(max(Acc))+1))
    
X_test['Pred']= pred
X_test['Outcome']= Y_test
from sklearn.metrics import classification_report

print(classification_report(X_test['Outcome'], X_test['Pred']))
pd.crosstab(X_test['Outcome'], X_test['Pred'], rownames=['True'], colnames=['Predicted'], margins=True)
