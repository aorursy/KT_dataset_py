# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from scipy import stats
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
# Any results you write to the current directory are saved as output.
from kmodes.kmodes import KModes
df_allData=pd.read_csv('../input/BlackFriday.csv')
print(df_allData.sample(n=2))
groupByUserData=df_allData.groupby(['User_ID'])

times=df_allData['User_ID'].value_counts()
times=times.sort_index()

#get the mean
meanData=groupByUserData.mean()

#get the mode
modeData=groupByUserData.agg(lambda x: stats.mode(x)[0][0])

mean_mode_data={'Gender':modeData['Gender'],'Occupation':modeData['Occupation'],'Age':modeData['Age'],'City_Category':modeData['City_Category'],'Marital_Status':modeData['Marital_Status'],'Product_CateGory_1':modeData['Product_Category_1'],'Stay_In_Current_City_Years':modeData['Stay_In_Current_City_Years']}
mean_mode_data=pd.DataFrame(mean_mode_data)
mean_mode_data['times']=times
mean_mode_data['Gender_M']=pd.get_dummies(mean_mode_data['Gender'])['M']
mean_mode_data=mean_mode_data.drop(['Gender'],axis=1)
mean_mode_data['Purchase']=meanData['Purchase']

print (mean_mode_data.sample(2))
X=pd.DataFrame({'Gender':modeData['Gender'],'Occupation':modeData['Occupation'],'Age':modeData['Age'],'City_Category':modeData['City_Category'],'Marital_Status':modeData['Marital_Status'],'Product_CateGory_1':modeData['Product_Category_1'],"Stay_In_Current_City_Years":modeData["Stay_In_Current_City_Years"]})

one_hot_city=pd.get_dummies(mean_mode_data['City_Category'])
one_hot_age=pd.get_dummies(mean_mode_data['Age'])
one_hot_occupation=pd.get_dummies(mean_mode_data['Occupation'])
one_hot_years=pd.get_dummies(mean_mode_data['Stay_In_Current_City_Years'])
one_hot_product=pd.get_dummies(mean_mode_data['Product_CateGory_1'])
XX=pd.concat([one_hot_age,one_hot_city,one_hot_occupation,one_hot_years,one_hot_product],axis=1)
XX['Gender_M']=mean_mode_data['Gender_M']
XX['Marital_Status']=mean_mode_data['Marital_Status']

print ("categorical data:")
print(X.sample(2))
print("one-hot encoding data:")
print(XX.sample(2))
from sklearn.metrics import jaccard_similarity_score
ecArr=[]
jcArr=[]
jcXArr=[]
for i in range(2,10):
    km=KModes(n_clusters=i)
    y=km.fit_predict(X)
    tempArrjc=[]
    tempArrec=[]
    tempArrjcX=[]
    for j in range(i):
        #print(sum(y==j))
        #print(XX[y==j].mode())
        jcscore=[]
        ecscore=[]
        jcXscore=[]
        for k in XX[y==j].T:
            try:
                #jcscore.append(jaccard_similarity_score(XX.loc[k],XX[y==j].mode().T[0]))
                
                ecscore.append(np.linalg.norm(np.array(XX.loc[k])-np.array(XX[y==j].mode().T[0])))
                
                jcXscore.append(jaccard_similarity_score(list(X.loc[k]),list(X[y==j].mode().T[0])))

            except:
                #print(XX.loc[k].T)
                #print(XX[y==j].mode())
                print(k)
                break;
        #print(np.mean(jcscore))
        #tempArrjc.append(np.mean(jcscore))
        #tempArrec.append(np.mean(ecscore))
        tempArrjcX.append(np.mean(jcXscore))

    print("n_cluster =",i,":",np.mean(tempArrjcX))
    #jcArr.append(np.mean(tempArrjc))
    #ecArr.append(np.mean(tempArrec))
    jcXArr.append(np.mean(tempArrjcX))
XXXX=X.drop(['Marital_Status','Product_CateGory_1','Stay_In_Current_City_Years','Age'],axis=1)
print(XXXX.sample(2))
from sklearn.metrics import jaccard_similarity_score
ecArr=[]
jcArr=[]
jcXArr=[]
for i in range(10,11):
    km=KModes(n_clusters=i)
    y=km.fit_predict(XXXX)
dis_jc=[]
dis_ec=[]
for i in range(10):
    dis_jc.append(jaccard_similarity_score(list(XXXX[y==i].mode().T[0]),list(XXXX[y!=i].mode().T[0])))
    
print("average jc distance in selected features:",np.mean(dis_jc))
    
for i in range(10):
    dis_ec.append(np.linalg.norm((np.array(XX[y==i].mode().T[0])-np.array(XX[y!=i].mode().T[0]))))
    
print("average ec distance in all one-hot features:",np.mean(dis_ec))   
purchase_y=pd.DataFrame({"y":y,"Purchase":mean_mode_data["Purchase"]})
plt.scatter(purchase_y['y'],purchase_y['Purchase'])
for i in range(10):
    plt.scatter(i,purchase_y[purchase_y['y']==i].Purchase.mean(),c='r')
XXXXX=X.drop(['Stay_In_Current_City_Years'],axis=1)
print(XXXXX.sample(2))
ecArr=[]
jcArr=[]
jcXArr=[]
for i in range(10,11):
    km=KModes(n_clusters=i)
    y=km.fit_predict(XXXXX)

dis_jc=[]
dis_ec=[]
for i in range(10):
    dis_jc.append(jaccard_similarity_score(list(XXXXX[y==i].mode().T[0]),list(XXXXX[y!=i].mode().T[0])))
    
for i in range(10):
    dis_ec.append(np.linalg.norm((np.array(XX[y==i].mode().T[0])-np.array(XX[y!=i].mode().T[0]))))
    
print("average jc distance in selected features:",np.mean(dis_jc))
print("average ec distance in all one-hot features:",np.mean(dis_ec))
purchase_y=pd.DataFrame({"y":y,"Purchase":mean_mode_data["Purchase"]})
plt.scatter(purchase_y['y'],purchase_y['Purchase'])
for i in range(10):
    plt.scatter(i,purchase_y[purchase_y['y']==i].Purchase.mean(),c='r')