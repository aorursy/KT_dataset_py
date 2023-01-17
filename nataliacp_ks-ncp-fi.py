import os
import numpy as np
import pandas as pd
%pylab inline
multipleCh_init=pd.read_csv('../input/multipleChoiceResponses.csv')
Femme=multipleCh_init[multipleCh_init.Q1=='Female']
X=Femme[['Q2','Q24','Q4','Q9','Q23','Q5','Q17','Q3']] 
X=X.dropna() #It is necessary to eliminate rows with nan values
# We are going to ignore females which did not respond specifically to all questions. 
X=X[X.Q4.str.contains('I prefer not to answer')==False]
X=X[X.Q9.str.contains('I do not wish to disclose my approximate yearly compensation')==False]
X=X[X.Q3.str.contains('I do not wish to disclose my location')==False]
X.head()
a=multipleCh_init[['Q2','Q24','Q4','Q9','Q23','Q5','Q17','Q3']].iloc[0,:]
for i in range(a.shape[0]):
    print(X.columns[i],a[i])
X.shape # shape of the survey
n=X.shape[0]     #number of respondents
p=X.shape[1]-3   # number of variables in which we are going to perform PCA
X1=np.zeros((n,p)) 
X1.shape
X.Q2.unique() 
for i in range(0,n):
    if(X.Q2.iloc[i]=='80+'):
        X1[i,0]=90
    else:
        lower=int(X.Q2.iloc[i][0]+X.Q2.iloc[i][1])
        upper=int(X.Q2.iloc[i][3]+X.Q2.iloc[i][4])
        X1[i,0]=(upper+lower)/2
    
X.Q24.unique()
import re
for i in range(0,n):
    if(X.Q24.iloc[i]=='40+'):
        X1[i,1]=45.0    
    elif(len([float(s) for s in re.findall(r'-?\d+\.?\d*', X.Q24.iloc[i])])>0): 
        #re.findall finds the caracters that correspond to numbers in a string
        a=[float(s) for s in re.findall(r'-?\d+\.?\d*', X.Q24.iloc[i])]
        a=np.absolute(a)
        prom=np.sum(a)/2
        X1[i,1]=prom
    else:
        X1[i,1]=0.0     
X.Q4.unique()
for i in range(0,n):
    if(X.Q4.iloc[i]=='Bachelor’s degree' or X.Q4.iloc[i]=='Professional degree' ):
        X1[i,2]=1.0
    elif(X.Q4.iloc[i]=='Master’s degree'):
        X1[i,2]=2.0
    elif(X.Q4.iloc[i]=='Doctoral degree'):
        X1[i,2]=3.0
    else:
        X1[i,2]=0.0    
X.Q9.unique()
for i in range(0,n):
    if(X.Q9.iloc[i]=='500,000+'):
        X1[i,3]=750000 
    else:
        a=[float(s) for s in re.findall(r'-?\d+\.?\d*', X.Q9.iloc[i])]
        #re.findall finds the caracters that correspond to numbers in a string
        a=np.absolute(a)
        prom=np.sum(a[0:2])/2
        X1[i,3]=prom*1000
X.Q23.unique()
for i in range(0,n):
    a=[float(s) for s in re.findall(r'-?\d+\.?\d*', X.Q23.iloc[i])]
    #re.findall finds the caracters that correspond to numbers in a string
    a=np.absolute(a)
    prom=np.mean(a)
    X1[i,4]=prom
for i in range (p):                 
        prom=np.mean(X1[:,i])
        desv=np.std(X1[:,i]) 
        X1[:,i]=(X1[:,i]-prom)/desv
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X1)
pca.explained_variance_ratio_
z=pca.fit_transform(X1) #matrix containing the projections over the components
componentes=pca.components_ 
# The components are displayed by rows and in order
#of fraction of exlained variance
componentes
vectors_c1c2=np.zeros((5,2))#array containing the first and second component for the five features
vectors_c1c2[:,0]=componentes[0,:]
vectors_c1c2[:,1]=componentes[1,:]
vectors_c1c3=np.zeros((5,2))#array containing the first and third component for the five features
vectors_c1c3[:,0]=componentes[0,:]
vectors_c1c3[:,1]=componentes[2,:]
vectors_c2c3=np.zeros((5,2))#array containing the second and third component for the five features
vectors_c2c3[:,0]=componentes[1,:]
vectors_c2c3[:,1]=componentes[2,:]
vectors_c1c2
import matplotlib.pyplot as plt
colors=np.array(['red','blue','orange','black','green'])
fig=plt.figure(figsize=(7,17))
ax=plt.subplot(3,1,1)
origin = [0], [0]
ax.scatter(z[:,0],z[:,1])
plt.xlabel("1st component")
plt.ylabel("2nd component")
#This generates the vectors corresponding to each feature in the corresponding components:
for i in range(vectors_c1c2.shape[0]):
    plt.quiver(*origin,vectors_c1c2[i,0],vectors_c1c2[i,1], scale=2.5,label=X.columns[i],color=colors[i])
plt.legend()
plt.subplot(3,1,2)
origin = [0], [0]
plt.scatter(z[:,0],z[:,2])
plt.xlabel("1st component")
plt.ylabel("3rd component")
#This generates the vectors corresponding to each feature in the corresponding components:
for i in range(vectors_c1c2.shape[0]):
    plt.quiver(*origin,vectors_c1c3[i,0],vectors_c1c3[i,1], scale=7.0,label=X.columns[i],color=colors[i])
plt.legend()
plt.subplot(3,1,3)
origin = [0], [0]
plt.scatter(z[:,1],z[:,2])
plt.xlabel("2nd component")
plt.ylabel("3rd component")
#This generates the vectors corresponding to each feature in the corresponding components:
for i in range(vectors_c1c2.shape[0]):
    plt.quiver(*origin,vectors_c2c3[i,0],vectors_c2c3[i,1], scale=5.0,label=X.columns[i],color=colors[i])
plt.legend()

print('Q23:'+multipleCh_init.Q23.iloc[0])
print('Q4:'+multipleCh_init.Q4.iloc[0])
def histo(nombres):  #nombres= variable for which we want to generate the histogram 
    a=np.unique(nombres) #gets all posible values for the variable
    cont=np.zeros(len(a)) # we are  going to count the # of apperances for every possible value
    l=list(nombres) # we transform to list, to use the function count
    for i in range(len(a)):           
        cont[i]= l.count(a[i])   
    mas=a[cont>5] # we will keep the ones that have more than 5 appearences in order to make
    #histograms better visualy (5 is totally arbitrary)
    nombres_mas=[]
    for i in range(len(nombres)):
        if(nombres[i] in mas):
            nombres_mas.append(nombres[i])
    h=plt.hist(nombres_mas,bins=len(mas),density=True)
    plt.xticks(rotation='vertical')
    
def graficas(variable):
    figura=plt.figure(figsize=(26,25))
    limits=[-2.5,-1.0,0.0,1.0,2.5]
    for i in range(len(limits)-1):
        var_corte=X[str(variable)][(z[:,1]>limits[i]) & (z[:,1]<limits[i+1])]
        var_corte1=np.array(var_corte)
        plt.subplot(4,1,i+1)
        h=histo(var_corte1)
        plt.title("["+str(limits[i])+","+str(limits[i+1])+"]") #on the top of every histogram
        #is shown the corresponding interval from the second component
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=2.5)
print('on the top of every histogram is shown the corresponding interval from the second component ') 
#is shown the corresponding interval from the second component )
print('Q3:'+multipleCh_init.Q3.iloc[0])
graficas('Q3')#on the top of every histogram 
#is shown the corresponding interval from the second component 
#(remenber, this component) has an important weight from the feature of question Q23
print('on the top of every histogram is shown the corresponding interval from the second component ') 
print('Q17:'+multipleCh_init.Q17.iloc[0])
graficas('Q17')#on the top of every histogram 
#is shown the corresponding interval from the second component
print('on the top of every histogram is shown the corresponding interval from the second component ') 
print('Q5:'+multipleCh_init.Q5.iloc[0])
graficas('Q5')#on the top of every histogram 
#is shown the corresponding interval from the second component

