import pandas as pd
import numpy as np
import matplotlib.pylab as plt
dat_1=pd.read_csv('../input/havingfun/current-data-on-the-geographic-distribution-of-covid-19-cases-worldwide.csv')
dat_2=pd.read_csv('../input/havingfun/coronavirus-world-airport-impacts.csv')
dat_3=pd.read_csv('../input/havingfun/acaps-covid-19-government-measures-dataset.csv')
dat_4=pd.read_csv('../input/havingfun/global-school-closures-covid-19.csv')
dat_5=pd.read_csv('../input/havingfun/regional-mobility.csv')
print(dat_1.keys())
print(dat_2.keys())
print(dat_3.keys())
print(dat_4.keys())
print(dat_5.keys())
C=list(set(dat_1['countriesandterritories']))
len(C)
j=np.where(dat_1['countriesandterritories']=='Colombia')
death = dat_1['deaths'][j[0]]
cases = dat_1['cases'][j[0]]
n=len(j[0])
y=abs(np.arange(n))
plt.figure(figsize=(10,5))
plt.subplot(121)
plt.title('Deaths')
plt.scatter(y,np.cumsum(np.array(death)),s=2,label='Colombia')
plt.legend(loc=0.0)
plt.xlabel('Number of days')
plt.subplot(122)
plt.title('Cases')
plt.scatter(y,np.cumsum(np.array(cases)),s=2,label='Colombia')
plt.legend(loc=0.0)
plt.xlabel('Number of days')
C
plt.figure(figsize=(14,5))
C1=C[:5]+['Colombia']
plt.subplot(121)
for cpa in C1:
    j=np.where(dat_1['countriesandterritories']==cpa)
    death = dat_1['deaths'][j[0]]
    cases = dat_1['cases'][j[0]]
    n=len(j[0])
    y=abs(np.arange(n))
    plt.scatter(y,np.cumsum(np.array(death)),s=2,label=cpa)
plt.title('Deaths')
plt.legend(loc=0.0)
plt.xlabel('Number of days')
plt.subplot(122)
plt.title('Cases')
for cpa in C1:
    j=np.where(dat_1['countriesandterritories']==cpa)
    death = dat_1['deaths'][j[0]]
    cases = dat_1['cases'][j[0]]
    n=len(j[0])
    y=abs(np.arange(n))
    plt.scatter(y,np.cumsum(np.array(cases)),s=2,label=cpa)
plt.legend(loc=0.0)
plt.xlabel('Number of days')
plt.figure(figsize=(18,5))
plt.subplot(121)
D=['Argentina','United Kingdom','Colombia','Italy','China','Germany','United_States_of_America']
for cpa in D:
    j=np.where(dat_1['countriesandterritories']==cpa)
    death = dat_1['deaths'][j[0]]
    cases = dat_1['cases'][j[0]]
    n=len(j[0])
    y=abs(np.arange(n))
    plt.scatter(y,np.cumsum(np.array(death)),s=2,label=cpa)
plt.legend(loc=0.0)
plt.title('Deaths')
plt.xlabel('Number of days')
plt.subplot(122)
plt.title('Cases')
for cpa in D:
    j=np.where(dat_1['countriesandterritories']==cpa)
    death = dat_1['deaths'][j[0]]
    cases = dat_1['cases'][j[0]]
    n=len(j[0])
    y=abs(np.arange(n))
    plt.scatter(y,np.cumsum(np.array(cases)),s=2,label=cpa)
plt.xlabel('Number of days')
plt.legend(loc=0.0)
A=set(dat_2['continent'])
plt.figure(figsize=(10,6))
for i in A:
    k = np.where(dat_2['continent']==i)
    plt.scatter(dat_2['longitude'][k[0]],dat_2['latitude_d'][k[0]],s=0.2,label=i)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Airports around the world')
Country=list(set(dat_1['countriesandterritories']))
#2 letras
D1_2l=list(set(dat_1['geoid']))
#3 letras
D1_3l=list(set(dat_1['countryterritorycode']))

#2 letras
D2=list(set(dat_2['iso_countr']))[1:]

#3 letras
D3=list(set(dat_3['iso']))

#3 letras
D4=list(set(dat_4['iso']))

#Country
D5=list(set(dat_5['country']))
Country_C=[]
for i in D5:
    if i in Country:
        Country_C.append(i)
len(Country_C)
Iso_2=[]
for i in D1_2l:
    if i in D2:
        Iso_2.append(i)
len(Iso_2)
Iso_3=[]
for i in D1_3l:
    if (i in D3)&(i in D4):
        Iso_3.append(i)
len(Iso_3)
def contin(Name):
    x=0
    if(Name=='Europe'):
        x=1
    elif(Name=='Asia'):
        x=2
    elif(Name=='Africa'):
        x=3
    elif(Name=='Oceania'):
        x=4
    
    return x
Country_list=[]
target=[]
for i in Country_C:
    ii=np.where(dat_1['countriesandterritories']==i)[0][0]
    l2=dat_1['geoid'][ii]
    l3=dat_1['countryterritorycode'][ii]
    continent=dat_1['continentexp'][ii]
    con=contin(continent)
    if(l2 in Iso_2)&(l3 in Iso_3):
        Country_list.append([i,l2,l3,con])
        target.append(con)
print(len(target),len(Country_list))
n=len(Country_list)
data=np.zeros((n,18))
dat_6=np.array(dat_5)
#Funciones
def data1(name):
    j=np.where(dat_1['countriesandterritories']==name)[0]
    casos=np.sum(dat_1['cases'][j])
    muertes=np.sum(dat_1['deaths'][j])
    
    return [muertes,casos]

def data2(l2):
    j=np.where(dat_2['iso_countr']==l2)[0]
    cant=len(j)
    a=len(dat_2['type'][j])
    b=len(np.where(dat_2['type'][j]=='large_airport'))
    
    return [cant,b/a]

def data34(l3):
    j=np.where(dat_3['iso']==l3)[0]
    k=np.where(dat_4['iso']==l3)[0]
    
    return [len(j),len(k)]

def data5(name):
    b=np.where((dat_5['country']==name) & (dat_5['region']=='Total'))[0]
    c=int(len(b)/4)
    A=list(np.mean(dat_6[b,3:][:c],axis=0))+list(np.mean(dat_6[b,3:][-c:],axis=0))
    
    return A

def data_obt(name,l2,l3):
    return data1(name)+data2(l2)+data34(l3)+data5(name)
for i in range(len(Country_list)):
    da=Country_list[i]
    data[i,:]=data_obt(da[0],da[1],da[2])
print(np.sum(np.isnan(data)))
Warn=[]
for i in range(len(Country_list)):
    na=np.sum(np.isnan(data[i,:]))
    if na!=0:
        Warn.append(i)
        print(i,Country_list[i][0])
Warn=np.array(Warn)
data=np.delete(data, Warn, axis=0)
Country_list=np.delete(Country_list,Warn,axis=0)
target=np.delete(target,Warn)
print(np.sum(np.isnan(data)))
print(np.shape(data),np.shape(target))
Country_list
import sklearn.ensemble # para el random forest
import sklearn.model_selection # para split train-test
import sklearn.metrics # para calcular el f1-score
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(data, target, test_size=0.5)
predictors=['cases','deaths','Number of Airports','Medium size of airports','Number of Measures','Schools closed']
predictors=predictors+list(dat_5.keys()[3:])
for i in dat_5.keys()[3:]:
    j=i+' end'
    predictors=predictors+[j]
n_trees = np.arange(1,400,25)
f1_train = []
f1_test = []
feature_importance = np.zeros((len(n_trees), len(predictors)))

for i, n_tree in enumerate(n_trees):
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_tree, max_features='sqrt')
    clf.fit(x_train, y_train)
    f1_train.append(sklearn.metrics.f1_score(y_train, clf.predict(x_train),average='macro'))
    f1_test.append(sklearn.metrics.f1_score(y_test, clf.predict(x_test),average='macro'))
    feature_importance[i, :] = clf.feature_importances_
plt.scatter(n_trees,f1_test)
plt.ylim(0.3,1)
plt.xlabel('n_trees')
plt.ylabel('F1 Score')
# Grafica los features mas importantes
avg_importance = np.average(feature_importance, axis=0)
a = pd.Series(avg_importance, index=predictors)
a.nlargest().plot(kind='barh')
plt.xlabel('Average Feature Importance')
def stat(data,avg):
    rt=np.zeros(len(avg))
    for i in range(len(avg)):
        rt[i]=np.mean(data[:,avg[i]])
    return rt
avg=[6,0,14,8,5]
avg_data=np.zeros((5,5))
for i in range(5):
    n_1=np.where(target==i)[0]
    avg_data[i,:]=stat(data[n_1,:],avg)
name_con=['America','Europe','Asia','Africa','Oceania']
plt.figure(figsize=(26,6))
col=['blue','orange','green','red','purple']
for i in range(5):
    k=150+i+1
    plt.subplot(k)
    plt.scatter(name_con,avg_data[:,i],label=predictors[avg[i]],c=col[i])
    plt.legend(loc=0.0)    
    plt.xlabel('Continents')
    if i==1:
        plt.ylabel('Number of cases')
    elif i==4:
        plt.ylabel('Number of Schools Closed')
    else:
        plt.ylabel('Mobility')
import sklearn.cluster
tsne = sklearn.manifold.TSNE(perplexity=20)
tsne.fit(data)
embedding = tsne.embedding_
plt.scatter(embedding[:,0], embedding[:,1], c=target, cmap='Paired', s=1.0)
plt.colorbar(boundaries=np.arange(6)-0.5).set_ticks(np.arange(5))
print(name_con)
# clusters sobre los resultados de tsne
n_clusters = 5
k_means = sklearn.cluster.KMeans(n_clusters=n_clusters)
k_means.fit(embedding)
cluster = k_means.predict(embedding) 
distance = k_means.transform(embedding) 

plt.scatter(embedding[:,0], embedding[:,1], c=cluster, cmap='Paired', s=1.0)
plt.colorbar(boundaries=np.arange(6)-0.5).set_ticks(np.arange(5))
# clusters sobre los resultados de tsne
plt.figure(figsize=(12,6))
n_clusters = 5
k_means = sklearn.cluster.KMeans(n_clusters=n_clusters)
k_means.fit(embedding)
cluster = k_means.predict(embedding) 
distance = k_means.transform(embedding) 

plt.scatter(embedding[:,0], embedding[:,1], c=cluster, cmap='Paired', s=1.0)
plt.colorbar(boundaries=np.arange(6)-0.5).set_ticks(np.arange(5))
i_1=np.where(target==0)[0]
plt.scatter(embedding[i_1,0], embedding[i_1,1], c='red', cmap='Paired', s=2.0)
for i in i_1:
    if np.array(Country_list)[i,2]=='COL':
        plt.text(embedding[i,0]+0.2, embedding[i,1]-0.4, np.array(Country_list)[i,2],color='red',fontsize=12)
    else:
        plt.text(embedding[i,0]+0.2, embedding[i,1]+0.2, np.array(Country_list)[i,2])
np.array(Country_list)[i_1,2]
