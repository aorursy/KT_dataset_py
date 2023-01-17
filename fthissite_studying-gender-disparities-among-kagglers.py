import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
data=pd.read_csv('../input/multipleChoiceResponses.csv',engine='python')
mkeys=data.keys()
#eliminamos las columnas con respuestas personales que no se pueden categorizar
data=data.drop([mkeys[2], mkeys[8], mkeys[10], mkeys[44], mkeys[56], mkeys[64], mkeys[83], mkeys[85], mkeys[87], mkeys[107], mkeys[109], mkeys[123],mkeys[125], mkeys[150], mkeys[194], mkeys[223], mkeys[249],mkeys[262], mkeys[264],mkeys[276], mkeys[304],mkeys[306], mkeys[329], mkeys[341], mkeys[371], mkeys[385], mkeys[394]],axis=1)
data.drop([data.index[0]],inplace=True)
mkeys=data.keys()
data=data.fillna(-1)
data.head()
males=data.loc[data['Q1']=='Male']
females=data.loc[data['Q1']=='Female']
other=data.loc[data['Q1']=='Prefer to self-describe']
males=males.drop([mkeys[1]],axis=1)
females=females.drop([mkeys[1]],axis=1)
other=other.drop([mkeys[1]],axis=1)
gkeys=other.keys() #llaves para los df de cada genero
Co_m=males['Q3'].values
Co_m_n=males['Q3'].value_counts()

Co_f=females['Q3'].values
Co_f_n=females['Q3'].value_counts()

Co_o=other['Q3'].values
Co_o_n=other['Q3'].value_counts()
Co_f_n[:10].plot.barh(x='Female',legend=False)
Co_o_n[:10].plot.barh()
Co_m_n[:10].plot.barh()
grouped_m=males.groupby(['Q3','Q4']).size()
plt.figure(figsize=(25,20))
plt.subplot(5,2,1)
grouped_m=grouped_m.rename(index={'Some college/university study without earning a bachelorâ€™s degree': 'Unfinished studies'})
grouped_m=grouped_m.rename(index={'No formal education past high school': 'High school'})
grouped_m=grouped_m.rename(index={'Masterâ€™s degree': 'MSc'})
grouped_m=grouped_m.rename(index={'Bachelorâ€™s degree': 'Bachelor'})
plt.title('India')
grouped_m.loc['India'][1:].plot.barh()
plt.subplot(5,2,2)
plt.title('United States of America')
grouped_m.loc['United States of America'][1:].plot.barh()
plt.subplot(5,2,3)
plt.title('China')
grouped_m.loc['China'][1:].plot.barh()
plt.subplot(5,2,4)
plt.title('Other')
grouped_m.loc['Other'][1:].plot.barh()
plt.subplot(5,2,5)
plt.title('Russia')
grouped_m.loc['Russia'][1:].plot.barh()
plt.subplot(5,2,6)
plt.title('Brazil')
grouped_m.loc['Brazil'][1:].plot.barh()
grouped_f=females.groupby(['Q3','Q4']).size()

plt.figure(figsize=(25,20))
grouped_f=grouped_f.rename(index={'Some college/university study without earning a bachelorâ€™s degree': 'Unfinished studies'})
grouped_f=grouped_f.rename(index={'No formal education past high school': 'High school'})
grouped_f=grouped_f.rename(index={'Masterâ€™s degree': 'MSc'})
grouped_f=grouped_f.rename(index={'Bachelorâ€™s degree': 'Bachelor'})

plt.subplot(5,2,1)
plt.title('United States of America')
grouped_f.loc['United States of America'][1:].plot.barh()
plt.subplot(5,2,2)
plt.title('India')
grouped_f.loc['India'][1:].plot.barh()
plt.subplot(5,2,3)
plt.title('China')
grouped_f.loc['China'][1:].plot.barh()
plt.subplot(5,2,4)
plt.title('Other')
grouped_f.loc['Other'][1:].plot.barh()
plt.subplot(5,2,5)
plt.title('United Kingdom of Great Britain and Northern Ireland')
grouped_f.loc['United Kingdom of Great Britain and Northern Ireland'][1:].plot.barh()
plt.subplot(5,2,6)
plt.title('I do not wish to disclose my location')
grouped_f.loc['I do not wish to disclose my location'][1:].plot.barh()
grouped_o=other.groupby(['Q3','Q4']).size()

plt.figure(figsize=(25,20))
grouped_o=grouped_o.rename(index={'Some college/university study without earning a bachelorâ€™s degree': 'Unfinished studies'})
grouped_o=grouped_o.rename(index={'No formal education past high school': 'High school'})
grouped_o=grouped_o.rename(index={'Masterâ€™s degree': 'MSc'})
grouped_o=grouped_o.rename(index={'Bachelorâ€™s degree': 'Bachelor'})

plt.subplot(5,2,1)
plt.title('United States of America')
grouped_o.loc['United States of America'][1:].plot.barh()
plt.subplot(5,2,2)
plt.title('Russia')
grouped_o.loc['Russia'][1:].plot.barh()
plt.subplot(5,2,3)
plt.title('I do not wish to disclose my location')
grouped_o.loc['I do not wish to disclose my location'][1:].plot.barh()

plt.subplot(5,2,4)
plt.title('United Kingdom of Great Britain and Northern Ireland')
grouped_o.loc['United Kingdom of Great Britain and Northern Ireland'][1:].plot.barh()
plt.subplot(5,2,5)
plt.title('Other')
grouped_o.loc['Other'][1:].plot.barh()
plt.subplot(5,2,6)
plt.title('India')
grouped_o.loc['India'][1:].plot.barh()
objetos=data.select_dtypes(include=['object']).copy()
objetos=objetos.drop(mkeys[0],axis=1)
objetos=pd.get_dummies(objetos)
enc_data=pd.concat([data.iloc[:,0],objetos,data.select_dtypes(exclude=['object']).copy()],axis=1)
mkeys=enc_data.keys()
col0=enc_data[[mkeys[0]]].values.astype(np.float)
mms=MinMaxScaler()
col_fit=mms.fit_transform(col0)
enc_data[mkeys[0]] = pd.DataFrame(col_fit).fillna(0)
enc_data.head()
pca=PCA(n_components=10)
enc_data[mkeys[0]] = enc_data[mkeys[0]].fillna(0)
pca.fit(enc_data)
var_exp=0.0
i=0
var=pca.explained_variance_
var_s=sum(var)
sigma_arr=[]
while (var_exp<0.8):
    var_exp+=var[i]/var_s
    i+=1
print ('se llega al 80% de varianza con los primeros '+str(i)+' componentes')
pc_=pca.transform(enc_data)
Ks = range(1, 10)
km = [KMeans(n_clusters=i) for i in Ks]
score = [km[i].fit(pc_).score(pc_) for i in range(len(km))]
inertia=[km[i].inertia_ for i in range(len(km))]
fig1, ax=plt.subplots(1,2, figsize=(12,5))
ax[0].scatter(Ks,score,marker='x')
ax[0].plot(Ks,score)
ax[0].scatter(Ks[2],score[2],color='r',marker='o')
ax[0].set_xlabel('$K$')
ax[0].set_ylabel('Score')

ax[1].scatter(Ks,inertia,marker='x')
ax[1].plot(Ks,inertia)
ax[1].scatter(Ks[2],inertia[2],color='r',marker='o')
ax[1].set_xlabel('$K$')
ax[1].set_ylabel('Within-cluster variation')
K_opt=km[2] #el Kmeans optimo
cluster_i=K_opt.fit_predict(pc_) #indices de los clusters
for i in range(0,3):
    plt.scatter(np.asarray(pc_)[cluster_i == i, 0], np.asarray(pc_)[cluster_i == i,1], alpha=0.7)

plt.scatter(K_opt.cluster_centers_[:,0], K_opt.cluster_centers_[:,1],marker='x',s=100,c='black')#centroides