import pandas as pd

import numpy as np

import sklearn as skl

import seaborn as sns

import matplotlib.pyplot as plt

df=pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')

dftest=pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')
df
df.columns
# df2['Channel1']=df2['Channel1'].str.replace("No tv connection", "No")

# df2['Channel2']=df2['Channel2'].str.replace("No tv connection", "No")

# df2['Channel3']=df2['Channel3'].str.replace("No tv connection", "No")

# df2['Channel4']=df2['Channel4'].str.replace("No tv connection", "No")

# df2['Channel5']=df2['Channel5'].str.replace("No tv connection", "No")

# df2['Channel6']=df2['Channel6'].str.replace("No tv connection", "No")



# dftest['Channel1']=dftest['Channel1'].str.replace("No tv connection", "No")

# dftest['Channel2']=dftest['Channel2'].str.replace("No tv connection", "No")

# dftest['Channel3']=dftest['Channel3'].str.replace("No tv connection", "No")

# dftest['Channel4']=dftest['Channel4'].str.replace("No tv connection", "No")

# dftest['Channel5']=dftest['Channel5'].str.replace("No tv connection", "No")

# dftest['Channel6']=dftest['Channel6'].str.replace("No tv connection", "No")
dftest

df2=df
# df3 = pd.DataFrame()   



# gender_dummies=pd.get_dummies(df2.gender)

# df3=pd.concat([df2, gender_dummies], axis=1)



# Married_dummies=pd.get_dummies(df2.Married)

# df3=pd.concat([df2, Married_dummies], axis=1)

# Import label encoder 

from sklearn import preprocessing 

  

# label_encoder object knows how to understand word labels. 

label_encoder = preprocessing.LabelEncoder() 

  

# Encode labels in column 'species'. 

df2['gender']= label_encoder.fit_transform(df2['gender'])

df2['Married']= label_encoder.fit_transform(df2['Married'])

df2['Children']= label_encoder.fit_transform(df2['Children'])

# df2['TVConnection']= label_encoder.fit_transform(df2['TVConnection'])

df2['Channel1']= label_encoder.fit_transform(df2['Channel1'])

df2['Channel2']= label_encoder.fit_transform(df2['Channel2'])

df2['Channel3']= label_encoder.fit_transform(df2['Channel3'])

df2['Channel4']= label_encoder.fit_transform(df2['Channel4'])

df2['Channel5']= label_encoder.fit_transform(df2['Channel5'])

df2['Channel6']= label_encoder.fit_transform(df2['Channel6'])

df2['Internet']= label_encoder.fit_transform(df2['Internet'])

df2['HighSpeed']= label_encoder.fit_transform(df2['HighSpeed'])

df2['AddedServices']= label_encoder.fit_transform(df2['AddedServices'])

# df2['Subscription']= label_encoder.fit_transform(df2['Subscription'])

# df2['PaymentMethod']= label_encoder.fit_transform(df2['PaymentMethod'])



dftest['gender']= label_encoder.fit_transform(dftest['gender'])

dftest['Married']= label_encoder.fit_transform(dftest['Married'])

dftest['Children']= label_encoder.fit_transform(dftest['Children'])

# dftest['TVConnection']= label_encoder.fit_transform(dftest['TVConnection'])

dftest['Channel1']= label_encoder.fit_transform(dftest['Channel1'])

dftest['Channel2']= label_encoder.fit_transform(dftest['Channel2'])

dftest['Channel3']= label_encoder.fit_transform(dftest['Channel3'])

dftest['Channel4']= label_encoder.fit_transform(dftest['Channel4'])

dftest['Channel5']= label_encoder.fit_transform(dftest['Channel5'])

dftest['Channel6']= label_encoder.fit_transform(dftest['Channel6'])

dftest['Internet']= label_encoder.fit_transform(dftest['Internet'])

dftest['HighSpeed']= label_encoder.fit_transform(dftest['HighSpeed'])

dftest['AddedServices']= label_encoder.fit_transform(dftest['AddedServices'])

# dftest['Subscription']= label_encoder.fit_transform(dftest['Subscription'])

# dftest['PaymentMethod']= label_encoder.fit_transform(dftest['PaymentMethod'])
t=pd.get_dummies(df2['TVConnection'], columns=['Cable', 'DTH', 'No_tv'])

df2['Cable']=t['Cable']

df2['DTH']=t['DTH']

df2['NO_TV']=t['No']

df2=df2.drop('TVConnection', axis=1)

df2



t=pd.get_dummies(dftest['TVConnection'], columns=['Cable', 'DTH', 'No_tv'])

dftest['Cable']=t['Cable']

dftest['DTH']=t['DTH']

dftest['NO_TV']=t['No']

dftest=dftest.drop('TVConnection', axis=1)

dftest
t=pd.get_dummies(df2['Subscription'])

t

df2['Annually']=t['Annually']

df2['Biannually']=t['Biannually']

df2['Monthly']=t['Monthly']

df2=df2.drop('Subscription', axis=1)

df2



t=pd.get_dummies(dftest['Subscription'])

t

dftest['Annually']=t['Annually']

dftest['Biannually']=t['Biannually']

dftest['Monthly']=t['Monthly']

dftest=dftest.drop('Subscription', axis=1)

dftest
t=pd.get_dummies(df2['PaymentMethod'])

t

df2['Bank transfer']=t['Bank transfer']

df2['Cash']=t['Cash']

df2['Credit card']=t['Credit card']

df2['Net Banking']=t['Net Banking']



df2=df2.drop('PaymentMethod', axis=1)

df2



t=pd.get_dummies(dftest['PaymentMethod'])

t

dftest['Bank transfer']=t['Bank transfer']

dftest['Cash']=t['Cash']

dftest['Credit card']=t['Credit card']

dftest['Net Banking']=t['Net Banking']



dftest=dftest.drop('PaymentMethod', axis=1)

dftest
# df=pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')

# df.drop(['TotalCharges'], axis=1)

# df=pd.get_dummies(df)

# corr = df.corr()



# # Generate a mask for the upper triangle

# mask = np.zeros_like(corr, dtype=np.bool)

# mask[np.triu_indices_from(mask)] = True



# # Set up the matplotlib figure

# f, ax = plt.subplots(figsize=(15, 20))



# # Generate a custom diverging colormap

# cmap = sns.diverging_palette(250, 20, as_cmap=True)



# # Draw the heatmap with the mask and correct aspect ratio

# sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

#             square=True, linewidths=.5, cbar_kws={"shrink": .5})



# plt.show()
df2['TotalCharges']
df[df['TotalCharges'].str.replace(".", "").str.isdigit()==False]
df2[df2['TotalCharges'].str.replace(".", "").str.isdigit()==False]

dftest[dftest['TotalCharges'].str.replace(".", "").str.isdigit()==False]
dftest.loc[71, 'TotalCharges'] = 2298.24

dftest.loc[580, 'TotalCharges'] = 2298.24

dftest.loc[637, 'TotalCharges'] = 2298.24

dftest.loc[790, 'TotalCharges'] = 2298.24

dftest.loc[1505, 'TotalCharges'] = 2298.24
dftest[dftest['TotalCharges'].str.replace(".", "").str.isdigit()==False]
#### df2[df2['TotalCharges'].str.replace(".", "").str.isdigit()==False]

df2=df2[df2.custId != 951]

df2=df2[df2.custId != 2066]

df2=df2[df2.custId != 3095]

df2=df2[df2.custId != 6770]

df2=df2[df2.custId != 3069]

df2=df2[df2.custId != 2340]

df2[df2['TotalCharges'].str.replace(".", "").str.isdigit()==False]



# dftest=dftest[dftest.custId != 4611]

# dftest=dftest[dftest.custId != 4416]

# dftest=dftest[dftest.custId != 1101]

# dftest=dftest[dftest.custId != 1146]

# dftest=dftest[dftest.custId != 6315]



df2[df2['TotalCharges'].str.replace(".", "").str.isdigit()==False]

dftest[dftest['TotalCharges'].str.replace(".", "").str.isdigit()==False]
df2.info()

dftest.info()
df2['TotalCharges']=df2.TotalCharges.astype(float)

dftest['TotalCharges']=dftest.TotalCharges.astype(float)
# df2.info()

# dftest.info()

df.columns
df3train=pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')

df3train=df3train.drop(['custId','gender', 'Subscription', 'PaymentMethod', 'MonthlyCharges',

                     'TVConnection', 'Channel2', 'Channel1', 'SeniorCitizen'], axis=1)



df3train.loc[544, 'TotalCharges'] = 2276.90

df3train.loc[1348, 'TotalCharges'] = 2276.90

df3train.loc[1553, 'TotalCharges'] = 2276.90

df3train.loc[2504, 'TotalCharges'] = 2276.90

df3train.loc[3083, 'TotalCharges'] = 2276.90

df3train.loc[4766, 'TotalCharges'] = 2276.90



df3train['TotalCharges']=df3train['TotalCharges'].astype(float)

df3train=pd.get_dummies(df3train)

df3train=df3train.drop([ 'Married_No', 'Children_No',

                       'Channel3_No tv connection', 'Channel4_No tv connection',

                       'Channel5_No tv connection', 'Channel6_No tv connection', 'AddedServices_No'], axis=1)

# df3train=df3test.drop(['custId', 'gender', 'Internet', 'HighSpeed'], axis=1)



print('starting')

df3test=pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')

df3test=df3test.drop(['custId', 'gender', 'Subscription', 'PaymentMethod', 'MonthlyCharges',

                     'TVConnection', 'Channel2', 'Channel1', 'SeniorCitizen'], axis=1)



df3test.loc[71, 'TotalCharges'] = 2298.24

df3test.loc[580, 'TotalCharges'] = 2298.24

df3test.loc[637, 'TotalCharges'] = 2298.24

df3test.loc[790, 'TotalCharges'] = 2298.24

df3test.loc[1505, 'TotalCharges'] = 2298.24

df3test['TotalCharges']=df3test['TotalCharges'].astype(float)

df3test=pd.get_dummies(df3test)



df3test=df3test.drop(['Married_No', 'Children_No', 

                     'Channel3_No tv connection', 'Channel4_No tv connection',

                       'Channel5_No tv connection', 'Channel6_No tv connection', 'AddedServices_No'], axis=1)
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)

X_principal = pca.fit_transform(df3train) 

X_principal = pd.DataFrame(X_principal) 

X_principal.columns = ['P1', 'P2'] 
X_principal
# df3
# df2
x=df3train.drop('Satisfied', axis=1)

finalDf = pd.concat([X_principal, df3train[['Satisfied']]], axis = 1)
finalDf
fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 component PCA', fontsize = 20)

targets = [1, 0]

colors = ['r', 'g']

for target, color in zip(targets,colors):

    indicesToKeep = finalDf['Satisfied'] == target

    ax.scatter(finalDf.loc[indicesToKeep, 'P1']

               , finalDf.loc[indicesToKeep, 'P2']

               , c = color

               , s = 50)

ax.legend(targets)

ax.grid()
from sklearn.cluster import SpectralClustering

spectral_model_nn = SpectralClustering(n_clusters = 2, affinity ='nearest_neighbors')
# labels_nn = spectral_model_nn.fit_predict(X_principal)
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 2000, n_init = 10, random_state = 0)

y_kmeans = kmeans.fit_predict(df3train)
y_kmeans.sum()
df2['Satisfied'].sum()
# sklearn.cluster.SpectralClustering(n_clusters=8, eigen_solver=None, random_state=None, n_init=10, gamma=1.0, affinity=’rbf’, n_neighbors=10, eigen_tol=0.0, assign_labels=’kmeans’, degree=3, coef0=1, kernel_params=None, n_jobs=None)

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(affinity='euclidean', compute_full_tree=True,

                        connectivity=None, distance_threshold=None,

                        linkage='ward', memory=None, n_clusters=2,

                        pooling_func='deprecated')

y_pred=cluster.fit_predict(df3test)
# df3test=df3test.drop(['Internet', 'HighSpeed', 'Channel2', 'gender', 'MonthlyCharges'], axis=1)
# df4train.columns
df3train=pd.read_csv('/kaggle/input/eval-lab-3-f464/train.csv')

# df3train=df3train.drop(['custId','Internet', 'HighSpeed', 'PaymentMethod'], axis=1)



features=['SeniorCitizen','tenure', 'MonthlyCharges', 'Children', 'Married', 'Channel5', 'Channel6',

                   'AddedServices', 'Subscription']

# df3train=df3train[features]



df3train.loc[544, 'TotalCharges'] = 2276.90

df3train.loc[1348, 'TotalCharges'] = 2276.90

df3train.loc[1553, 'TotalCharges'] = 2276.90

df3train.loc[2504, 'TotalCharges'] = 2276.90

df3train.loc[3083, 'TotalCharges'] = 2276.90

df3train.loc[4766, 'TotalCharges'] = 2276.90

df3train['TotalCharges']=df3train['TotalCharges'].astype(float)

df3train=pd.get_dummies(df3train)

# df3train=df3train.drop([ 'Married_No', 'Children_No',

#                        'Channel3_No tv connection', 'Channel4_No tv connection',

#                        'Channel5_No tv connection', 'Channel6_No tv connection', 'AddedServices_No'], axis=1)

# df3train=df3test.drop(['custId', 'gender', 'Internet', 'HighSpeed'], axis=1)



print('starting')

df3test=pd.read_csv('/kaggle/input/eval-lab-3-f464/test.csv')

# df3test=df3test.drop(['custId','Internet', 'HighSpeed', 'PaymentMethod'], axis=1)

df3test=df3test[features]



df3test.loc[71, 'TotalCharges'] = 2298.24

df3test.loc[580, 'TotalCharges'] = 2298.24

df3test.loc[637, 'TotalCharges'] = 2298.24

df3test.loc[790, 'TotalCharges'] = 2298.24

df3test.loc[1505, 'TotalCharges'] = 2298.24

df3test['TotalCharges']=df3test['TotalCharges'].astype(float)

df3test=pd.get_dummies(df3test)



# df3test=df3test.drop(['Married_No', 'Children_No', 

#                      'Channel3_No tv connection', 'Channel4_No tv connection',

#                        'Channel5_No tv connection', 'Channel6_No tv connection', 'AddedServices_No'], axis=1)
corr = df3train.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(20, 20))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(250, 30, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



plt.show()
X=df3train.drop(['Satisfied'], axis=1)
from sklearn.cluster import KMeans

num=25

thresh=17

kmeans = KMeans(n_clusters = num, init = 'k-means++', max_iter = 2000, n_init = 6, random_state = 10)

y_pred = kmeans.fit_predict(X)

y_pred=pd.DataFrame(data=y_pred)



answer=pd.concat([y_pred, df3train['Satisfied']], axis=1)

answer.columns=['col1', 'col2']

arr=[]

for j in range(num):

    sum1=0

    sum2=0

    for i in range(len(answer)):

        if(answer['col1'][i]==j):

            if(answer['col2'][i]==1):

                sum1+=1

            else:

                sum2+=1

    tmp=(sum2/(sum1+sum2))*100

    print(j, tmp)

    if(tmp>=thresh):

        arr.append(j)
# from sklearn.cluster import KMeans

# kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 2000, n_init = 10, random_state = 10)

# y_pred = kmeans.fit_predict(df3test)



# for i in range(len(y_pred)):

#     if(y_pred[i] in arr):

#         y_pred[i]=0

#     else:

#         y_pred[i]=1

# y_pred.sum()
y_pred=pd.DataFrame(data=y_pred)

answer=pd.concat([dftest['custId'], y_pred], axis=1)

answer.columns=['custId', 'Satisfied']



answer.to_csv('answer_eval_lab_3.csv', index=False)

answer
answer['Satisfied'].sum()