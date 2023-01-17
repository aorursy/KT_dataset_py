#Importing important module of python such as pandas for importing data and for data analysis, numpy for numerical operation of mathematics, matplotlib for ploting graph and matplotlib inline to ploting graph witin jupyter notebook.



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
#Reading csv file using pandas :-

df = pd.read_csv('../input/musk-dataset/musk_csv.csv')
#Showing top five row of data:-

df.head()
#Describe the data for analysis.

df.describe().transpose()
#Now we see the elements in class variable.

df['class'].value_counts()
#Now we see number of unique values in each column of data

df.nunique()
#Now we take copy of data and named it copy_df.

copy_df=df.copy()
#Now we remove three categorical column which may not effect the class .

copy_df.drop(['molecule_name','ID','conformation_name'],axis=1,inplace=True)
#Showing two row of data.

copy_df.head(2)
#Now we see is there any null values in between data.

copy_df.isnull().values.any()
#Now we import train test split from sklearn for splitting data.

from sklearn.model_selection import train_test_split
#Now we plot boxplot to see outliers and data distribution visually with the help of seaborn.

import seaborn as sns

sns.boxplot(x=copy_df['f1'])
sns.boxplot(x=copy_df['f4'])
sns.boxplot(x=copy_df['f20'])
#Now we define function that remove outliers with median values.

def outlier_detect(df):

    for i in df.describe().columns:

        Q1=df.describe().at['25%',i]

        Q3=df.describe().at['75%',i]

        IQR=Q3 - Q1

        LTV=Q1 - 1.5 * IQR

        UTV=Q3 + 1.5 * IQR

        x=np.array(df[i])

        p=[]

        for j in x:

            if j < LTV or j>UTV:

                p.append(df[i].median())

            else:

                p.append(j)

        df[i]=p

    return df
#Now we define X,y value where X containing data with no class column and y with class column of data.

X = copy_df.drop('class', axis=1)

y = copy_df['class']
#Now we split the data in 80:20 ration using train test split.

X_train,X_test,Y_train,Y_test=train_test_split(X, y, test_size = 0.20,random_state=101)
#Now we remove first stage outliers from X_train data

corr_X_train= outlier_detect(X_train)
#Now we remove first stage outliers from X_test data.

corr_X_test= outlier_detect(X_test)
# Now we import keras module for build a model.

import keras

from keras.models import Model

from keras.layers import *
#We check the shape of corr_X_train and corr_X_test data.

print(corr_X_train.shape)

print(corr_X_test.shape)
#Importing Standard Scale from sklearn to convert Data in standard scaled data.

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(corr_X_train)

scaler.fit(corr_X_test)
X_train_scaled = scaler.transform(corr_X_train)

X_test_scaled = scaler.transform(corr_X_test)
# Importing Principal component Analysis.

from sklearn.decomposition import PCA
pca = PCA(n_components=10)
pca.fit(X_train_scaled)

pca.fit(X_test_scaled)
pca_X_train = pca.transform(X_train_scaled)

pca_X_test = pca.transform(X_test_scaled)
X_train_pca = pd.DataFrame(

    pca_X_train,

    columns=['PC1','PC2', 'PC3','PC4','PC5', 'PC6','PC7','PC8', 'PC9', 'PC10'])

X_train_pca.head()
X_test_pca = pd.DataFrame(

    pca_X_test,

    columns=['PC1','PC2', 'PC3','PC4','PC5', 'PC6','PC7','PC8', 'PC9', 'PC10'])

X_test_pca.head()
# We plot a pairplot of X_train_pca and see data distribution visually.

sns.pairplot(X_train_pca, diag_kind='kde')
#Build the Model.

Inp=Input(shape=(10,))

x=Dense(400,activation='sigmoid',name='Hidden_layer1')(Inp)

x=Dense(300,activation='relu',name='Hidden_layer2')(x)

x=Dense(200,activation='relu',name='Hidden_layer3')(x)

x=Dense(100,activation='relu',name='Hidden_layer4')(x)

x=Dense(80,activation='relu',name='Hidden_layer5')(x)

output=Dense(1,activation='sigmoid',name='Output_layer')(x)
#Now we Define the object of model and summary of model.

model=Model(Inp,output)

model.summary()
#Importing optimizer

from keras import optimizers
#define the hypermeters for model such as learning rate, epoch, batch_size etc.

l_rate=0.0001

training_epoch=50

batch_size=700

adma=optimizers.adam(lr=l_rate)
#Now we define compile method of method where we use Adam optimizer.

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
#Now we fit the model with X_train_pca , Y_train with  validation sets of X_test_pca, Y_test. 

opt =model.fit(X_train_pca,Y_train,batch_size=batch_size,epochs=training_epoch,verbose=2,validation_data=(X_test_pca,Y_test))

# Now we fetch important keys of our model.

print(opt.history.keys())
plt.plot(opt.history['loss'],label='train')

plt.xlabel('epochs')

plt.plot(opt.history['val_loss'],label='test')

plt.ylabel('loss')

plt.legend()

plt.show()
plt.plot(opt.history['accuracy'],label='train')

plt.plot(opt.history['val_accuracy'],label='test')

plt.xlabel('epochs')

plt.ylabel('accuracy')

plt.legend()

plt.show()

#Now we predict the value for X_test_pca.

Y_pred=model.predict(X_test_pca).astype('int').flatten()

print(Y_pred)
#Now we import both classification report and confusion matrix.

from sklearn.metrics import classification_report, confusion_matrix

cls = classification_report(Y_test,Y_pred)

cls_1 = confusion_matrix(Y_test,Y_pred)

print(cls)

print(cls_1)
#Now we save our model.

model.save("model.h5")
# Importing Kmean clustering from sklearn.

from sklearn.cluster import KMeans
#We find the clusters error w.r.t number of cluster.

cluster_range=range(2,6)

cluster_errors=[]

for num_clusters in cluster_range:

    clusters=KMeans(num_clusters,n_init=5)

    clusters.fit(X_train)

    labels=clusters.labels_

    centroids=clusters.cluster_centers_

    cluster_errors.append(clusters.inertia_)

clusters_df=pd.DataFrame({'num_cluster':cluster_range, 'cluster_errors': cluster_errors})

clusters_df[0:15]
#We plot graph b/w clusters error and number of cluster.

plt.figure(figsize=(12,6))

plt.plot(clusters_df.num_cluster, clusters_df.cluster_errors, marker='o')