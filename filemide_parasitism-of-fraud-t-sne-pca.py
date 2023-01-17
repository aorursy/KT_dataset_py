!nvidia-smi
!cat /usr/local/cuda/version.txt


## Passing Y as input while conda asks for confirmation, we use yes command

!yes Y | conda install faiss-gpu cudatoolkit=10.0 -c pytorch
!wget https://anaconda.org/CannyLab/tsnecuda/2.1.0/download/linux-64/tsnecuda-2.1.0-cuda100.tar.bz2

!tar xvjf tsnecuda-2.1.0-cuda100.tar.bz2 --wildcards 'lib/*'

!tar xvjf tsnecuda-2.1.0-cuda100.tar.bz2 --wildcards 'site-packages/*'

!cp -r site-packages/* /opt/conda/lib/python3.6/site-packages/

# !export LD_LIBRARY_PATH="/kaggle/working/lib/" 

!cp /kaggle/working/lib/libfaiss.so /usr/local/cuda/lib64/


!apt search openblas

!yes Y | apt install libopenblas-dev 

# !printf '%s\n' 0 | update-alternatives --config libblas.so.3 << 0

# !apt-get install libopenblas-dev 

!
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import time

import os

import gc

import seaborn as sns

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA, TruncatedSVD

from sklearn import preprocessing

import matplotlib.patches as mpatches



import faiss

from tsnecuda import TSNE





print(os.listdir("../input"))

import seaborn as sns

import warnings

warnings.simplefilter("ignore")

plt.style.use('ggplot')

color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]
[]
def id_split(dataframe):

    dataframe['device_name'] = dataframe['DeviceInfo'].str.split('/', expand=True)[0]

    dataframe['device_version'] = dataframe['DeviceInfo'].str.split('/', expand=True)[1]



    dataframe['OS_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[0]

    dataframe['version_id_30'] = dataframe['id_30'].str.split(' ', expand=True)[1]



    dataframe['browser_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[0]

    dataframe['version_id_31'] = dataframe['id_31'].str.split(' ', expand=True)[1]



    dataframe['screen_width'] = dataframe['id_33'].str.split('x', expand=True)[0]

    dataframe['screen_height'] = dataframe['id_33'].str.split('x', expand=True)[1]



    dataframe['id_34'] = dataframe['id_34'].str.split(':', expand=True)[1]

    dataframe['id_23'] = dataframe['id_23'].str.split(':', expand=True)[1]



    dataframe.loc[dataframe['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'

    dataframe.loc[dataframe['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'

    dataframe.loc[dataframe['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'

    dataframe.loc[dataframe['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'

    dataframe.loc[dataframe['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'

    dataframe.loc[dataframe['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'

    dataframe.loc[dataframe['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'

    dataframe.loc[dataframe['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'

    dataframe.loc[dataframe['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'

    dataframe.loc[dataframe['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'

    dataframe.loc[dataframe['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'

    dataframe.loc[dataframe['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'

    dataframe.loc[dataframe['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'

    dataframe.loc[dataframe['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'

    dataframe.loc[dataframe['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'

    dataframe.loc[dataframe['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'

    dataframe.loc[dataframe['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'



    dataframe.loc[dataframe.device_name.isin(dataframe.device_name.value_counts()[dataframe.device_name.value_counts() < 200].index), 'device_name'] = "Others"

    dataframe['had_id'] = 1

    gc.collect()

    

    return dataframe
def Negativedownsampling(train, ratio) :

    



    # Number of data points in the minority class

    number_records_fraud = len(train[train.isFraud == 1])

    fraud_indices = np.array(train[train.isFraud == 1].index)



    # Picking the indices of the normal classes

    normal_indices = train[train.isFraud == 0].index



    # Out of the indices we picked, randomly select "x" number (number_records_fraud)

    random_normal_indices = np.random.choice(normal_indices, number_records_fraud*ratio, replace = False)

    random_normal_indices = np.array(random_normal_indices)



    # Appending the 2 indices

    under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])



    # Under sample dataset

    under_sample_data = train.iloc[under_sample_indices,:]

    

    # Showing ratio

    print("Percentage of normal transactions: ", round(len(under_sample_data[under_sample_data.isFraud == 0])/len(under_sample_data),2)* 100,"%")

    print("Percentage of fraud transactions: ", round(len(under_sample_data[under_sample_data.isFraud == 1])/len(under_sample_data),2)* 100,"%")

    print("Total number of transactions in resampled data: ", len(under_sample_data))

    

    return under_sample_data



train_transaction = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')

test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')

sample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')
# merge 

df_train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

df_test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)



print("Train shape : "+str(df_train.shape))

print("Test shape  : "+str(df_test.shape))





del train_transaction, train_identity, test_transaction, test_identity

gc.collect()
drop_col = ['V300', 'V309', 'V111', 'C3', 'V124', 'V106', 'V125', 'V315', 'V134', 'V102', 'V123', 'V316', 'V113', 'V136', 'V305', 'V110', 'V299', 'V289', 'V286', 'V318', 'V103', 'V304', 'V116', 'V298', 'V284', 'V293', 'V137', 'V295', 'V301', 'V104', 'V311', 'V115', 'V109', 'V119', 'V321', 'V114', 'V133', 'V122', 'V319', 'V105', 'V112', 'V118', 'V117', 'V121', 'V108', 'V135', 'V320', 'V303', 'V297', 'V120']
# df_train = id_split(df_train)

# df_test = id_split(df_test)
print("Train shape : "+str(df_train.shape))

print("Test shape  : "+str(df_test.shape))

# df_train.device_name
df = X_train[X_train.ProductCD=='W'] 

plt.scatter(df[df.isFraud==0].TransactionDT,df[df.isFraud==0]['cents'])

plt.scatter(df[df.isFraud==1].TransactionDT,df[df.isFraud==1]['cents'])
features = []

rm_cols = ['TransactionID', 'TransactionDT', 'isFraud']



features = [col for col in list(df_train) if col not in rm_cols]

features = [col for col in list(features) if col not in drop_col]
for f in features:

    if(str(df_train[f].dtype)!="object" and str(df_train[f].dtype) !="category") :

        df_train[f] = df_train[f].replace(np.nan,-999)

        df_test[f] = df_test[f].replace(np.nan,-999)
# for f in ['device_name','OS_id_30', 'version_id_30', 'browser_id_31', 'version_id_31', 'screen_width', 'screen_height', 'id_34' , 'id_23']:

#     df_train[f] = df_train[f].replace(np.nan,-999)

#     df_test[f] = df_test[f].replace(np.nan,-999)

    
df_train.head()
# Label Encoding

for f in features:

    if  (str(df_train[f].dtype)=="object" or str(df_train[f].dtype)=="category")  :  

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(df_train[f].values) + list(df_test[f].values))

        df_train[f] = lbl.transform(list(df_train[f].values))

        df_test[f] = lbl.transform(list(df_test[f].values))

        

df_train = df_train.reset_index()

df_test = df_test.reset_index()
df_train.head()
label = 'isFraud'
np.random.seed(32)



df_train.sort_values('TransactionDT', inplace = True)

df_train_resampling_1 = Negativedownsampling(df_train, 1)
df_train_resampling_1.isFraud.plot.hist()
y_train = df_train_resampling_1[label]
# Filling NaN series

X_train = df_train_resampling_1.fillna(-999)
# np.random.seed(24)

# # label = 'isFraud'

# X_train.sort_values('TransactionDT', inplace = True)

# X_train = Negativedownsampling(X_train, 9)

X_train = X_train.drop(label, axis=1)
pca_result_100 = PCA(n_components=150)

pca_result_100 = pca_result_100.fit_transform(X_train)

# print('Cumulative explained variation for 50 principal components: {}'.format(np.sum(pca_result_100.explained_variance_ratio_)))
t0 = time.time()

tsne_model = TSNE(n_components=2, perplexity=290.0,  n_iter=1100, num_neighbors=175, learning_rate=280.0,).fit_transform(X_train)

t1 = time.time()

print("T-SNE took {:.2} s".format(t1-t0))
tsne_df = pd.DataFrame(tsne_model)

tsne_df = pd.concat([tsne_df,y_train], axis=1)
# (n_components=2, perplexity=300.0,  n_iter=1000, num_neighbors=150, learning_rate=250.0,).
sns.FacetGrid(tsne_df, hue="isFraud" , size=10).map(plt.scatter, 0, 1).add_legend()

plt.show()
# tsne_df.shape
# https://www.kaggle.com/pavansanagapati/anomaly-detection-credit-card-fraud-analysis

random_state = 42

n_components = 2



# T-SNE Implementation

# t0 = time.time()

# X_reduced_tsne = TSNE(n_components=n_components, random_state=random_state).fit_transform(X_train.values)

# t1 = time.time()

# print("T-SNE took {:.2} s".format(t1-t0))



# # PCA Implementation

# t0 = time.time()

# X_reduced_pca = PCA(n_components=n_components, random_state=random_state).fit_transform(X_train.values)

# t1 = time.time()

# print("PCA took {:.2} s".format(t1-t0))



# # TruncatedSVD

# t0 = time.time()

# X_reduced_svd = TruncatedSVD(n_components=n_components, algorithm='randomized', random_state=random_state).fit_transform(X_train.values)

# t1 = time.time()

# print("Truncated SVD took {:.2} s".format(t1-t0))
# f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))

# # labels = ['No Fraud', 'Fraud']

# f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)



# blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')

# red_patch = mpatches.Patch(color='#AF0000', label='Fraud')



# # t-SNE scatter plot

# ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y_train == 0), cmap='coolwarm', label='No Fraud', linewidths=2)

# ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y_train == 1), cmap='coolwarm', label='Fraud', linewidths=2)

# ax1.set_title('t-SNE', fontsize=14)

# ax1.grid(True)

# ax1.legend(handles=[blue_patch, red_patch])



# # PCA scatter plot

# ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y_train == 0), cmap='coolwarm', label='No Fraud', linewidths=2)

# ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y_train == 1), cmap='coolwarm', label='Fraud', linewidths=2)

# ax2.set_title('PCA', fontsize=14)

# ax2.grid(True)

# ax2.legend(handles=[blue_patch, red_patch])



# # TruncatedSVD scatter plot

# ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y_train == 0), cmap='coolwarm', label='No Fraud', linewidths=2)

# ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y_train == 1), cmap='coolwarm', label='Fraud', linewidths=2)

# ax3.set_title('Truncated SVD', fontsize=14)

# ax3.grid(True)

# ax3.legend(handles=[blue_patch, red_patch])



# plt.show()