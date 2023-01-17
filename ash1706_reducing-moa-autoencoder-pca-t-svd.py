# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd 

import numpy as np



import seaborn as sns

sns.set_style("dark")

import matplotlib.pyplot as plt





from sklearn.manifold import TSNE

from sklearn.decomposition import TruncatedSVD



from time import time



import os



import warnings

warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler




from xgboost import XGBClassifier

from sklearn.model_selection import KFold

from category_encoders import CountEncoder

from sklearn.pipeline import Pipeline

from sklearn.metrics import log_loss , accuracy_score



import matplotlib.pyplot as plt



from sklearn.multioutput import MultiOutputClassifier



import os

import warnings

warnings.filterwarnings('ignore')

df_train = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")

df_test = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")

train_scored = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")

train_non_scored = pd.read_csv("/kaggle/input/lish-moa/train_targets_nonscored.csv")

SEED = 42

NFOLDS = 5

DATA_DIR = '/kaggle/input/lish-moa/'

np.random.seed(SEED)
features = df_train
common  = ['sig_id',

 'cp_type',

 'cp_time',

 'cp_dose']





genes = list(filter(lambda x : "g-" in x  , list(features)))



cells = list(filter(lambda x : "c-" in x  , list(features)))
train = df_train

train['type'] = 'train'

test = df_test

test['type'] = 'test'

X = train.append(test)

X = pd.get_dummies(columns = ['cp_type' , 'cp_dose', 'cp_time'], drop_first =True , data = X)

target  = train_scored.drop(['sig_id'] , axis =1)





# lets label encode cp_type , cp_dose and cp_time

# X = pd.get_dummies(columns = ['cp_type' , 'cp_dose', 'cp_time'], drop_first =True , data = X)

# X[numeric_cols] = StandardScaler().fit_transform(X[numeric_cols])

numeric_cols = genes+cells

classifier = MultiOutputClassifier(XGBClassifier(tree_method='gpu_hist'))

# classifier = MultiOutputClassifier(XGBClassifier())



clf = Pipeline([

                ('classify', classifier)

               ])

params = {'classify__estimator__colsample_bytree': 0.6522,

          'classify__estimator__gamma': 3.6975,

          'classify__estimator__learning_rate': 0.0503,

          'classify__estimator__max_delta_step': 2.0706,

          'classify__estimator__max_depth': 10,

          'classify__estimator__min_child_weight': 31.5800,

          'classify__estimator__n_estimators': 166,

          'classify__estimator__subsample': 0.8639

         }



_ = clf.set_params(**params)
def modeling_xg_boost(x, y, x_test):

    oof_preds = np.zeros(y.shape)

    test_preds = np.zeros((test.shape[0], y.shape[1]))

    oof_losses = []

    kf = KFold(n_splits=NFOLDS)

    for fn, (trn_idx, val_idx) in enumerate(kf.split(x, y)):

        print('Starting fold: ', fn)

        X_train, X_val = x[trn_idx], x[val_idx]

        y_train, y_val = y[trn_idx], y[val_idx]

        

        # drop where cp_type==ctl_vehicle (baseline)

        ctl_mask = X_train[:,-4]==0

        X_train = X_train[~ctl_mask,:]

        y_train = y_train[~ctl_mask]

        

        clf.fit(X_train, y_train)

        val_preds = clf.predict_proba(X_val) # list of preds per class

        val_preds = np.array(val_preds)[:,:,1].T # take the positive class

        oof_preds[val_idx] = val_preds

        

        loss = log_loss(np.ravel(y_val), np.ravel(val_preds))

        oof_losses.append(loss)

        preds = clf.predict_proba(x_test)

        preds = np.array(preds)[:,:,1].T # take the positive class

        test_preds += preds / NFOLDS

        

    print(oof_losses)

    print('Mean OOF loss across folds', np.mean(oof_losses))

    print('STD OOF loss across folds', np.std(oof_losses))

    control_mask = X[X['type'] =='train']['cp_type_trt_cp'] ==0

    oof_preds[control_mask] = 0



    print('\n OOF log loss: ', log_loss(np.ravel(y), np.ravel(oof_preds)))

    

    return  test_preds

    
n_comp = [2,4,6,10 ,16,20,30,50,100,150,200,300,450,600] # list containing different values of components

explained = [] # explained variance ratio for each component of PCA

for x in n_comp:

    pca_gene = PCA(n_components=x)

    pca_gene.fit(X[genes])

    explained.append(pca_gene.explained_variance_ratio_.sum())

    print("Number of components = %r and explained variance = %r"%(x,pca_gene.explained_variance_ratio_.sum()))

plt.plot(n_comp, explained)

plt.xlabel('Number of components')

plt.ylabel("Explained Variance")

plt.title("Plot of Number of components v/s explained variance")

plt.show()
n_comp = [1,2,4,6,8,10,16,20,30,50] # list containing different values of components

explained = [] # explained variance ratio for each component of PCA

for x in n_comp:

    pca_cell = PCA(n_components=x)

    pca_cell.fit(X[cells])

    explained.append(pca_cell.explained_variance_ratio_.sum())

    print("Number of components = %r and explained variance = %r"%(x,pca_cell.explained_variance_ratio_.sum()))

plt.plot(n_comp, explained)

plt.xlabel('Number of components')

plt.ylabel("Explained Variance")

plt.title("Plot of Number of components v/s explained variance")

plt.show()
#pca_gene data

pca_gene = PCA(n_components=450)

pca_gene_data = pca_gene.fit_transform(X[genes])

inter_pc_gene = pd.DataFrame(data = pca_gene_data)



Pca = X

transformed_genes = [str(i)+str('_gene') for i in list(inter_pc_gene) ]

Pca[transformed_genes] = inter_pc_gene[:]







pca_cell = PCA(n_components=2)

pca_cell_data = pca_cell.fit_transform(X[cells])

inter_pc_cell = pd.DataFrame(data = pca_cell_data

             , columns = ['PC1', 'PC2'])

Pca['PC1_cell'] = inter_pc_cell['PC1']

Pca['PC2_cell'] = inter_pc_cell['PC2']
features_final_pca = transformed_genes + ['PC1_cell', 'PC2_cell', 'cp_type_trt_cp', 'cp_dose_D2', 'cp_time_48', 'cp_time_72']


x_pca = Pca[Pca['type']  == 'train'][features_final_pca].to_numpy()

y_pca = target.to_numpy()

x_test_pca = Pca[Pca['type']  == 'test'][features_final_pca].to_numpy()
#  p = modeling_xg_boost(x_pca ,y_pca,x_test_pca )
n_comp = [2,4,6,10 ,16,20,30,50,100,150,200,300,450,600] # list containing different values of components

explained = [] # explained variance ratio for each component of PCA

for x in n_comp:

    svd_gene =TruncatedSVD(n_components=x)

    svd_gene.fit(X[genes])

    explained.append(svd_gene.explained_variance_ratio_.sum())

    print("Number of components = %r and explained variance = %r"%(x,svd_gene.explained_variance_ratio_.sum()))

plt.plot(n_comp, explained)

plt.xlabel('Number of components')

plt.ylabel("Explained Variance")

plt.title("Plot of Number of components v/s explained variance")

plt.show()
n_comp = [1,2,4,6,8,10,16,20,30,50] # list containing different values of components

explained = [] # explained variance ratio for each component of PCA

for x in n_comp:

    svd_cell = TruncatedSVD(n_components=x)

    svd_cell.fit(X[cells])

    explained.append(svd_cell.explained_variance_ratio_.sum())

    print("Number of components = %r and explained variance = %r"%(x,svd_cell.explained_variance_ratio_.sum()))

plt.plot(n_comp, explained)

plt.xlabel('Number of components')

plt.ylabel("Explained Variance")

plt.title("Plot of Number of components v/s explained variance")

plt.show()
svd_gene = TruncatedSVD(n_components=450)

svd_gene_data = svd_gene.fit_transform(X[genes])

inter_pc_gene = pd.DataFrame(data = svd_gene_data)



Svd = X

transformed_genes = [str(i)+str('_gene') for i in list(inter_pc_gene) ]

Svd[transformed_genes] = inter_pc_gene[:]

svd_cell = TruncatedSVD(n_components=2)

svd_cell_data =svd_cell.fit_transform(X[cells])

inter_pc_cell = pd.DataFrame(data = svd_cell_data

             , columns = ['PC1', 'PC2'])

Svd['PC1_cell'] = inter_pc_cell['PC1']

Svd['PC2_cell'] = inter_pc_cell['PC2']
features_final_svd = transformed_genes + ['PC1_cell', 'PC2_cell', 'cp_type_trt_cp', 'cp_dose_D2', 'cp_time_48', 'cp_time_72']
x_svd = Svd[Svd['type']  == 'train'][features_final_svd].to_numpy()

y_svd = target.to_numpy()

x_test_svd = Svd[Svd['type']  == 'test'][features_final_svd].to_numpy()
# p2 = modeling_xg_boost(x_svd ,y_svd,x_test_svd )
from keras.models import Model

from keras.layers import Input, Dense

from keras import regularizers

from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()

# data_scaled = scaler.fit_transform(X[numeric_cols])

data_scaled_genes = X[genes]

data_scaled_cells = X[cells]
# Fixed dimensions

input_dim_genes = X[genes].shape[1]  # 8

encoding_dim_genes = 10

# Number of neurons in each Layer [8, 6, 4, 3, ...] of encoders

input_layer_genes = Input(shape=(input_dim_genes, ))

encoder_layer_1_genes = Dense(6, activation="tanh", activity_regularizer=regularizers.l1(10e-5))(input_layer_genes)

encoder_layer_2_genes = Dense(4, activation="tanh")(encoder_layer_1_genes)

encoder_layer_3_genes = Dense(encoding_dim_genes, activation="tanh")(encoder_layer_2_genes)
# Creat encoder model

encoder_genes = Model(inputs=input_layer_genes, outputs=encoder_layer_3_genes)

# Use the model to predict the factors which sum up the information of interest rates.

encoded_data_genes = pd.DataFrame(encoder_genes.predict(data_scaled_genes))

encoded_cols_list_gene = [str(i)+'_feature' for i in list(encoded_data_genes)]

encoded_data_genes.columns = encoded_cols_list_gene

# Fixed dimensions

input_dim_cells = X[cells].shape[1]  # 8

encoding_dim_cells = 2

# Number of neurons in each Layer [8, 6, 4, 3, ...] of encoders

input_layer_cells = Input(shape=(input_dim_cells, ))

encoder_layer_1_cells = Dense(6, activation="tanh", activity_regularizer=regularizers.l1(10e-5))(input_layer_cells)

encoder_layer_2_cells = Dense(4, activation="tanh")(encoder_layer_1_cells)

encoder_layer_3_cells = Dense(encoding_dim_cells, activation="tanh")(encoder_layer_2_cells)
# Creat encoder model

encoder_cells = Model(inputs=input_layer_cells, outputs=encoder_layer_3_cells)

# Use the model to predict the factors which sum up the information of interest rates.

encoded_data_cells = pd.DataFrame(encoder_cells.predict(data_scaled_cells))

encoded_cols_list_cells = [str(i)+'_feature_cells' for i in list(encoded_data_cells)]

encoded_data_cells.columns = encoded_cols_list_cells



final_encoded = encoded_data_cells

final_encoded[encoded_cols_list_gene] = encoded_data_genes[:]

encoded_cols = encoded_cols_list_cells + encoded_cols_list_gene
final_encoded[['cp_type_trt_cp', 'cp_dose_D2', 'cp_time_48', 'cp_time_72' , 'type']]   = X[['cp_type_trt_cp', 'cp_dose_D2', 'cp_time_48', 'cp_time_72' , 'type']].values
feats = encoded_cols + ['cp_type_trt_cp', 'cp_dose_D2' , 'cp_time_48',  'cp_time_72']

x_encode = final_encoded[final_encoded['type']  == 'train'][feats].to_numpy()

y_encode = target.to_numpy()

x_test_encode = final_encoded[final_encoded['type']  == 'test'][feats].to_numpy()
test_preds = modeling_xg_boost(x_encode ,y_encode,x_test_encode)
# p
# test_pred_f = (test_preds +p+p2)/3
control_mask = final_encoded[final_encoded['type'] =='test']['cp_type_trt_cp'] == 0

test_preds[control_mask] = 0
sub = pd.read_csv(DATA_DIR + 'sample_submission.csv')

sub.iloc[:,1:] = test_preds

sub.to_csv('submission.csv', index=False)
sub.head()
##