# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install tensorflow==1.15.0
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

%matplotlib inline



import tensorflow as tf

from tqdm import tqdm

print(tf.__version__)



# Make numpy values easier to read.

np.set_printoptions(precision=3, suppress=True)
path = "/kaggle/input/walmart-recruiting-store-sales-forecasting/"

dataset = pd.read_csv(path + "train.csv.zip", names=['Store','Dept','Date','weeklySales','isHoliday'],sep=',', header=0)

features = pd.read_csv(path + "features.csv.zip",sep=',', header=0,

                       names=['Store','Date','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4',

                              'MarkDown5','CPI','Unemployment','IsHoliday']).drop(columns=['IsHoliday'])

stores = pd.read_csv(path + "stores.csv", names=['Store','Type','Size'],sep=',', header=0)

dataset = dataset.merge(stores, how='left').merge(features, how='left')



dataset.head()
sales = dataset.groupby(['Dept', 'Date', 'Store'])['weeklySales'].sum().unstack()



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

data = scaler.fit_transform(sales).astype(np.float32)

sales_scaled = pd.DataFrame(data=data, columns=sales.columns, index=sales.index)



sales_complete = sales_scaled[sales_scaled.isna().sum(axis=1) == 0]

print(sales_complete.shape)
data_new = sales_complete.to_numpy()
#%% System Parameters

# 1. Mini batch size

mb_size = 128

# 2. Missing rate

#p_miss = 0.2

# 3. Hint rate

p_hint = 0.9

# 4. Loss Hyperparameters

alpha = 10

# 5. Train Rate

train_rate = 0.8



# Parameters

No, Dim = sales_complete.shape



# Hidden state dimensions

H_Dim1 = Dim

H_Dim2 = Dim



print(Dim)
# Prepare the data set

miss_rate = 0.2



def binary_sampler(p, rows, cols):

    unif_random_matrix = np.random.uniform(0.0, 1.0, size=[rows, cols])

    binary_random_matrix = (unif_random_matrix < p).astype(np.float32)

    return binary_random_matrix



data_m = binary_sampler(1-miss_rate, No, Dim)

miss_data_x = sales_complete.copy().to_numpy()

miss_data_x[data_m == 0] = 123.456 # np.nan will create error in X*M

train_dataset = tf.data.Dataset.from_tensor_slices(miss_data_x).batch(mb_size, drop_remainder=True)

maskset = tf.data.Dataset.from_tensor_slices(data_m).batch(mb_size, drop_remainder=True)
Missing = data_m
def normalization (data):

    '''Normalize data in [0, 1] range.



    Args:

    - data: original data



    Returns:

    - norm_data: normalized data

    - norm_parameters: min_val, max_val for each feature for renormalization

    '''



    # Parameters

    _, dim = data.shape

    norm_data = data.copy()



    # MixMax normalization

    min_val = np.zeros(dim)

    max_val = np.zeros(dim)



    # For each dimension

    for i in range(dim):

        min_val[i] = np.nanmin(norm_data[:,i])

        norm_data[:,i] = norm_data[:,i] - np.nanmin(norm_data[:,i])

        max_val[i] = np.nanmax(norm_data[:,i])

        norm_data[:,i] = norm_data[:,i] / (np.nanmax(norm_data[:,i]) + 1e-6)   



    # Return norm_parameters for renormalization

    norm_parameters = {'min_val': min_val,

                     'max_val': max_val}



    return norm_data, norm_parameters
norm_data,norm_parameters=normalization(data_new)

norm_data_x = np.nan_to_num(norm_data, 0)
norm_parameters
#%% Train Test Division    

   

idx = np.random.permutation(No)

#idx=list(idx)



Train_No = int(No * train_rate)

Test_No = No - Train_No

batch_idx=idx[:Train_No]



# Train / Test Features

trainX = norm_data_x #use entire dataset for training

testX = norm_data_x #supply the entire data as test data 



# Train / Test Missing Indicators

trainM = Missing

testM = Missing #supply entire missing mask
# 1. Xavier Initialization Definition

def xavier_init(size):

    in_dim = size[0]

    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)

    return tf.random_normal(shape = size, stddev = xavier_stddev)

    

# Hint Vector Generation

def sample_M(m, n, p):

    A = np.random.uniform(0., 1., size = [m, n])

    B = A > p

    C = 1.*B

    return C
#%% GAIN Architecture   

   

#%% 1. Input Placeholders

# 1.1. Data Vector

X = tf.placeholder(tf.float32, shape = [None, Dim])

# 1.2. Mask Vector 

M = tf.placeholder(tf.float32, shape = [None, Dim])

# 1.3. Hint vector

H = tf.placeholder(tf.float32, shape = [None, Dim])

# 1.4. X with missing values

New_X = tf.placeholder(tf.float32, shape = [None, Dim])
#%% 2. Discriminator

D_W1 = tf.Variable(xavier_init([Dim*2, H_Dim1]), name="D_W1")     # Data + Hint as inputs

D_b1 = tf.Variable(tf.zeros(shape = [H_Dim1]), name="D_b1")



D_W2 = tf.Variable(xavier_init([H_Dim1, H_Dim2]), name="D_W2")

#D_W2 = tf.Variable(xavier_init([11, 38]), name="D_W2")

D_b2 = tf.Variable(tf.zeros(shape = [H_Dim2]), name="D_b2")

#D_b2 = tf.Variable(tf.zeros(shape = [38]), name="D_b2")



D_W3 = tf.Variable(xavier_init([H_Dim2, Dim]), name="D_W3")

#D_W3 = tf.Variable(xavier_init([38, Dim]), name="D_W3")

D_b3 = tf.Variable(tf.zeros(shape = [Dim]), name="D_b3")       # Output is multi-variate



theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]
#%% 3. Generator

G_W1 = tf.Variable(xavier_init([Dim*2, H_Dim1]), name="G_W1")     # Data + Mask as inputs (Random Noises are in Missing Components)

G_b1 = tf.Variable(tf.zeros(shape = [H_Dim1]), name="G_b1")



G_W2 = tf.Variable(xavier_init([H_Dim1, H_Dim2]), name="G_W2")

#G_W2 = tf.Variable(xavier_init([11, 38]), name="G_W2")

G_b2 = tf.Variable(tf.zeros(shape = [H_Dim2]), name="G_b2")

#G_b2 = tf.Variable(tf.zeros(shape = [38]), name="G_b2")



G_W3 = tf.Variable(xavier_init([H_Dim2, Dim]), name="G_W3")

#G_W3 = tf.Variable(xavier_init([38, Dim]), name="G_W3")

G_b3 = tf.Variable(tf.zeros(shape = [Dim]), name="G_b3")



theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]
#%% GAIN Function



#%% 1. Generator

def generator(new_x,m):

    inputs = tf.concat(axis = 1, values = [new_x,m])  # Mask + Data Concatenate

    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)

    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)   

    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3) # [0,1] normalized Output

    

    return G_prob

    

#%% 2. Discriminator

def discriminator(new_x, h):

    inputs = tf.concat(axis = 1, values = [new_x,h])  # Hint + Data Concatenate

    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)  

    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)

    D_logit = tf.matmul(D_h2, D_W3) + D_b3

    D_prob = tf.nn.sigmoid(D_logit)  # [0,1] Probability Output

    

    return D_prob
#%% 3. Other functions

# Random sample generator for Z

def sample_Z(m, n):

    return np.random.uniform(0., 0.01, size = [m, n])        



# Mini-batch generation

def sample_idx(m, n):

    A = np.random.permutation(m)

    idx = A[:n]

    return idx
def renormalization (norm_data, norm_parameters):

    '''Renormalize data from [0, 1] range to the original range.



    Args:

    - norm_data: normalized data

    - norm_parameters: min_val, max_val for each feature for renormalization



    Returns:

    - renorm_data: renormalized original data

    '''



    min_val = norm_parameters['min_val']

    max_val = norm_parameters['max_val']



    _, dim = norm_data.shape

    renorm_data = norm_data.copy()



    for i in range(dim):

        renorm_data[:,i] = renorm_data[:,i] * (max_val[i] + 1e-6)   

        renorm_data[:,i] = renorm_data[:,i] + min_val[i]



    return renorm_data
def rounding (imputed_data, data_x):

    '''Round imputed data for categorical variables.



    Args:

    - imputed_data: imputed data

    - data_x: original data with missing values



    Returns:

    - rounded_data: rounded imputed data

    '''



    _, dim = data_x.shape

    rounded_data = imputed_data.copy()



    for i in range(dim):

        temp = data_x[~np.isnan(data_x[:, i]), i]

        # Only for the categorical variable

        if len(np.unique(temp)) < 100:

              rounded_data[:, i] = np.round(rounded_data[:, i])



    return rounded_data
#%% Structure

# Generator

G_sample = generator(New_X,M)



# Combine with original data

Hat_New_X = New_X * M + G_sample * (1-M)



# Discriminator

D_prob = discriminator(Hat_New_X, H)



#%% Loss

D_loss1 = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1-M) * tf.log(1. - D_prob + 1e-8)) 

G_loss1 = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))

MSE_train_loss = tf.reduce_mean((M * New_X - M * G_sample)**2) / tf.reduce_mean(M)



D_loss = D_loss1

G_loss = G_loss1 + alpha * MSE_train_loss 



#%% MSE Performance metric

MSE_test_loss = tf.reduce_mean(((1-M) * X - (1-M)*G_sample)**2) / tf.reduce_mean(1-M)



#%% Solver

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)

G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
# Sessions

import time

sess1 = tf.Session()

sess1.run(tf.global_variables_initializer())



#%% Iterations

train_losses = []

test_losses = []



#%% Start Iterations

t=time.time()

for it in tqdm(range(1000)):    

    

    #%% Inputs

    mb_idx = sample_idx(Train_No, mb_size)

    X_mb = trainX[mb_idx,:]  

    #print(X_mb.shape)

    

    Z_mb = sample_Z(mb_size, Dim) 

#     M_mb = trainM[mb_idx,:]  

    M_mb = trainM[:mb_size, :]

    #print(M_mb.shape)

    H_mb1 = sample_M(mb_size, Dim, 1-p_hint)

    H_mb = M_mb * H_mb1 #Hint matrix

    

    New_X_mb = M_mb * X_mb+(1-M_mb) * Z_mb  # random value z in inserted in place of missing Data

    #print("Missing data shape",New_X_mb.shape)

    

    _, D_loss_curr = sess1.run([D_solver, D_loss1], feed_dict = {M: M_mb, New_X: New_X_mb, H: H_mb})

    _, G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = sess1.run([G_solver, G_loss1, MSE_train_loss, MSE_test_loss],

                                                                       feed_dict = {X: X_mb, M: M_mb, New_X: New_X_mb, H: H_mb})

    #print('Train loss: ',np.sqrt(MSE_train_loss_curr), 'Test loss: ', np.sqrt(MSE_test_loss_curr))

    train_losses.append(np.sqrt(MSE_train_loss_curr))

    test_losses.append(np.sqrt(MSE_test_loss_curr))

print("Test dataset imputation")    

#%% Final Loss

t_final=time.time()    

Z_mb = sample_Z(len(norm_data), Dim) 

M_mb = testM

X_mb = norm_data

        

New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb  # Missing Data Introduce

    

    

MSE_final,imputed_data = sess1.run([MSE_test_loss,G_sample], feed_dict = {X:X_mb , M: testM, New_X: New_X_mb})

imputed_data = testM * norm_data + (1-testM) * imputed_data



 # Renormalization

imputed_data = renormalization(imputed_data, norm_parameters)  

        

# Rounding

#data=data.to_numpy()

imputed_data = rounding(imputed_data, data_new)  

print('Final Test RMSE: ' + str(np.sqrt(MSE_final)))

print('Time cost: ',t_final-t)

#sess1.close()
data_new.shape
# How to split into 6 subsets with ~50% overlap?

# Subset 1: 0 - 11

# Subset 2: 6 - 17

# Subset 3: 12 - 23

# Subset 4: 18 - 29

# Subset 5: 24 - 35

# Subset 6: 30 - 41

# Remove 42 - 44



# 12 features for each subset, and 6 features overlap

# Number of instances: 4417 / 6 ~= 700



# All is needed: 1. split data_new and Missing into 6 subsets. 2. repeat the above tests.