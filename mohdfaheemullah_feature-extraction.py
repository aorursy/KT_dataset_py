#####Load python library

# Importing pandas to perform operations using DataFrames 

import pandas as pd  



# Importing numpy to perform Matrix operations 

import numpy as np



# Importing matplotlib to plot graphs

import matplotlib.pyplot as plt
# Importing the following libraries for preprocessing

from sklearn.preprocessing import StandardScaler



# Importing the library for PCA

from sklearn.decomposition import PCA as sklearnPCA
# importing electric motor data set

electric_motor_data=pd.read_csv('../input/electric-motor-temperature/pmsm_temperature_data.csv')
# Information on the data

print (electric_motor_data.info())

print ('\n')
electric_motor_data.isnull().sum()
electric_motor_data.describe()
# creating a list of columns of the original data 

electric_motor_data_columns_list = list(electric_motor_data.columns)

print (electric_motor_data_columns_list)

print ('\n')
# dropping column 'Id' 

electric_motor_data = electric_motor_data.drop(['profile_id'],axis=1)

print(electric_motor_data)
# Scaling data using (x-mu)/sigma 

scaler                       = StandardScaler()

Input_electric_motor_data_columns_list = list(electric_motor_data.columns)

electric_motor_data[Input_electric_motor_data_columns_list] = scaler.fit_transform(electric_motor_data[Input_electric_motor_data_columns_list])

print (electric_motor_data)
# computing covariance using scaled data (renamed the data as 'input_data')

input_data                   = electric_motor_data[Input_electric_motor_data_columns_list]

covariance_matrix            = input_data.cov()

print (covariance_matrix)
# Computing Eigen values and Eigen vectors of the Covariance Matrix 

eig_vals, eig_vecs = np.linalg.eig(covariance_matrix[Input_electric_motor_data_columns_list].values)

len(eig_vals)

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i])for i in range(len(eig_vals))]

#abs - absolute value

eig_pairs.sort(key = lambda x: x[0], reverse=True)# sort eig_pairs in descending order based on the eigen values
print('Eigenvalues in descending order:')

for i in eig_pairs:

    print(i[0])
# setting threshold as '95% variance'  

threshold = 0.95
# Computing number of PCS required to captured specified variance

print('Explained variance in percentage:\n')

Total_variance = 0.0

count          = 0

eigv_sum       = np.sum(eig_vals)

for i,j in enumerate(eig_pairs):

    variance_explained = (j[0]/eigv_sum).real

    print('eigenvalue {}: {}'.format(i+1, (j[0]/eigv_sum).real*100 ))

    Total_variance     = Total_variance+variance_explained

    count              = count+1

# using break command to comeout of the 'for' loop after meeting the threshold

    if (Total_variance>=threshold):

        break

print(Total_variance)
len(eig_vecs)

count
# select required PCs based on the count  - projection matrix w=d*k

reduced_dimension   = np.zeros((len(eig_vecs),count))

for i in range(count):

    reduced_dimension[:,i]= eig_pairs[i][1]
# Projecting the scaled data onto the reduced space (using eigen vectors)

projected_data = electric_motor_data[Input_electric_motor_data_columns_list].values.dot(reduced_dimension)

projected_dataframe = pd.DataFrame(projected_data,

                                   columns=['Feature_1','Feature_2','Feature_3','Feature_4','Feature_5','Feature_6'])
projected_dataframe.head()