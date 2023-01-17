import pandas as pd
import numpy as np
cars = pd.read_csv('../input/Imports85Imp.csv') # imported the dataset given by you
cars.head() # shows the first five rows which helps to what kind of data is presented and can think of how to interpret the data

drop1=["V1_imp","V2_imp","V3_imp","V4_imp","V4_imp","V5_imp","V6_imp","V6_imp","V7_imp","V8_imp","V9_imp","V10_imp","V11_imp","V12_imp","V13_imp",
      "V14_imp","V15_imp","V16_imp","V17_imp","V18_imp","V19_imp","V20_imp","V21_imp","V22_imp","V23_imp","V24_imp","V25_imp","V26_imp",
       ]
cars1=cars.drop(drop1, axis=1)
cars1.columns=['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 
           'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio',
            'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
cars1.head()
l = cars1['make'] # we are considering lable make only for visulization purpose, PCA Analysis will be done afterwards

drop2=['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
       'drive-wheels', 'engine-location','engine-type', 'num-of-cylinders', 'fuel-system', ]
cars2=cars1.drop(drop2, axis=1)# removing all other categorical to select continous attributes
cars2.head()
cars2.shape
from sklearn import decomposition
pca = decomposition.PCA()
# configuring the parameteres
# the number of components = 2
pca.n_components = 2
pca_data = pca.fit_transform(cars2)

# pca_reduced will contain the 2-d projects of simple data
print("shape of pca_reduced.shape = ", pca_data.shape)
# attaching the label for each 2-d data point 
import seaborn as sn
import matplotlib.pyplot as plt
pca_data = np.vstack((pca_data.T, l)).T

# creating a new data fram which help us in ploting the result data
pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal", "make"))
sn.FacetGrid(pca_df,hue="make",size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.show()
pca.n_components = 14

pca_data = pca.fit_transform(cars2)
percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);

cum_var_explained = np.cumsum(percentage_var_explained)

# Plot the PCA spectrum
plt.figure(1, figsize=(6, 4))

plt.clf()
plt.plot(cum_var_explained,linewidth=2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.show()
from sklearn.preprocessing import StandardScaler
standardized_data = StandardScaler().fit_transform(cars2)
print(standardized_data.shape)
standardized_data
# configuring the parameteres
# the number of components = 2
pca.n_components = 2
pca_data = pca.fit_transform(standardized_data)

# pca_reduced will contain the 2-d projects of simple data
print("shape of pca_reduced.shape = ", pca_data.shape)
import seaborn as sn
import matplotlib.pyplot as plt
pca_data = np.vstack((pca_data.T, l)).T

# creating a new data fram which help us in ploting the result data
pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal", "make"))
sn.FacetGrid(pca_df,hue="make",size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.show()

pca.n_components = 14
pca_data = pca.fit_transform(standardized_data)

percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);

cum_var_explained = np.cumsum(percentage_var_explained)

# Plot the PCA spectrum
plt.figure(1, figsize=(6, 4))

plt.clf()
plt.plot(cum_var_explained,linewidth=2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.show()
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
normalized_data = preprocessing.normalize(cars2)
print(normalized_data.shape)
normalized_data
# configuring the parameteres
# the number of components = 2
pca.n_components = 2
pca_data_n = pca.fit_transform(normalized_data)

# pca_reduced will contain the 2-d projects of simple data
print("shape of pca_reduced.shape = ", pca_data.shape)
import seaborn as sn
import matplotlib.pyplot as plt
pca_data = np.vstack((pca_data_n.T, l)).T

# creating a new data fram which help us in ploting the result data
pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal", "make"))
sn.FacetGrid(pca_df,hue="make",size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.show()
pca.n_components = 14
pca_data = pca.fit_transform(normalized_data)

percentage_var_explained = pca.explained_variance_ / np.sum(pca.explained_variance_);

cum_var_explained = np.cumsum(percentage_var_explained)

# Plot the PCA spectrum
plt.figure(1, figsize=(6, 4))

plt.clf()
plt.plot(cum_var_explained,linewidth=2)
plt.axis('tight')
plt.grid()
plt.xlabel('n_components')
plt.ylabel('Cumulative_explained_variance')
plt.show()