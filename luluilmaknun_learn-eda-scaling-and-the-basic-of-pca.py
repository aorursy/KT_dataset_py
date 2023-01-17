# Import Library



import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Mengambil data, kode disesuaikan dengan platform (Kaggle)



data = pd.read_csv('../input/birds-bones-and-living-habits/bird.csv', index_col='id')

display(data)
data.info()
# Deskripsi statistik data



data.describe()
# Membersihkan data yang Null



cleaned = data.dropna()
cleaned_y = cleaned['type']

cleaned_x = cleaned.drop(['type'], axis=1)
# Plotting boxplot. Terdapat beberapa entry yang terdeteksi sebagai outlier, dengan definisi outlier adalah data yang keluuar dari boxplot



for feature in cleaned_x.columns:

    sns.boxplot(x="type", y=feature, data=cleaned)

    sns.swarmplot(x="type", y=feature, data=cleaned, color="0.3")

    plt.show()
# Type adalah fitur "utama" yang paling menarik, karena paling mungkin untuk dilakukan prediksi padanya.



plt.title("Bird Type Data Count")

sns.countplot(x="type", data=cleaned)
# Menghitung Covariance Matrix



cov_matrix = np.cov(cleaned_x.T)

cov_matrix
# Plotting Covariance Matrix



plt.figure(figsize=(15,15))



sns.heatmap(cov_matrix.T, 

        annot=True,

        cbar = False,

        fmt="0.2f",

        cmap="YlGnBu",

        xticklabels=cleaned_x.columns,

        yticklabels=cleaned_x.columns)

plt.title("Covariance matrix")
# Plotting Covariance Matrix



plt.figure(figsize=(15,15))



sns.heatmap(cleaned_x.corr(), 

        annot=True,

        cbar = False,

        fmt="0.2f",

        cmap="YlGnBu",

        xticklabels=cleaned_x.columns,

        yticklabels=cleaned_x.columns)

plt.title("Correlation matrix")
for feature in df_special_l.columns:

    for typ in cleaned_y.unique():

        df = cleaned[cleaned['type'] == typ]

        sns.distplot(a=df[feature], label=typ)

    plt.title(feature)

    plt.show()
for feature in df_special_w.columns:

    for typ in cleaned_y.unique():

        df = cleaned[cleaned['type'] == typ]

        sns.distplot(a=df[feature], label=typ)

    plt.title(feature)

    plt.show()
sns.pairplot(cleaned, hue='type')
# Kategori fitur "Length"



df_special_l = data.drop(['tarw','tibw','femw','ulnaw','humw', 'type'], axis=1)

df_special_l
# Plotting distribusi data pada fitur-fitur kategori "length"



plt.figure(figsize=(15,10))



for feature in df_special_l.columns:

    sns.distplot(a=df_special_l[feature], label=feature)

plt.legend()

plt.title("Distribution of 'Length' features")

plt.show()
# Kategori fitur "Diameter"



df_special_w = data.drop(['tarl','tibl','feml','ulnal','huml','type'], axis=1)

df_special_w
# Plotting distribusi data pada fitur-fitur kategori "length"



plt.figure(figsize=(15,10))



for feature in df_special_w.columns:

    sns.distplot(a=df_special_w[feature], label=feature)

plt.title("Distribution of 'Diameter' features")

plt.show()
# Scatter plot 



for i in range(len(df_special_l.columns)):

    sns.scatterplot(x=df_special_l.columns[i], y=df_special_w.columns[i], data=cleaned, hue='type')

    plt.title("%s vs %s" % (df_special_l.columns[i],df_special_w.columns[i]))

    plt.show()
def normalisasi(data, scaler):

    data_norm = scaler.fit_transform(data)

    return data_norm
min_max_scale = MinMaxScaler()

min_max = normalisasi(cleaned_x, min_max_scale)

new_min_max = pd.DataFrame(min_max, columns=cleaned_x.columns)

new_min_max['type'] = cleaned_y

new_min_max
new_min_max.describe()
# Plotting Covariance Matrix



plt.figure(figsize=(15,15))



sns.heatmap(np.cov(min_max.T), 

        annot=True,

        cbar = False,

        fmt="0.3f",

        cmap="YlGnBu",

        xticklabels=cleaned_x.columns,

        yticklabels=cleaned_x.columns)

plt.title("Covariance matrix")
standard_scale = StandardScaler()

standard = normalisasi(cleaned_x, standard_scale)

new_standard = pd.DataFrame(standard, columns=cleaned_x.columns)

new_standard['type'] = cleaned_y

new_standard
new_standard.describe()
# Plotting Covariance Matrix



plt.figure(figsize=(15,15))



sns.heatmap(np.cov(standard.T), 

        annot=True,

        cbar = False,

        fmt="0.2f",

        cmap="YlGnBu",

        xticklabels=cleaned_x.columns,

        yticklabels=cleaned_x.columns)

plt.title("Covariance matrix")
# Mencari eigenvector dan eigenvalue dari covariance matrix



eig_values, eig_vectors = np.linalg.eig(cov_matrix)

print("Eigen Values of dataset: ", eig_values)

print()

print("Eigen vector of dataset: ", eig_vectors)
# Melihat signifikansi masing-masing eigen dengan frekuensi kumulatif



eig_sum = np.sum(eig_values)

data_eig = [(i / eig_sum)*100 for i in sorted(eig_values, reverse=True)]

data_fr = np.cumsum(data_eig)

data_fr
sns.lineplot(y=data_fr, x=range(len(data_fr)))

sns.lineplot(y=99, x=range(len(data_fr)))

plt.title("Cummulative frequency of Eigenvalue")
# Dengan library

from sklearn.decomposition import PCA



pca = PCA(0.99)

skl_pca = pca.fit_transform(cleaned_x).T
# Top eigen values of reduced dimention



eig_selected = pca.explained_variance_

print("Numer of dimension: ", len(eig_selected))
skl_pca = skl_pca.T

skl_pca = pd.DataFrame(skl_pca)

skl_pca
skl_pca['type'] = cleaned_y
sns.pairplot(skl_pca, hue='type')