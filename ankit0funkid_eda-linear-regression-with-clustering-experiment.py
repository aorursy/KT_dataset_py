import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

plt.rc('figure', figsize=(10,10))
# Importing the data

data = pd.read_csv('/kaggle/input/iris/Iris.csv')
# Understanding the properties of the columns

data.describe()
data.info()
 # Splitting the data in 3:1 Ratio for Training and Testing

from sklearn.model_selection import train_test_split

x_data = data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

y_data = data[['Species']]

x_train,x_test, y_train, y_test = train_test_split(x_data,y_data,test_size = 0.33, random_state = 0)
# Correlation of various values
data_tr = pd.concat([x_train,y_train], axis = 1)

sns.heatmap(data_tr.corr())
# The corr function does not consider the Species column as its non-numeric
# Understanding the categorical values



figure , ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)

sns.violinplot('Species','SepalLengthCm', data = data_tr

            ,ax = ax1

            ,fliersize = 7)

sns.violinplot('Species','SepalWidthCm', data = data_tr

            ,ax = ax2

           ,fliersize = 7)

sns.violinplot('Species','PetalLengthCm', data = data_tr

            ,ax = ax3

           ,fliersize = 7)

sns.violinplot('Species','PetalWidthCm', data = data_tr

            ,ax = ax4

           ,fliersize = 7)
# Label Encoding

from sklearn.preprocessing import LabelEncoder

data_tr_speciesCat = data_tr.copy()

lb_species = LabelEncoder()

data_tr_speciesCat['Species'] = lb_species.fit_transform(data_tr_speciesCat['Species'])

data_tr_speciesCat.head()
plt.rc('figure'

       , figsize=(20,10))

sns.heatmap(data_tr_speciesCat.corr()

            ,cmap = 'Greens')
# But this label encoding will provide weightd to different categories
#One Hot Encoding

# Trying out One Hot Encoding to see how the corr gets affected
data_tr_OHE = data_tr.copy()

data_tr_OHE = pd.get_dummies(data_tr_OHE

                             ,columns = ['Species']

                             ,prefix = ['Species'])

data_tr_OHE.head()
corr_df = data_tr_OHE.corr()

corr_df.drop(['Species_Iris-setosa','Species_Iris-versicolor','Species_Iris-virginica']

             ,axis = 1

             ,inplace = True)

corr_df.drop(['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']

             ,axis = 0

             ,inplace = True)

corr_df
sns.heatmap(corr_df

            ,vmin = -1

            ,annot = True

            ,cmap = 'BuPu')
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

X = data_tr_speciesCat[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

Y = data_tr_speciesCat[['Species']]

lm.fit(X,Y)

print('Co-effecient: ',lm.coef_)

print('Intercept: ',lm.intercept_)
# Predicting values from this model

yhat = lm.predict(X)
ax1 = sns.distplot(data_tr_speciesCat[['Species']], hist = False, color = 'r', label = 'Actual Value')

sns.distplot(yhat, hist = False, color = 'b', label = 'Fitted Value' , ax = ax1)
from sklearn.metrics import mean_squared_error

print('MSE :',mean_squared_error(Y,yhat))

print('R-Squared :',lm.score(X,Y))
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree = 3)

x_poly = poly_features.fit_transform(X)

poly_lm = LinearRegression()

poly_lm.fit(x_poly,Y)
print('Co-effecient: ',poly_lm.coef_)

print('Intercept: ',poly_lm.intercept_)
# Predicting values uising this model

y_poly_predict = poly_lm.predict(x_poly)
ax1 = sns.distplot(data_tr_speciesCat[['Species']], hist = False, color = 'r', label = 'Actual Value')

sns.distplot(y_poly_predict, hist = False, color = 'b', label = 'Fitted Value' , ax = ax1)
y_poly_predict[:5]
# Plotting the Predicted values against the actual values

polyReg_df = pd.DataFrame({})

polyReg_df['Actual Values'] = data_tr_speciesCat['Species']

polyReg_df['Predicted Values'] = y_poly_predict

plt.scatter(polyReg_df['Actual Values'],polyReg_df.index,label = 'Actual Values')

plt.scatter(polyReg_df['Predicted Values'],polyReg_df.index,label = 'Predicted Values')

plt.legend()
polyReg_round = polyReg_df.copy()

polyReg_round['Predicted Values'] = np.absolute(np.round(polyReg_df['Predicted Values'],0))

polyReg_round.head()
ax1 = sns.distplot(polyReg_round['Actual Values'], hist = False, color = 'r', label = 'Actual Value')

sns.distplot(polyReg_round['Predicted Values'], hist = False, color = 'b', label = 'Fitted Value' , ax = ax1)
print('MSE (non-rounded):',mean_squared_error(Y,y_poly_predict))

print('MSE (rounded):',mean_squared_error(Y,polyReg_round['Predicted Values']))

print('R-Squared :',poly_lm.score(x_poly,Y))
# Instead of using the Rounding Off, let's try clustering for the same
from sklearn.cluster import KMeans 

from sklearn import metrics 

from scipy.spatial.distance import cdist 
# Getting the optimum cluster value

X = polyReg_df

distortions = [] 

inertias = [] 

mapping1 = {} 

mapping2 = {} 

K = range(1,10) 

  

for k in K: 

    #Building and fitting the model 

    kmeanModel = KMeans(n_clusters=k)

    kmeanModel.fit(X)     

      

    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 

                      'euclidean'),axis=1)) / X.shape[0]) 

    inertias.append(kmeanModel.inertia_) 

  

    mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_, 

                 'euclidean'),axis=1)) / X.shape[0] 

    mapping2[k] = kmeanModel.inertia_ 

plt.plot(K, distortions, 'bx-') 

plt.xlabel('Values of K') 

plt.ylabel('Distortion') 

plt.title('The Elbow Method using Distortion') 

plt.show() 
# The elbow is at 3

knn_mod = KMeans(n_clusters = 3)

knn_mod.fit(polyReg_df)
knn_df = polyReg_df.copy()

knn_df['Clusters'] = knn_mod.labels_

knn_df['Clusters'].replace({2:1,1:2},inplace = True)
knn_df.head()
plt.scatter(knn_df['Predicted Values'],knn_df.index,c = knn_df['Clusters'],label = 'Predicted Values')

plt.scatter(knn_df['Actual Values'],knn_df.index,c = 'black', label = 'Actual Values')

# plt.scatter(df_temp['y_rounded'],df_temp.index,marker = 'x')

plt.legend()
print('MSE :',mean_squared_error(knn_df['Actual Values'],knn_df['Clusters']))
print('---------------------------------------------------------------------------------------------------------')

print(x_test.head())

print('---------------------------------------------------------------------------------------------------------')

print(y_test.head())

print('---------------------------------------------------------------------------------------------------------')
# Encoding the Test Target Data

y_test_cat = y_test.copy()

lb_species = LabelEncoder()

y_test_cat['Species'] = lb_species.fit_transform(y_test_cat['Species'])

y_test_cat.head()
# Fitting Polynomial Features for the Test X data

x_poly_test = poly_features.fit_transform(x_test)
y_poly_test = poly_lm.predict(x_poly_test)
y_val_df = y_test_cat.copy()

y_val_df['y_predicted'] = y_poly_test
ax1 = sns.distplot(y_test_cat, hist = False, color = 'r', label = 'Actual Value')

sns.distplot(y_poly_test, hist = False, color = 'b', label = 'Fitted Value' , ax = ax1)
plt.scatter(y_val_df['Species'],y_val_df.index,label = 'Predicted Values')

plt.scatter(y_val_df['y_predicted'],y_val_df.index, label = 'Actual Values')

# plt.scatter(df_temp['y_rounded'],df_temp.index,marker = 'x')

plt.legend()
knn_test_mod = KMeans(n_clusters=3)

knn_test_mod.fit(y_val_df[['y_predicted']])

knn_pred_df = y_val_df.copy()

knn_pred_df['Clusters'] = knn_test_mod.labels_
plt.scatter(knn_pred_df['Species'],knn_pred_df.index,label = 'Predicted Values')

plt.scatter(knn_pred_df['y_predicted'],knn_pred_df.index,c = knn_pred_df['Clusters'], label = 'Actual Values')

# plt.scatter(df_temp['y_rounded'],df_temp.index,marker = 'x')

plt.legend()
sns.scatterplot(data=knn_pred_df, x='Species', y='y_predicted', hue='Clusters')
group_data = knn_pred_df.groupby(['Species','Clusters'])

# for key in gb.groups.keys():

#     print(key,':',gb.get_group(key).count().values[0])
# Getting unique values

species_vals = knn_pred_df['Species'].unique()

clusters_vals = knn_pred_df['Clusters'].unique()
max_vals = {}

for key in group_data.groups.keys():

    max_vals[key[0]] = 0



map_dict = max_vals.copy()



for key in group_data.groups.keys():

#     print(group_data.get_group(key).count().values[0])

    if max_vals[key[0]] < group_data.get_group(key).count().values[0]:

        max_vals[key[0]] = group_data.get_group(key).count().values[0]

        map_dict[key[0]] = key
mapping = {}

for tup in list(map_dict.values()):

    mapping[tup[1]] = tup[0]

mapping
knn_pred_df['Clusters'].replace(mapping,inplace = True)
sns.scatterplot(data=knn_pred_df, x='Species', y='y_predicted', hue='Clusters')
ax1 = sns.distplot(knn_pred_df['Clusters'], hist = False, color = 'b', label = 'Fitted Value' ,kde_kws=dict(linewidth=5,shade = True,alpha = 0.5))

sns.distplot(knn_pred_df['Species'], hist = False, color = 'r', label = 'Actual Value', ax = ax1,kde_kws=dict(linewidth=2))
print('MSE :',mean_squared_error(knn_pred_df['Species'],knn_pred_df['Clusters']))
# Checking the MSE if we used the Rounding Off logic

print('MSE :',mean_squared_error(knn_pred_df['Species'],np.absolute(np.round(knn_pred_df['Clusters'],0))))
ax1 = sns.distplot(np.absolute(np.round(knn_pred_df['Clusters'],0)), hist = False, color = 'b', label = 'Fitted Value' ,kde_kws=dict(linewidth=5,shade = True,alpha = 0.5))

sns.distplot(knn_pred_df['Species'], hist = False, color = 'r', label = 'Actual Value', ax = ax1,kde_kws=dict(linewidth=2))
from sklearn.metrics import accuracy_score
acc_score = accuracy_score(knn_pred_df['Species'],knn_pred_df['Clusters'])
acc_score
from sklearn.metrics import confusion_matrix

confusion_mat = pd.DataFrame(confusion_matrix(knn_pred_df['Species'],knn_pred_df['Clusters']),index = [0,1,2], columns = [0,1,2])
confusion_mat