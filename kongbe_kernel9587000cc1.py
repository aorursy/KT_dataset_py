# Main libraries

import pandas as pd

import numpy as np



# Visual libraries

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D

from sklearn.impute import  SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
### Start pre processing data



# df = pd.read_csv("C:/Users/Kong Be/Desktop/weatherAUS.csv")

df = pd.read_csv("../input/weatherAUS.csv")

df = df.drop(columns ='RISK_MM')  # delete feature RISK_MM
df.head(n=5)
# heat map of correlation of features

correlation_matrix = df.corr()

fig = plt.figure( num = 'correlation of features', figsize = (10,8))

sns.heatmap( correlation_matrix , vmax = 0.8 , square = True )

plt.show()
# Chose features that correlation confetic >0.9

features_chose = []

height = len(correlation_matrix.index)

for i in range(height):

   for j in range(height):

          if correlation_matrix.iloc[i,j] < 1 and correlation_matrix.iloc[i,j] > 0.9:

              features_chose.extend([correlation_matrix.columns.values[j] , correlation_matrix.columns.values[i]])

# Delete duplicates features

target_features = []

for i in features_chose:

    if features_chose.count(i) > 1 :

        features_chose.remove(i)

        target_features.append(i)

print(target_features)

#target_features.append('RainTomorrow')

df1 = df.loc[:,target_features]

df1.loc[:,'RainTomorrow'] = df.loc[:,'RainTomorrow'].values

df1.loc[:,'RainTomorrow'].replace(['Yes','No'],[1,0] , inplace = True)

print(" \n\t\tFive examples of Data Frame")

print(df1.head())

print(len(df1.columns.values))
#Standardize the data

x = df1.loc[:,target_features].values

y = df1.loc[:,'RainTomorrow'].values

x = StandardScaler().fit_transform(x)



df1 = pd.DataFrame(data = x , columns = target_features )

# Checking Missing values --> count missing values

missing_matrix = df1.isnull()

count_missingValues = df1.isnull().sum().sum()

print(count_missingValues)
# Xoá các giá trị NA ( không phải số) để áp dụng đc hàm PCA.fit_transform bằng hàm SimpleImputer

# (chuyển các NA thành các mean trên từng column)

imp = SimpleImputer(missing_values=np.nan, strategy='mean')

x = imp.fit_transform(x)

#Create a new Dataframe

df1_new = pd.DataFrame(data = x , columns = target_features )

print("Count of the missing values in Data Frame now is ",end = '')

print(df1.isnull().sum().sum())
#PCA transformation to 2D

pca = PCA(n_components = 2)

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents , columns = ['principal component 1','principal component 2'])

print(principalDf.head())
#Visualization 2D

fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1)

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 component PCA', fontsize = 20)

targets = [0, 1]

colors = ['r', 'g']

for target, color in zip(targets,colors):

    indicesToKeep = y == target

    ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1']

               , principalDf.loc[indicesToKeep, 'principal component 2']

               , c = color

               , s = 50)

ax.legend(targets)

ax.grid()

plt.show()
# Data Splitting

x_train , x_test , y_train , y_test = train_test_split(principalDf, y , train_size = .85)

print(x_train.shape)
print(x_test.shape)
# Apply LogisticRegression

logisticRegr = LogisticRegression(solver='lbfgs')



result = logisticRegr.predict(x_test)
logisticRegr.fit(x_train,y_train)
result = logisticRegr.predict(x_test)

print(result)

print("20 predicted first labels")

print(result[range(20)])
score = logisticRegr.score(x_test,y_test)

print(score)