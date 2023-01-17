# 1.0 Clear memory

%reset -f
# 1.1 Call data manipulation libraries

import pandas as pd

import numpy as np
# 1.2 Feature creation libraries

from sklearn.random_projection import SparseRandomProjection as sr  # Projection features

from sklearn.cluster import KMeans                    # Cluster features

from sklearn.preprocessing import PolynomialFeatures  # Interaction features
# 1.3 For feature selection

# Ref: http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import mutual_info_classif  # Selection criteria
# 1.4 Data processing

# 1.4.1 Scaling data in various manner

from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale

# 1.4.2 Transform categorical (integer) to dummy

from sklearn.preprocessing import OneHotEncoder# 1.5 Splitting data

from sklearn.model_selection import train_test_split
# 1.5 Splitting data

from sklearn.model_selection import train_test_split
# 1.6 Decision tree modeling

# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree

# http://scikit-learn.org/stable/modules/tree.html#tree

from sklearn.tree import  DecisionTreeClassifier as dt

# 1.7 RandomForest modeling

from sklearn.ensemble import RandomForestClassifier as rf
# 1.8 Plotting libraries to plot feature importance

import matplotlib.pyplot as plt

import seaborn as sns
# 1.9 Misc

import os, time, gc
################## AA. Reading data from files and exploring ####################



# 2.0 Set working directory and read file

#os.chdir("C:\\Users\\ashok\\OneDrive\\Documents\\xgboost\\Heartdisease")

#os.chdir("C:\\Users\\admin\\Documents\\python\\Heartdisease")

#os.listdir()
# 2.1 Read heart files

#heart = pd.read_csv("../input/heart.csv")

heart = pd.read_csv("../input/heart.csv")

#heart = pd.read_csv("heart.csv")
# 2.2 Look at data

heart.head(2)
heart.shape
# 2.3 Data types

heart.dtypes.value_counts() 
# 2.4 Target classes are almost balanced

heart.target.value_counts()
# 2.6.1 Sum each row, and check which one of them is 0

#       axis = 1 ==> Across columns

x = np.sum(heart, axis = 1)

x = (x == 0)

x.head()
# 2.6.2 So, which index number is 0

# Ref: https://stackoverflow.com/questions/52173161/getting-a-list-of-indices-where-pandas-boolean-series-is-true

v = np.nonzero(x)[0]       # Which value is True

v       
# 2.6.3 Drop this row from test data

heart.drop(v, axis = 0, inplace = True)

heart.shape
# 3 Check if there are Missing values? None

heart.isnull().sum().sum()
#  4. Feature 1: Row sums of features 1:93. More successful

#                when data is binary.

heart['sum'] = heart.sum(numeric_only = True, axis=1)
# 4.1 Assume that value of '0' in a cell implies missing feature

#     Transform train and test dataframes

#     replacing '0' with NaN

#     Use pd.replace()

tmp_heart = heart.replace(0, np.nan)
# 4.2 Check if tmp_train is same as train or is a view

#     of train? That is check if tmp_train is a deep-copy



tmp_heart is heart
#tmp_train is train.values.base    # False

tmp_heart._is_view 
# 4.3 Check if 0 has been replaced by NaN

tmp_heart.head(1)
# 5. Feature 2 : For every row, how many features exist

#                that is are non-zero/not NaN.

#                Use pd.notna()

tmp_heart.notna().head(1)

heart["count_not0"] = tmp_heart.notna().sum(axis = 1)
# 6. Similary create other statistical features

#    Feature 3

#    Pandas has a number of statistical functions

#    Ref: https://pandas.pydata.org/pandas-docs/stable/reference/frame.html#computations-descriptive-stats



feat = [ "var", "median", "mean", "std", "max", "min"]

for i in feat:

    heart[i] = tmp_heart.aggregate(i,  axis =1)
# 7 Delete and release memory

del(tmp_heart)

gc.collect()
# 7.1 So what do we have finally

heart.shape                

heart.head(1)

heart.head(2)
# 11. Generate features using random projections

#     First stack train and test data, one upon another

tmp = pd.concat([heart],

                axis = 0,            # Stack one upon another (rbind)

                ignore_index = True

                )
tmp.shape 
# 13. Let us create 10 random projections/columns

#     This decision is arbitrary

NUM_OF_COM = 10 #need tuned

# 13.1 Create an instance of class

rp_instance = sr(n_components = NUM_OF_COM)

# 13.2 Transform the dataset

rp = rp_instance.fit_transform(tmp.iloc[:, :93])
rp
# 13.3 Transfrom resulting array to pandas dataframe

#      Also assign column names

rp = pd.DataFrame(rp, columns = ['r1','r2','r3','r4', 'r5', 'r6', 'r7', 'r8', 'r9', 'r10'] )

rp.head(2)
# 15. Before clustering, scale data

# 15.1 Create a StandardScaler instance

se = StandardScaler()

# 15.2 fit() and transform() in one step

tmp = se.fit_transform(tmp)

# 15.3

tmp.shape 
# 16. Perform kmeans using 93 features.

#     No of centroids is no of classes in the 'target'

centers = target.value_counts().size     # 9 unique classes

centers
# 17.1 Begin clustering

start = time.time()
# 17.2 First create object to perform clustering

kmeans = KMeans(n_clusters=centers, # How many

                n_init=30,          # No of times k-means will run with difft centroid seeds.

                n_jobs = 2)
poly = PolynomialFeatures(2,                      # Degree 2

                          interaction_only=True,  # Avoid e.g. square(a)

                          include_bias = False   # No constant term

                          )
# 21.1 If you skip kmeans, execute the commented statement

df =  poly.fit_transform(tmp[:, : 5])
heart['age'].unique()
index_sum=[]

for age in heart['age'].unique():

    index_sum.append(sum(heart[heart['age']==age].sum)/len(heart[heart['age']==age].sum))

    

plt.bar(['15-24 years','35-54 years','75+ years','25-34 years','55-74 years','5-14 years'],index_sum,align='center',alpha=0.5)

plt.xticks(rotation=90)

plt.show()
heart_region=heart['age'].value_counts()

heart_rvalues=heart_region.values

heart_rregion=heart_region.index
plt.figure(figsize=(10,10))

sns.barplot(x=heart_rregion,y=heart_rvalues)

plt.xticks(rotation=90)

plt.xlabel('age')

plt.ylabel('sex')

plt.title('age VS sex')

plt.show()
plt.figure(figsize=(10,10))

ax=sns.barplot(x=heart_rregion,y=heart_rvalues,palette=sns.cubehelix_palette(len(heart_rregion)))

plt.xlabel('chol')

plt.ylabel('fbs')

plt.xticks(rotation=90)

plt.title('cholesterol rate')

plt.show()
plt.figure(figsize=(5,5))

plt.scatter(heart['cp'], heart['trestbps'], s=(heart['restecg']**3), alpha=0.5)

plt.grid(True)



plt.xlabel("cp")

plt.ylabel("trestbps")



plt.suptitle("Heart Rates of cp vs trestbps", fontsize=18)



plt.show()
f, ax = plt.subplots(figsize=(6.5, 6.5))

sns.despine(f, left=True, bottom=True)

sns.scatterplot(x="chol", y="thalach",data=heart)

plt.show()
sns.boxplot(x='age',y='sex',data=heart,palette='PRGn')

plt.xticks(rotation=90)

plt.show()
plt.scatter(x=heart['restecg'],y=heart['fbs'],color='r')

plt.scatter(x=heart['restecg'],y=heart['chol'],color='b')

plt.show()