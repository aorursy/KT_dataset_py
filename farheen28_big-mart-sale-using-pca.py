# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Loading data.

train=pd.read_csv("../input/Train.csv")

test=pd.read_csv("../input/Test.csv")
print("Rows and Columns in training set:", train.shape)
print("Rows and columns in testing set: ", test.shape)
#getting familiar with the structure of dataset.

train.columns
for col in train:

    val=train[col].isnull().sum()

    if val>0.0:

        print("Number of missing values in column ",col,":",val)

    #else:

     #   print("Number of values in column ",col,":",85223-val)
for col in test:

    val=test[col].isnull().sum()

    if val>0.0:

        print("Number of missing values in column ",col,":",val)

    #else:

     #   print("Number of values in column ",col,":",5681-val)
train.info()
print(train["Item_Fat_Content"].value_counts())

print(train["Item_Type"].value_counts())

print(train["Outlet_Identifier"].value_counts())

print(train["Outlet_Size"].value_counts())

print(train["Outlet_Location_Type"].value_counts())

print(train["Outlet_Type"].value_counts())
combine=[test, train]

content_mapping = {'Low Fat': 1, 'Regular': 2, 'LF': 3, 'reg': 4, 'low fat': 5}

item_mapping = {'Fruits and Vegetables': 1, 'Snack Foods': 2, 'Household': 3, 'Frozen Foods': 4, 'Dairy': 5, 'Canned':6, 'Baking Goods':7, 'Health and Hygiene':8, 'Soft Drinks':9, 'Meat':10, 'Breads':11, 'Hard Drinks':12, 'Others':13, 'Starchy Foods':14, 'Breakfast':15, 'Seafood':16}

outletIdentifier_mapping ={'OUT027': 27, 'OUT013': 13, 'OUT046': 46, 'OUT035': 35, 'OUT049': 49, 'OUT045':45, 'OUT018':18, 'OUT017':17, 'OUT010':10, 'OUT019':19}

outlet_mapping = {'High': 1, 'Medium': 2, 'Small': 3}

Location_mapping = {'Tier 1': 1, 'Tier 2': 2, 'Tier 3': 3}

Type_mapping = {'Supermarket Type1': 1, 'Supermarket Type2': 2, 'Supermarket Type3': 3, 'Grocery Store':4}

for dataset in combine:

    dataset['Item_Fat_Content'] = dataset['Item_Fat_Content'].map(content_mapping)

    dataset['Item_Fat_Content'] = dataset['Item_Fat_Content'].fillna(0)

    dataset['Item_Type'] = dataset['Item_Type'].map(item_mapping)

    dataset['Item_Type'] = dataset['Item_Type'].fillna(0)

    dataset['Outlet_Size'] = dataset['Outlet_Size'].map(outlet_mapping)

    dataset['Outlet_Size'] = dataset['Outlet_Size'].fillna(0) 

    dataset['Outlet_Identifier'] = dataset['Outlet_Identifier'].map(outletIdentifier_mapping)

    dataset['Outlet_Identifier'] = dataset['Outlet_Identifier'].fillna(0)

    dataset['Outlet_Location_Type'] = dataset['Outlet_Location_Type'].map(Location_mapping)

    dataset['Outlet_Location_Type'] = dataset['Outlet_Location_Type'].fillna(0)

    dataset['Outlet_Type'] = dataset['Outlet_Type'].map(Type_mapping)

    dataset['Outlet_Type'] = dataset['Outlet_Type'].fillna(0)



train.head()

test.head()
train["Item_Identifier"] = train["Item_Identifier"].astype('category')

train.dtypes
#converting categorical data to numeric.

train["Item_Identifier"] = train["Item_Identifier"].cat.codes

train.head()
test["Item_Identifier"] = test["Item_Identifier"].astype('category')

test.dtypes
#converting categorical data to numeric.

test["Item_Identifier"] = test["Item_Identifier"].cat.codes

test.head()
#Overall Correlation

import seaborn as sns

import matplotlib.pyplot as plt

sns.heatmap(train.corr(),annot=True)

plt.show()
pd.crosstab(train['Outlet_Size'], train['Outlet_Identifier'])
for row in train.itertuples(index=True, name='Pandas'):

    if row[9] is None:

        if(row[7]==10 or row[7]==17 or row[7]==45):

            train.loc[row.Index, 'Outlet_Size'] = 0

    else:

        if(row[7]==13):

            train.loc[row.Index, 'Outlet_Size'] =1

        else:

            if(row[7]==18 or row[7]==27 or row[7]==49):

                train.loc[row.Index, 'Outlet_Size'] =2

            else:

                if(row[7]==19 or row[7]==35 or row[7]==46):

                    train.loc[row.Index, 'Outlet_Size'] =3
for row in test.itertuples(index=True, name='Pandas'):

    if row[9] is None:

        if(row[7]==10 or row[7]==17 or row[7]==45):

            test.loc[row.Index, 'Outlet_Size'] = 0

    else:

        if(row[7]==13):

            test.loc[row.Index, 'Outlet_Size'] =1

        else:

            if(row[7]==18 or row[7]==27 or row[7]==49):

                test.loc[row.Index, 'Outlet_Size'] =2

            else:

                if(row[7]==19 or row[7]==35 or row[7]==46):

                    test.loc[row.Index, 'Outlet_Size'] =3
train['Item_Weight'].fillna(train['Item_Weight'].dropna().median(), inplace=True)
test['Item_Weight'].fillna(test['Item_Weight'].dropna().median(), inplace=True)
import sklearn.preprocessing as preprocess

X_train = preprocess.scale(train)

X_test = preprocess.scale(test)
mean_Identifier = train["Item_Identifier"].mean()

mean_Weight = train["Item_Weight"].mean()

mean_Fat_Content = train["Item_Fat_Content"].mean()

mean_Visibility = train["Item_Visibility"].mean()

mean_Type = train["Item_Type"].mean()

mean_MRP = train["Item_MRP"].mean()

mean_OIdentifier = train["Outlet_Identifier"].mean()

mean_Year = train["Outlet_Establishment_Year"].mean()

mean_Size = train["Outlet_Size"].mean()

mean_Location = train["Outlet_Location_Type"].mean()

mean_Type = train["Item_Type"].mean()



print(mean_Identifier)

print(mean_Weight)

print(mean_Fat_Content)

print(mean_Visibility)

print(mean_Type)

print(mean_MRP)

print(mean_OIdentifier)

print(mean_Year)

print(mean_Size)

print(mean_Location)

print(mean_Type)
mean_vector = np.array([[mean_Identifier,mean_Weight,mean_Fat_Content,mean_Visibility,mean_Type,mean_MRP,mean_OIdentifier,mean_Year,mean_Size,mean_Location,mean_Type]])

print('Mean Vector:\n', mean_vector)

mean_vector.shape
X_train=np.array(train.drop("Item_Outlet_Sales", axis=1))
scatter_matrix = np.zeros((11,11))

for i in range(X_train.shape[0]):

    scatter_matrix += ((X_train[i,:].reshape(1,11) - mean_vector).T).dot(X_train[i,:].reshape(1,11)- mean_vector)

    #print(i)

print('Scatter Matrix:\n', scatter_matrix)
#Computing eigenvectors and corresponding eigenvalues.

# eigenvectors and eigenvalues for the from the scatter matrix

eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)



print("Eigen Values:\n",eig_val_sc)

print("Eigen Vectors:\n",eig_vec_sc)
for i in range(len(eig_val_sc)):

    eigvec_sc = eig_vec_sc[:,i].reshape(11,1).T

    

    print('Eigenvector {}: \n{}'.format(i+1, eigvec_sc))

    print('Eigenvalue {} from scatter matrix: {}'.format(i+1, eig_val_sc[i]))

    print(40 * '-')
# Make a list of (eigenvalue, eigenvector) tuples

eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]



# Sort the (eigenvalue, eigenvector) tuples from high to low

eig_pairs.sort(key=lambda x: x[0], reverse=True)



# Visually confirm that the list is correctly sorted by decreasing eigenvalues

for i in eig_pairs:

   print(i[0])