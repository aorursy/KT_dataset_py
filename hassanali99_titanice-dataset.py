#Import these packages 

import pandas as pd 



#Import packages to visualize the data set



import pandas as pd

import numpy as np

import seaborn as sn

import matplotlib.pyplot as plt



#normalization and standarization

from sklearn.preprocessing import Normalizer

from sklearn.preprocessing import StandardScaler
#assaign the train and test datasets to different variables



train_dataset = "../input/titanic/train.csv"

test_dataset_x = "../input/titanic/test.csv"

test_dataset_y = "../input/titanic/gender_submission.csv"
#read the csv files and print the sample rows from top, bottom and middle of the dataframe



df_train = pd.read_csv(train_dataset,delimiter= ',') 

df_test_x = pd.read_csv(test_dataset_x, delimiter = ',')

df_test_y = pd.read_csv(test_dataset_y, delimiter = ',')



df_train.head()
df_train.sample()
df_test_x.head()
df_test_x.sample()
df_train.info()
df_test_x.info()

df_test_y.info()
df_train.describe()
#Dropping redundat columns



to_drop = ['Name', 'SibSp', 'Parch','Ticket', 'Cabin']



df_train.drop(to_drop, inplace=True, axis=1)

df_test_x.drop(to_drop, inplace=True, axis=1)
df_train.head()
df_test_x.head()
#Check if passengerId is a unique value

df_train['PassengerId'].is_unique
#Set passenger ID to be the index of the data 

df_train = df_train.set_index('PassengerId')

df_test_x = df_test_x.set_index('PassengerId')

df_test_y = df_test_y.set_index('PassengerId')
df_test_x.head()

df_test_y.head()
#Visualizing the correlation between the training dataset 



corrMatrix = df_train.corr()

sn.heatmap(corrMatrix, annot=True)

plt.show()
missing_val_count_by_column_train = (df_train.isnull().sum())
print(missing_val_count_by_column_train)

clean_train = df_train.dropna(axis = 0, how ='any')  
missing_val_count_by_column_train = (clean_train.isnull().sum())

print(missing_val_count_by_column_train)
clean_train.info()
# my_inds = [2,3,5]

# somdata = clean_train.drop(my_inds)
index_age = df_test_x['Age'].index[df_test_x['Age'].apply(np.isnan)]

index_fare = df_test_x['Fare'].index[df_test_x['Fare'].apply(np.isnan)]
lst_index_age = index_age.tolist()

lst_index_fare = index_fare.tolist()

print(index_age,lst_index_fare)

print(type(lst_index_age))

print(type(lst_index_fare))
df_test_x = df_test_x.drop(lst_index_age)

df_test_x = df_test_x.drop(lst_index_fare)



df_test_y = df_test_y.drop(lst_index_age)

df_test_y = df_test_y.drop(lst_index_fare)



print(df_test_x.info())

print(df_test_y.info())
missing_val_count_by_column_train = (clean_train.isnull().sum())

print(missing_val_count_by_column_train)
# fare_train = clean_train['Fare']

# age_train = clean_train['Age']



# fare_train = clean_train.to_numpy()

# age_train = clean_train.to_numpy()





# fare_test = df_test_x['Fare']

# age_test = df_test_x['Age']



# fare_test = fare_test.to_numpy()

# age_test = age_test.to_numpy()
print(type(clean_train['Fare']))
clean_train.head()
cols_to_norm_train = clean_train.iloc[:, 3:5]



cols_to_norm_train = cols_to_norm_train.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
df_test_x.iloc[:, 2:4] = df_test_x.iloc[:, 2:4].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
clean_train.head()
df_test_x.head()
from sklearn.neighbors import KNeighborsClassifier







# Create KNN classifier



# knn = KNeighborsClassifier(n_neighbors = 3)



x_train = clean_train.iloc[:, 1:6]

y_train = clean_train.iloc[:, 0:1]



print(type(y_train))



from sklearn.linear_model import LogisticRegression



logReg = LogisticRegression()

logReg.fit(x_train,y_train)

acc = logReg.score(x_train,y_train)

acc