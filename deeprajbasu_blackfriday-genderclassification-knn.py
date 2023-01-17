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
import pandas as pd

df = pd.read_csv("/kaggle/input/black-friday/train.csv")





#limiting the amount of data we use to keep computation fast 

df = df.iloc[:]

df.shape
df.head()
df.describe()
print("Data columns")

print("--------------------------------------------")

print(pd.DataFrame(df.info()))
print(df.isnull().sum())


df.isnull().sum()/df.isnull().count()*100
df = df.drop('Product_Category_3', axis=1)



## also dropping user id category 

df = df.drop('User_ID', axis=1)

df = df.drop('Product_ID', axis=1)
df['Marital_Status'].unique()
df['Product_Category_2'].unique()
len(df['Product_Category_2'].unique())
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imputer = imputer.fit(pd.DataFrame(df['Product_Category_2']))





df['Product_Category_2'] = imputer.transform(pd.DataFrame(df['Product_Category_2']))

#data_train['Product_Category_2'] = np.round(data_train['Product_Category_2'])
imputer = imputer.fit(pd.DataFrame(df['Marital_Status']))

df['Marital_Status'] = imputer.transform(pd.DataFrame(df['Marital_Status']))
len(df['Product_Category_2'].unique())
df.isnull().sum()/df.isnull().count()*100
print("average purchase for male purchasers = ",

np.mean(df['Purchase'].loc[df['Gender'] == 'M']),'\n')

print("-"*115,'\n')

print("average purchase for female purchasers = ",

np.mean(df['Purchase'].loc[df['Gender'] == 'F']))
import matplotlib.pyplot as plt

import seaborn as sns

fig= plt.figure(figsize=(12,7))





sns.set(style="darkgrid")





x = pd.DataFrame({"male average purchase": [9437], "Female average purchase": [8734]})



sns.barplot(data=x)
print('Number of Female purchasers = ',df['Gender'][df['Gender'] == 'F'].count())

print('Number of male purchasers   = ',df['Gender'][df['Gender'] == 'M'].count())
#change gender from 'm' and 'f' to binary 

df.loc[:, 'Gender'] = np.where(df['Gender'] == 'M', 1, 0)



#renaming some columns 

df = df.rename(columns={

                #'Product_ID': 'ProductClass',

                'Product_Category_1': 'Category1',

                'Product_Category_2': 'Category2',

                'City_Category': 'City',

                'Stay_In_Current_City_Years': 'City_Stay'

})

#y = train.pop('Purchase')



df.head()
#len(df['ProductClass'].unique())
# from sklearn.preprocessing import LabelEncoder

# L_encoder =  LabelEncoder()

# for col in ['ProductClass']:    

#     df.loc[:, col] =L_encoder.fit_transform(df[col])

# df[['ProductClass']]
from sklearn.preprocessing import  OneHotEncoder

cats = ['Occupation', 'Age', 'City', 'Category1','Category2','City_Stay']



#creating the encoder, fit it to our data 

encoder = OneHotEncoder().fit(df[cats])

#generating feature names for our encoded data

encoder.get_feature_names(cats)
#building dataframe with encoded catgegoricals 



## we use index values from our original data 

## we GENERATE feature names using our encoder



endcoded_data = pd.DataFrame(encoder.transform(df[cats]).toarray(),index=df.index, columns=encoder.get_feature_names(cats))

endcoded_data.head()
df = pd.concat([df, endcoded_data],sort=False,axis=1)



df=df.drop(cats, axis=1)
df = df.fillna(0)

df.head(15)

X = df.drop('Gender',axis=1)

y = df.pop('Gender')



#X=np.nan_to_num(X)
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size= 0.25)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X_train,y_train)
knn.score(X_test,y_test)
# parameters = { 'algorithm' : ['ball_tree', 'kd_tree', 'brute'],

#                'leaf_size' : [18,20,25,27,30,32,34],

#                'n_neighbors' : [3,5,7,9,10,11,12,13]

#               }



# from sklearn.model_selection import GridSearchCV

# gridsearch = GridSearchCV(knn, parameters,verbose=3)

# gridsearch.fit(X_train,y_train)

# gridsearch.best_params_
knn = KNeighborsClassifier(algorithm = 'auto', leaf_size =35, n_neighbors =5)

knn.fit(X_train,y_train)

knn.score(X_test,y_test)

dft = pd.read_csv("/kaggle/input/black-friday/test.csv")



dft = dft.drop('Product_Category_3', axis=1)



## also dropping user id category 

dft = dft.drop('User_ID', axis=1)

dft = dft.drop('Product_ID', axis=1)



#Product_ID

dft.head()
dft = dft.iloc[:1]

dft
dft = dft.rename(columns={

                #'Product_ID': 'ProductClass',

                'Product_Category_1': 'Category1',

                'Product_Category_2': 'Category2',

                'City_Category': 'City',

                'Stay_In_Current_City_Years': 'City_Stay'

})

dft
# dft['ProductClass']='P00248942'

# dft
#change gender from 'm' and 'f' to binary 

dft['Gender'] = 9851

dft
p =pd.DataFrame(encoder.transform(dft[cats]).toarray(),columns=encoder.get_feature_names(cats))

p
dft=dft.drop(cats, axis=1)

dft
dft = pd.concat([dft, p],sort=False,axis=1)

df.head(1)
dft
# dft['ProductClass'] =L_encoder.transform(dft['ProductClass'])

# p
knn.predict(dft)