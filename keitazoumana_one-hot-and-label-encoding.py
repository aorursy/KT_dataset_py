import pandas as pd  

import matplotlib.pyplot as plt

from sklearn import preprocessing 
# Read the dataset 

tent_sales = pd.read_csv('../input/sales_data.csv')  

tent_sales.head()
tent_sales.shape
tent_sales.describe()
plt.figure(figsize=(10, 7))

pd.value_counts(tent_sales['IS_TENT']).plot.bar()

plt.show()
plt.figure(figsize=(10, 7))

pd.value_counts(tent_sales['MARITAL_STATUS']).plot.bar()

plt.show()
plt.figure(figsize=(10, 7))

pd.value_counts(tent_sales['GENDER']).plot.bar()

plt.show()
plt.figure(figsize=(10, 7))

pd.value_counts(tent_sales['PROFESSION']).plot.bar()

plt.show()
# Gender Column 

gender = ['M','F'] 



label_encoding = preprocessing.LabelEncoder()



'''

This will generate a unique ID for Male and 

a unique ID for Female. 

'''

label_encoding = label_encoding.fit(gender)
tent_sales['GENDER'] = label_encoding.transform(tent_sales['GENDER'].astype(str))
# Shows the categories that have been encoded.

label_encoding.classes_
tent_sales.sample(10)
# Marital Status Column 

tent_sales['MARITAL_STATUS'].unique()
one_hot_encoding = preprocessing.OneHotEncoder()

one_hot_encoding = one_hot_encoding.fit(tent_sales['MARITAL_STATUS'].values.reshape(-1, 1))
one_hot_encoding.categories_
one_hot_labels = one_hot_encoding.transform(

                tent_sales['MARITAL_STATUS'].values.reshape(-1,1)).toarray()

one_hot_labels
labels_df = pd.DataFrame()



labels_df['MARITAL_STATUS_Married'] = one_hot_labels[:,0]

labels_df['MARITAL_STATUS_Single'] = one_hot_labels[:,1]

labels_df['MARITAL_STATUS_Unspecified'] = one_hot_labels[:,2]



labels_df.head(10)
encoded_df = pd.concat([tent_sales, labels_df], axis=1)

encoded_df.drop('MARITAL_STATUS', axis=1, inplace=True)
encoded_df.sample(10)
# Marital Status Column 

tent_sales['PROFESSION'].unique()
tent_sales = pd.get_dummies(encoded_df, columns=['PROFESSION'])

tent_sales.sample(10)
# Read the dataset 

tent_sales = pd.read_csv('../input/sales_data.csv')  

tent_sales.head()
'''

Encode all of the categorical features while removing 

all the original categorical features.

'''

tent_sales = pd.get_dummies(tent_sales)

tent_sales.sample(10)