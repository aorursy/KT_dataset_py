# loading required libraries
import featuretools as ft
import numpy as np
import pandas as pd

train = pd.read_csv("../input/Train.csv")
test = pd.read_csv("../input/Test.csv")
# Data preperation

# saving identifiers
test_Item_Identifier = test['Item_Identifier']
test_Outlet_Identifier = test['Outlet_Identifier']

sales = train['Item_Outlet_Sales']
train.drop(['Item_Outlet_Sales'], axis=1, inplace=True)
#combine the train and test set as it saves us the trouble of performing the same step(s) twice.

combi = train.append(test, ignore_index=True)
combi.head()
combi['Outlet_Size'].value_counts()
combi.isnull().sum()
# imputing missing data
combi['Item_Weight'].fillna(combi['Item_Weight'].mean(), inplace = True)
combi['Outlet_Size'].fillna("missing", inplace = True)
combi.isnull().sum()
# Data processing

combi['Item_Fat_Content'].value_counts()
# dictionary to replace the categories
fat_content_dict = {'Low Fat':0, 'Regular':1, 'LF':0, 'reg':1, 'low fat':0}

combi['Item_Fat_Content'] = combi['Item_Fat_Content'].replace(fat_content_dict, regex=True)
#perform automated feature engineering! 
#It is necessary to have a unique identifier feature in the dataset
# so first we will create a unique identifier

combi['id'] = combi['Item_Identifier'] + combi['Outlet_Identifier']

combi.drop(['Item_Identifier'], axis=1, inplace=True)


#creating an EntitySet. 
#An EntitySet is a structure that contains multiple dataframes and relationships between them. 


# creating and entity set 'es'
es = ft.EntitySet(id = 'sales')

# adding a dataframe 
es.entity_from_dataframe(entity_id = 'bigmart', dataframe = combi, index = 'id')
#Our data contains information at two levels—item level and outlet level.
#Featuretools offers a functionality to split a dataset into multiple tables. 
#so created a new table ‘outlet’ from the BigMart table based on the outlet ID Outlet_Identifier.

es.normalize_entity(base_entity_id='bigmart', new_entity_id='outlet', index = 'Outlet_Identifier', 
additional_variables = ['Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type'])
print(es)
#Now we will use Deep Feature Synthesis to create new features automatically.
#DFS uses Feature Primitives to create features using multiple tables present in the EntitySet.

feature_matrix, feature_names = ft.dfs(entityset=es, 
target_entity = 'bigmart', 
max_depth = 2, 
verbose = 1, 
n_jobs = 3)
#It has generated a bunch of new features on its own.

#Let’s have a look at these newly created features.
feature_matrix.columns
feature_matrix.head()
#There is one issue with this dataframe – it is not sorted properly. 
#So We will have to sort it based on the id variable from the combi dataframe.

feature_matrix = feature_matrix.reindex(index=combi['id'])
feature_matrix = feature_matrix.reset_index()


#Now the dataframe feature_matrix will be in proper order.
#let's check
feature_matrix.head()
# using cataboost algorithm

from catboost import CatBoostRegressor

#atBoost requires all the categorical variables to be in the string format. 
#so, we will convert the categorical variables in our data to string first:

categorical_features = np.where(feature_matrix.dtypes == 'object')[0]

for i in categorical_features:
    feature_matrix.iloc[:,i] = feature_matrix.iloc[:,i].astype('str')
#Let’s split feature_matrix back into train and test sets.

feature_matrix.drop(['id'], axis=1, inplace=True)
train = feature_matrix[:8523]
test = feature_matrix[8523:]
# removing uneccesary variables
train.drop(['Outlet_Identifier'], axis=1, inplace=True)
test.drop(['Outlet_Identifier'], axis=1, inplace=True)
# identifying categorical features
categorical_features = np.where(train.dtypes == 'object')[0]
#Now splitting the train data into training and validation set to check the model’s performance locally.

from sklearn.model_selection import train_test_split

# splitting train data into training and validation set
xtrain, xvalid, ytrain, yvalid = train_test_split(train, sales, test_size=0.25, random_state=11)
#we can now train our model. The evaluation metric we will use is RMSE (Root Mean Squared Error).

model_cat = CatBoostRegressor(iterations=100, learning_rate=0.3, depth=6, eval_metric='RMSE', random_seed=7)

# training model
model_cat.fit(xtrain, ytrain, cat_features=categorical_features, use_best_model=True)
# validation score
model_cat.score(xvalid, yvalid)
