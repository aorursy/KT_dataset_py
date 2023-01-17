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
df = pd.read_csv("../input/inmueblesanalisis/ar_properties.csv")
df.isnull().sum()
df.describe()
df[["l1","l2","l3","l4"]].isnull().sum()


complete_prices_df = df.iloc[(df[["currency","price"]].dropna()).index,:]

complete_prices_df
#re-define indices

complete_prices_df = complete_prices_df.reset_index(drop=True)
categorical_cols = (complete_prices_df.dtypes[complete_prices_df.dtypes=="object"]).index

numerical_cols = (complete_prices_df.dtypes[complete_prices_df.dtypes=="float64"]).index
categorical_cols = pd.Series(categorical_cols)

categorical_cols
numerical_cols = (numerical_cols[numerical_cols != "l6"])

numerical_cols = pd.Series(numerical_cols)

numerical_cols
numerical_cols = pd.Series(numerical_cols).append(pd.Series(["currency"]))
complete_prices_df["operation_type"].value_counts()
dolar_only = complete_prices_df[(complete_prices_df["currency"]=="USD") & (complete_prices_df["operation_type"]=="Venta")].reset_index(drop=True)

lean_df1 = dolar_only[numerical_cols].dropna().reset_index(drop=True)
lean_df1
X = lean_df1.drop(axis=1,columns=["currency","price"])

y = lean_df1["price"]
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor



train_X_numerical, val_X_numerical, train_y_numerical, val_y_numerical = train_test_split(X,y,random_state=9)

#model training

gbr_numerical_notlabeled = GradientBoostingRegressor(random_state=1).fit(train_X_numerical,train_y_numerical)
from sklearn.metrics import mean_absolute_error

prediction1 = gbr_numerical_notlabeled.predict(val_X_numerical)

mean_absolute_error(val_y_numerical,prediction1)
dolar_only_features_to_impute = dolar_only[numerical_cols].drop(axis=1,columns=["currency","price"])

#so we use now the dataframe with all the rows with no missing price nor currency, we will impute it ONLY ON THE FEATURES. 

#There´s no apparent reason to impute on the target in this case. There´s a very big variance between prices and its other features.
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer().fit(dolar_only_features_to_impute)

new_X = pd.DataFrame(my_imputer.transform(dolar_only_features_to_impute),columns = [dolar_only_features_to_impute.columns])
len(new_X.index)==len(dolar_only["price"].index) #to make sure we still have the same number of rows in target and features
new_y = dolar_only["price"]
tr_X,val_X,tr_y,val_y = train_test_split(new_X,new_y,random_state=3)
from sklearn.ensemble import RandomForestRegressor


fst_rfr = RandomForestRegressor(random_state=1).fit(tr_X,tr_y)

preds_fst_rfr = fst_rfr.predict(val_X)

mean_absolute_error(val_y,preds_fst_rfr)
snd_gbr = GradientBoostingRegressor(random_state=1).fit(tr_X,tr_y)

preds_snd_gbr = snd_gbr.predict(val_X)

mean_absolute_error(val_y,preds_snd_gbr)
dolar_only_features_to_impute.isnull().sum().sort_values()
dolar_only_features_to_impute.columns
print((dolar_only_features_to_impute["rooms"]>6).mean())

print((dolar_only_features_to_impute["bathrooms"]>3).mean())

print((dolar_only_features_to_impute["bedrooms"]>3).mean())

#los valores con cuartos, baños y habitaciones con numeros raramente grandes son insignificantes.

#vale la pena imputar sus valores para no perder datos de otras columnas
df.describe()
print(df["l2"].value_counts())
df_ar = df[df["l1"]=="Argentina"]

df_ar_gbaNor = df_ar[df_ar["l2"]=="Bs.As. G.B.A. Zona Norte"]

print(df_ar_gbaNor["l3"].isnull().sum())

df_ar_gbaNor["l3"].value_counts()

df_ar_gbaNor_sell = df_ar_gbaNor[df_ar_gbaNor["operation_type"]=="Venta"]
df_ar_gbaNor_sell_in_dollars = df_ar_gbaNor_sell[df_ar_gbaNor_sell["currency"]=="USD"]

df_ar_gbaNor_sell_in_dollars
#here two variables from the untidy section are taken, they are just the names of the columns that are numeric and the columns that are categorical

print(categorical_cols)

numerical_cols
df_ar_gbaNor_sell_in_dollars[categorical_cols[5:10]]
df_ar_gbaNor_sell_in_dollars["l5"].value_counts()
print(df_ar_gbaNor_sell_in_dollars["l4"].isnull().sum()/df_ar_gbaNor_sell_in_dollars.shape[0])

print(df_ar_gbaNor_sell_in_dollars["l3"].isnull().sum()/df_ar_gbaNor_sell_in_dollars.shape[0])

#32% of the values in l4 (neighborhood) are missing, meanwhile 2% of the l3 (district) column are missing 

#we will consider imputation for these locations
relevant_categorical_cols = categorical_cols.iloc[[4,7,8,14]]
relevant_categorical_cols
relevant_numerical_cols = numerical_cols[0:8].drop([3,4])

relevant_numerical_cols
relevant_cols = relevant_categorical_cols.append(relevant_numerical_cols)

relevant_cols
df_ar_gbaNor_sell_in_dollars[relevant_cols].isnull().sum().div(df_ar_gbaNor_sell_in_dollars.shape[0])*100
df_ar_gbaNor_sell_in_dollars_loc_not_missing = df_ar_gbaNor_sell_in_dollars[df_ar_gbaNor_sell_in_dollars["lat"].notnull() & df_ar_gbaNor_sell_in_dollars["lon"].notnull() & df_ar_gbaNor_sell_in_dollars["l3"].notnull()]

df_ar_gbaNor_sell_in_dollars_loc_not_missing[relevant_cols].isnull().sum().div(df_ar_gbaNor_sell_in_dollars_loc_not_missing.shape[0])*100
relevant_cols_no_l4 = relevant_cols.drop([8,4,5])

relevant_cols_no_l4.drop(index=7)
df_ar_gbaNor_sell_in_dollars_loc_not_missing[relevant_cols_no_l4].isnull().sum().div(df_ar_gbaNor_sell_in_dollars_loc_not_missing.shape[0])
relevant_cols_no_l4
X = df_ar_gbaNor_sell_in_dollars_loc_not_missing[relevant_cols_no_l4.drop(index=7)]

y = df_ar_gbaNor_sell_in_dollars_loc_not_missing["price"][X.index]
X = X.reset_index(drop=True)

y = y.reset_index(drop=True)
X["surface_covered"].describe()
X.columns
from sklearn.impute import SimpleImputer



imputer = SimpleImputer()

imputed_cols = imputer.fit_transform(X[["rooms","surface_covered"]])
X_imputed_columns_df = pd.DataFrame(imputed_cols,columns=["rooms","surface_covered"])

X_imputed = X
X_imputed[["rooms","surface_covered"]] = X_imputed_columns_df
X_imputed
from sklearn.model_selection import train_test_split



train_X_imputed, test_X_imputed, train_y, test_y = train_test_split(X_imputed,y, test_size = 0.15,random_state=19)
from sklearn.preprocessing import OneHotEncoder

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

property_type_train_oh_encoded = pd.DataFrame(OH_encoder.fit_transform(train_X_imputed[["property_type"]]))

property_type_test_oh_encoded = pd.DataFrame(OH_encoder.transform(test_X_imputed[["property_type"]]))



property_type_train_oh_encoded.index = train_X_imputed.index

property_type_test_oh_encoded.index= test_X_imputed.index



property_type_train_oh_encoded
train_X_imputed_no_property_type = train_X_imputed.drop(["property_type"],axis="columns")

test_X_imputed_no_property_type = test_X_imputed.drop(["property_type"],axis="columns")



train_X_imputed_no_property_type
labeled_train_X_imputed = pd.concat([train_X_imputed_no_property_type,property_type_train_oh_encoded],axis=1)

labeled_train_X_imputed
labeled_train_X_imputed
labeled_test_X_imputed = pd.concat([test_X_imputed_no_property_type,property_type_test_oh_encoded],axis=1)
from sklearn.ensemble import RandomForestRegressor



fst_random_forest = RandomForestRegressor(random_state=4).fit(labeled_train_X_imputed,train_y)
fst_preds = fst_random_forest.predict(labeled_test_X_imputed)
from sklearn.metrics import mean_absolute_error

print("The error percentage is " + str(mean_absolute_error(test_y,fst_preds)*100/test_y.mean())+"%")

mean_absolute_error(test_y,fst_preds)
from sklearn.ensemble import GradientBoostingRegressor



snd_gradient_booster = GradientBoostingRegressor().fit(labeled_train_X_imputed,train_y)
snd_preds = snd_gradient_booster.predict(labeled_test_X_imputed)

print("The error percentage is " + str(mean_absolute_error(test_y,snd_preds)*100/test_y.mean()) + "%")

mean_absolute_error(test_y,snd_preds)
import eli5

from eli5.sklearn import PermutationImportance

perm=PermutationImportance(fst_random_forest,random_state=1).fit(labeled_test_X_imputed,test_y)

colnames = labeled_test_X_imputed.columns.tolist()

colnames

for k in range(len(colnames)):

    colnames[k] = str(colnames[k])

colnames
eli5.show_weights(perm , feature_names = colnames )
from sklearn.metrics import mean_absolute_error

for k in [10,1000,100000,10000000]:

    random_forest = RandomForestRegressor(random_state=4,max_leaf_nodes = k).fit(labeled_train_X_imputed,train_y)

    predictions = random_forest.predict(labeled_test_X_imputed)

    mae = mean_absolute_error(test_y,predictions)

    print("error: " + str(mae) + "; leafnodes: " + str(k) + " \n ")

    
