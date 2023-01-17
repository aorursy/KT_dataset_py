import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Start by importing the bq_helper module and calling on the specific active_project and dataset_name for the BigQuery dataset.
import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

dataset = bq_helper.BigQueryHelper(active_project="patents-public-data",
                                   dataset_name="ebi_chembl")
query1 = """ SELECT bei, molregno, tid
            FROM 
                `patents-public-data.ebi_chembl.ligand_eff_23`
                    INNER JOIN
                `patents-public-data.ebi_chembl.activities_23`
                    USING (`activity_id`)
                    INNER JOIN
                `patents-public-data.ebi_chembl.assays_23`
                    USING (`assay_id`)
                    INNER JOIN
                `patents-public-data.ebi_chembl.target_dictionary_23`
                    USING (`tid`)
                    INNER JOIN
                `patents-public-data.ebi_chembl.target_type_23`
                    USING (`target_type`)
            WHERE bei IS NOT NULL AND parent_type = 'PROTEIN'
            """
dataset.estimate_query_size(query1)
main = dataset.query_to_pandas_safe(query1)
del query1
main
query2 = """ SELECT *
            FROM 
                `patents-public-data.ebi_chembl.compound_properties_23`
            """
dataset.estimate_query_size(query2)
mol = dataset.query_to_pandas_safe(query2)
del query2
mol = mol.drop(columns=['mw_freebase', 'ro3_pass', 'molecular_species', 'mw_monoisotopic', 'full_molformula'])
mol 
query3 = """ SELECT tid, component_id
            FROM 
                `patents-public-data.ebi_chembl.target_components_23`
            """
dataset.estimate_query_size(query3)
tar_comp = dataset.query_to_pandas_safe(query3)
del query3
tar_comp.describe()
query4 = """ SELECT component_id, protein_class_id
            FROM 
                `patents-public-data.ebi_chembl.component_class_23`
            """
dataset.estimate_query_size(query4)
comp_class = dataset.query_to_pandas_safe(query4)
del query4
comp_class.describe() 
query5 = """ SELECT tid, target_type
            FROM 
                `patents-public-data.ebi_chembl.target_dictionary_23`
            """
dataset.estimate_query_size(query5)
tar = dataset.query_to_pandas_safe(query5)
del query5
tar
tar = tar.merge(tar_comp, on='tid', how='inner').merge(comp_class, on='component_id', how = 'inner').drop(columns='component_id').drop_duplicates()
del tar_comp, comp_class
tar.describe()
test = tar.groupby('tid')['protein_class_id'].apply(list) # Так как один белок может принадлежать к различным классам, то делаем списки классов
tar = tar.drop(columns='protein_class_id')
result = main.merge(mol, on='molregno', how='inner')
result = result.merge(tar, on='tid', how='inner')
result = result.merge(test.to_frame(), on='tid', how='inner')
result = result.dropna() # Избавляемся от строк с недостающими данными 
del test, tar, main, mol
result
from sklearn.preprocessing import MultiLabelBinarizer
s = result['protein_class_id']
mlb = MultiLabelBinarizer()
classes = pd.DataFrame(mlb.fit_transform(s),columns=mlb.classes_, index=result.index)
del s, mlb
classes
result = result.join(classes).drop(columns=['molregno', 'tid', 'protein_class_id'])
del classes
result
result = pd.get_dummies(result, columns=['target_type']) # One-hot-encoding
result = result.apply(pd.to_numeric) # Все в численные значения (изначально были object)
result.bei.describe()
Y = result.bei
X = result.drop(columns=['bei'])
del result
from sklearn.model_selection import train_test_split
X_temp, X_test, Y_temp, Y_test =  train_test_split(X,Y,test_size = 0.15, random_state = 6)
X_train, X_val, Y_train, Y_val = train_test_split(X_temp, Y_temp, test_size = 0.2, random_state = 7)
del X, Y, X_temp, Y_temp
from xgboost import XGBRegressor
my_model = XGBRegressor(n_estimators=2000, learning_rate=0.3, tree_method='gpu_hist')
my_model.fit(X_train, Y_train, early_stopping_rounds=5, 
             eval_set=[(X_val, Y_val)], eval_metric = 'rmse', verbose = True)
from xgboost import plot_importance
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(20, 10))
plot_importance(my_model, ax=ax, max_num_features=10)
from sklearn.metrics import mean_absolute_error
Y_pred = my_model.predict(X_test)
mean_absolute_error(Y_test, Y_pred)