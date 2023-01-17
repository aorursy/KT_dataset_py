
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pd.plotting.register_matplotlib_converters()
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
df_crude = pd.read_csv("../input/argentina-venta-de-propiedades/ar_properties_crude.csv", index_col="id")
df_crude.head(5)
#CABA = Ciudad Autónoma de Buenos Aires = Buenos Aires City
df_CABA_dolar = df_crude[(df_crude["l1"]=="Argentina") & (df_crude["l2"]=="Capital Federal") & (df_crude["currency"]=="USD")]
df_CABA_dolar
df_CABA_dolar.columns
df_CABA_dolar["ad_type"].unique()
df_CABA_dolar = df_CABA_dolar.drop(columns=["ad_type"])
df_CABA_dolar = df_CABA_dolar.drop(columns=["start_date", "end_date", "created_on", "price_period"])
df_CABA_dolar = df_CABA_dolar.drop(columns=["l1", "l2" ,"currency"])
missing_percentage = df_CABA_dolar.isnull().sum()*100/len(df_CABA_dolar.index)
missing_percentage
df_CABA_dolar = df_CABA_dolar.drop(columns=["l4","l5","l6"])
df_CABA_dolar["operation_type"].value_counts()
df_CABA_dolar = df_CABA_dolar[df_CABA_dolar["operation_type"]=="Venta"]
df_CABA_dolar = df_CABA_dolar.drop(columns=["title", "description","operation_type"])
df_CABA_dolar
import folium
from folium import Marker
from folium.plugins import HeatMap
map_2 = folium.Map(width = 700, height = 500, location=[-34.586662, -58.436620], titles="cartodbposition", zoom_start=12)
df_CABA_dolar_noLatNorLonMissing = df_CABA_dolar[df_CABA_dolar["lat"].notnull() & df_CABA_dolar["lon"].notnull()]
HeatMap(data=df_CABA_dolar_noLatNorLonMissing[["lat","lon"]], radius=12).add_to(map_2)
map_2
df_CABA_dolar["l3"].value_counts().head(8)
df_CABA_dolar.groupby(by=["l3"], axis=0)["price"].median().sort_values(ascending=False).head(7)
df_CABA_dolar.groupby(by=["l3"], axis=0)["price"].median().sort_values(ascending=False).tail(7)

plt.figure(figsize=(10,7))
plt.ticklabel_format(style='plain', axis='x')
sns.distplot(df_CABA_dolar["price"])

plt.ylim(0,10**-7)
plt.xlim(0,4000000)
df_CABA_dolar.describe()

plt.figure(figsize=(15,10))
plt.hist(x=df_CABA_dolar["property_type"])
properties = df_CABA_dolar
current_missing_percentages = (properties.isnull().sum()/properties.shape[0]).sort_values(ascending=False)
current_missing_percentages
print(properties[properties["surface_covered"].isnull() & properties["surface_total"].isnull()].shape[0])
properties[properties["surface_covered"].isnull() & properties["surface_total"].isnull()].shape[0]/properties.shape[0]
def columns_correlation_with_target(df,target):
    for feature in df.select_dtypes(exclude=["object"]).columns:
        if feature!=target:
            print("Correlation between ", feature, " and ", target, ": ", df[target].corr(df[feature]))
columns_correlation_with_target(properties,"price")
properties = properties.drop(columns=["bedrooms"])
properties
core_properties = properties.drop(columns=["lat","lon"])
core_properties = core_properties.dropna(axis=0)
core_properties.shape
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
X_core = core_properties.drop(columns=["price"])
y_core = core_properties["price"]

X_core_train, X_core_valid, y_core_train, y_core_valid = train_test_split(X_core,y_core)

OHE4 = OneHotEncoder(handle_unknown="ignore",sparse=False)

obj_cols = [col for col in X_core_valid.columns if X_core_valid[col].dtype=="object"]

OHE4_cat_train = pd.DataFrame(OHE4.fit_transform(X_core_train[obj_cols]))
OHE4_cat_valid = pd.DataFrame(OHE4.transform(X_core_valid[obj_cols]))

OHE4_cat_train.index = X_core_train.index
OHE4_cat_valid.index = X_core_valid.index

num_cols = [col for col in X_core_valid.columns if X_core_valid[col].dtype=="float64"]
num_cols
encoded_X_core_train = pd.concat([OHE4_cat_train,X_core_train[num_cols]],axis=1)
encoded_X_core_valid = pd.concat([OHE4_cat_valid, X_core_valid[num_cols]], axis=1)


RFR2 = RandomForestRegressor(random_state=4).fit(encoded_X_core_train,y_core_train)
RFR2_preds = RFR2.predict(encoded_X_core_valid)
RFR2_MAE = mean_absolute_error(RFR2_preds,y_core_valid)
RFR2_MAE
bath_rooms_ratio = core_properties["bathrooms"]/core_properties["rooms"]
surf_covered_by_total= core_properties["surface_covered"]/core_properties["surface_total"]
l3_type = core_properties["l3"]+core_properties["property_type"]
def newFeatureTester(df, new_column):
    X_core["new_feature"] = new_column
    
    X_train, X_test, y_train, y_test = train_test_split(X_core,y_core)
    
    objec_cols = [col for col in X_test.columns if X_test[col].dtype=="object"]
    
    OHE = OneHotEncoder(handle_unknown="ignore", sparse=False)
    OHEncoded_cats_train = pd.DataFrame(OHE.fit_transform(X_train[objec_cols]))
    OHEncoded_cats_test = pd.DataFrame(OHE.transform(X_test[objec_cols]))
    
    OHEncoded_cats_train.index = X_train.index
    OHEncoded_cats_test.index = X_test.index
    
    numericals_train = X_train.select_dtypes(exclude=["object"])
    numericals_test = X_test.select_dtypes(exclude=["object"])
    
    OHEncoded_train = pd.concat([OHEncoded_cats_train,numericals_train], axis=1)
    OHEncoded_test = pd.concat([OHEncoded_cats_test, numericals_test], axis=1) 
    
    model = RandomForestRegressor(random_state=3).fit(OHEncoded_train,y_train)
    preds = model.predict(OHEncoded_test)
    
    mae = mean_absolute_error(preds,y_test)
    
    print("MAE with ", new_column.name,": " , mae)
    
    mae_avg_price = mae/(y_core.mean())
    
    print("MAE/AVG PRICE: ", new_column.name,": ",  mae_avg_price)
    
    return [mae,mae_avg_price]
    
res = pd.DataFrame({"bath_rooms_ratio" : newFeatureTester(core_properties,bath_rooms_ratio),
"surf_covered_by_total" : newFeatureTester(core_properties,surf_covered_by_total),
"l3_type":newFeatureTester(core_properties,l3_type)},index=["MAE","MAE/AVG Price"])
res
dict_new_features = {"bath_rooms_ratio":bath_rooms_ratio, "surf_covered_by_total":surf_covered_by_total, "l3_type":l3_type}
df_new_features = pd.DataFrame(dict_new_features)
df_new_features

X_core_plus_new = pd.concat([X_core,df_new_features], axis=1)
X_core_plus_new.isnull().sum()
import eli5
from eli5.sklearn import PermutationImportance
X_train, X_test, y_train, y_test = train_test_split(X_core_plus_new,y_core)
object_cols = [col for col in X_train.columns if X_train[col].dtype=="object"]

OHE2 = OneHotEncoder(handle_unknown="ignore", sparse=False)

labeled_obj_cols_train = pd.DataFrame(OHE2.fit_transform(X_train[object_cols]))
labeled_obj_cols_test = pd.DataFrame(OHE2.transform(X_test[object_cols]))

#OneHotEncoder removed indexes, put them back...
labeled_obj_cols_train.index = X_train.index
labeled_obj_cols_test.index= X_test.index

numeric_X_train = X_train.select_dtypes(exclude=["object"])
numeric_X_test = X_test.select_dtypes(exclude=["object"])
labeled_X_train  =  pd.concat([labeled_obj_cols_train,numeric_X_train], axis=1)
labeled_X_test = pd.concat([labeled_obj_cols_test,numeric_X_test], axis=1)
RFReg = RandomForestRegressor(random_state=7).fit(labeled_X_train,y_train)

permutator = PermutationImportance(RFReg,random_state=2).fit(labeled_X_test,y_test)
permutator.estimator
colnames_labeled = labeled_X_train.columns.tolist()
colnames_labeled_all_as_strings = [str(name) for name in colnames_labeled]
eli5.show_weights(permutator, feature_names=colnames_labeled_all_as_strings, top=len(colnames_labeled_all_as_strings))
preds = permutator.predict(labeled_X_test)
mae = mean_absolute_error(preds,y_test)
print("Mean absolute error: ",mae, " . Error in relation to mean price: ", mae/y_test.mean(), "% .")
X_core_plus_new = X_core_plus_new.drop(columns=["surf_covered_by_total"])
from sklearn.impute import SimpleImputer
properties_with_missing = properties.drop(columns=["lat","lon"])
properties_with_missing_numericals = properties_with_missing.dropna(subset=["l3","property_type"])
properties_with_missing_numericals
X = properties_with_missing_numericals.drop(columns=["price"])
y=properties_with_missing_numericals["price"]
X_train, X_test, y_train, t_test = train_test_split(X,y)
obj_cols = [col for col in X_test.columns if X_test[col].dtype=="object"]
OHE3 = OneHotEncoder(handle_unknown="ignore",sparse=False)

OHE3_cat_X_train = pd.DataFrame(OHE3.fit_transform(X_train[obj_cols]))
OHE3_cat_X_test = pd.DataFrame(OHE3.transform(X_test[obj_cols]))
#One Hot Encoder lost indexes, put them back...
OHE3_cat_X_train.index = X_train.index
OHE3_cat_X_test.index = X_test.index

numerical_X_train = X_train.select_dtypes(exclude=["object"])
numerical_X_test = X_test.select_dtypes(exclude=["object"])

OHE3_X_train = pd.concat([OHE3_cat_X_train,numerical_X_train], axis=1)
OHE3_X_test = pd.concat([OHE3_cat_X_test,numerical_X_test],axis=1)
imputer = SimpleImputer(strategy="median")

imputed_X_train = pd.DataFrame(imputer.fit_transform(OHE3_X_train))
imputed_X_test = pd.DataFrame(imputer.transform(OHE3_X_test))
#imputer removed column names, put them back    
imputed_X_train.columns = OHE3_X_train.columns
imputed_X_test.columns = OHE3_X_test.columns
RFR_new = RandomForestRegressor(random_state = 1).fit(imputed_X_train,y_train)
predictions_RFR_new = RFR_new.predict(imputed_X_test)
mae_RFR_new = mean_absolute_error(predictions_RFR_new,t_test)
mae_RFR_new
from joblib import dump
dump(RFR_new,"imputed_RFR.joblib")
"""X_core_train, X_core_valid, y_core_train, y_core_valid = train_test_split(labeled_X_train,y_train)
OHE4 = OneHotEncoder(handle_unknown="ignore",sparse=False)

OHE4_cat_train = pd.DataFrame(OHE4.fit_transform(X_core_train[obj_cols]))
OHE4_cat_valid = pd.DataFrame(OHE4.transform(X_core_valid[obj_cols]))

OHE4_cat_train.index = X_core_train.index
OHE4_cat_valid.index = X_core_valid.index

num_cols = [col for col in X_core_valid.columns if X_core_valid[col].dtype=="float64"]
num_cols
encoded_X_core_train = pd.concat([OHE4_cat_train,X_core_train[num_cols]],axis=1)
encoded_X_core_valid = pd.concat([OHE4_cat_valid, X_core_valid[num_cols]], axis=1)"""
from xgboost import XGBRegressor
XGBR2 = XGBRegressor(random_state=6,n_estimators=900,early_stopping_rounds=10, 
                     eval_set=[encoded_X_core_valid,y_core_valid],verbose=False).fit(encoded_X_core_train,y_core_train)
XGBR2_preds = XGBR2.predict(encoded_X_core_valid)
XGBR2_MAE = mean_absolute_error(y_core_valid,XGBR2_preds)
XGBR2_MAE
dump(XGBR2,"XGBR2.joblib")
encoded_X_core_train
labeled_X_train
RFR2 = RandomForestRegressor(random_state=4).fit(encoded_X_core_train,y_core_train)
RFR2_preds = RFR2.predict(encoded_X_core_valid)
RFR2_MAE = mean_absolute_error(RFR2_preds,y_core_valid)
RFR2_MAE
RFR3 = RandomForestRegressor(random_state=4).fit(labeled_X_train,y_core_train)
RFR3_preds = RFR3.predict(labeled_X_test)
RFR3_MAE = mean_absolute_error(RFR3_preds,y_core_valid)
RFR3_MAE
from joblib import dump, load
dump(RFR2, 'baseline_random_forest.joblib')#37.6k
for col in X_train.columns:
    if X_train[col].dtype=="object":
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')
LGBM = LGBMRegressor(random_state=12).fit(X_train,y_train)
preds_LGBM = LGBM.predict(X_test)
mean_absolute_error(preds_LGBM,y_test)
LGBM2 = LGBMRegressor(random_state=12).fit(labeled_X_train,y_train)
preds_LGBM2 = LGBM2.predict(labeled_X_test)
mean_absolute_error(preds_LGBM2,y_test)
from catboost import CatBoostRegressor

CBR = CatBoostRegressor(random_state=9,cat_features=["l3",'l3_type', 'property_type']).fit(X_train,y_train)
preds = CBR.predict(X_test)
MAE = mean_absolute_error(preds,y_test)
MAE
CBR2 = CatBoostRegressor(random_state=9).fit(labeled_X_train,y_train)
preds = CBR2.predict(labeled_X_test)
MAE = mean_absolute_error(preds,y_test)
MAE
dump(RFR2, 'baseline_random_forest.joblib')#37.6k
dump(RFR3, 'l3_types_random_forest.joblib')#37.6k
dump(XGBR2,"XGBR2.joblib")#40k
dump(RFR_new,"imputed_RFR.joblib")#70k
from sklearn.model_selection import GridSearchCV
parameters_forsearch = {
    "n_estimators" : [100,250,500,750]
}
search = GridSearchCV(RandomForestRegressor(),parameters_forsearch,cv=2)
search.fit(encoded_X_core_train,y_core_train)
print(search.best_params_)
current_model = load("baseline_random_forest.joblib")

