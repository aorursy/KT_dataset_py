import pandas as pd

insurance=pd.read_csv("../input/insurance.csv")

insurance.head()

insurance["age_per_bmi"]=insurance["age"]/insurance["bmi"]

insurance.corr()["charges"].sort_values(ascending=False)



from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(insurance, test_size=0.2, random_state=42)



insurance=train_set.copy()

insurance=train_set.drop("charges", axis=1)

insurance_labels = train_set["charges"].copy()



from sklearn.preprocessing import Imputer

imputer=Imputer()

imputer.strategy="median"

insurance_num=insurance.drop("region", axis=1).drop("smoker", axis=1).drop("sex", axis=1)

insurance_num.head()

imputer.fit(insurance_num)

#print(imputer.statistics_)

#print(insurance_num.median().values)

imp_insurance_num=imputer.transform(insurance_num)

insurance_tr=pd.DataFrame(imp_insurance_num, columns=insurance_num.columns)



from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()

insurance_cat=insurance["region"]

insurance_cat1=insurance["smoker"]

insurance_cat2=insurance["sex"]

insurance_cat_encoded=encoder.fit_transform(insurance_cat)

insurance_cat1_encoded=encoder.fit_transform(insurance_cat1)

insurance_cat2_encoded=encoder.fit_transform(insurance_cat2)

insurance_cat_encoded

insurance_cat1_encoded

insurance_cat2_encoded



from sklearn.preprocessing import OneHotEncoder

hot_encoder = OneHotEncoder()

insurance_cat_hot_encoded=hot_encoder.fit_transform(insurance_cat_encoded.reshape(-1,1))

insurance_cat_hot_encoded



from sklearn.preprocessing import LabelBinarizer

encoder=LabelBinarizer(sparse_output=True)

insurance_cat_1hot=encoder.fit_transform(insurance_cat)

insurance_cat_1hot



import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin



age_co, bmi_co=0,2



class CombinedAttributesAdder(BaseEstimator,TransformerMixin):

    

    def __init__(self,age_per_bmi= True):

        self.age_per_bmi=age_per_bmi

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        if self.age_per_bmi == True:

            age_per_bmi = X[:,age_co]/X[:,bmi_co]

            return np.c_[X, age_per_bmi]

        else:

            return np.c_[X]

        

attr_adder = CombinedAttributesAdder(age_per_bmi=False)

insurance_extras= attr_adder.transform(insurance.values)

#print(insurance_extras)



from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline



num_pipeline = Pipeline([

                         ('imputer', Imputer(strategy="median")),

                         ("attribs_adder", CombinedAttributesAdder(age_per_bmi=False)),

                         ("std_scaler", StandardScaler()),

                        ])



insurance_num_tr = num_pipeline.fit_transform(insurance_num)



from sklearn.pipeline import FeatureUnion



num_attribs=list(insurance_num)

cat_attribs=["sex"]



class CustomLabelBinarizer(BaseEstimator, TransformerMixin):

    def __init__(self, sparse_output=False):

        self.sparse_output = sparse_output

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        enc = LabelBinarizer(sparse_output=self.sparse_output)

        return enc.fit_transform(X)



class DataFrameSelector(BaseEstimator, TransformerMixin): 

    def __init__(self, attribute_names):

        self.attribute_names = attribute_names 

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        return X[self.attribute_names].values

    

num_pipeline = Pipeline([

                         ('selector', DataFrameSelector(num_attribs)),

                         ('imputer', Imputer(strategy="median")),

                         ("attribs_adder", CombinedAttributesAdder(age_per_bmi=False)),

                         ("std_scaler", StandardScaler()),

                        ])



cat_pipeline = Pipeline([

    ("selector", DataFrameSelector(cat_attribs)),

    ("label_binarizer", CustomLabelBinarizer()),

])



cat_pipeline1 = Pipeline([

    ("selector", DataFrameSelector(["smoker"])),

    ("label_binarizer", CustomLabelBinarizer()),

])



cat_pipeline2 = Pipeline([

    ("selector", DataFrameSelector(["region"])),

    ("label_binarizer", CustomLabelBinarizer()),

])



full_pipeline = FeatureUnion(transformer_list = [

    ("num_pipeline", num_pipeline),

    ("cat_pipeline", cat_pipeline),

    ("cat_pipeline1", cat_pipeline),

    ("cat_pipeline2", cat_pipeline),

])



insurance_prepared = full_pipeline.fit_transform(insurance)

insurance_prepared.shape



from sklearn.linear_model import LinearRegression 

lin_reg = LinearRegression()

lin_reg.fit(insurance_prepared, insurance_labels)



some_data = insurance.iloc[:5]

some_labels = insurance_labels.iloc[:5]

some_data_prepared = full_pipeline.transform(some_data)

#print("Predictions:\t", lin_reg.predict(some_data_prepared))

#print("Labels:\t\t", list(some_labels))



from sklearn.metrics import mean_squared_error

insurance_predictions = lin_reg.predict(insurance_prepared)

lin_mse = mean_squared_error(insurance_labels, insurance_predictions) 

lin_rmse = np.sqrt(lin_mse)

print(lin_rmse)



from sklearn.tree import DecisionTreeRegressor 

tree_reg = DecisionTreeRegressor()

tree_reg.fit(insurance_prepared, insurance_labels)



some_data = insurance.iloc[:5]

some_labels = insurance_labels.iloc[:5]

some_data_prepared = full_pipeline.transform(some_data)

#print("Predictions:\t", tree_reg.predict(some_data_prepared))

#print("Labels:\t\t", list(some_labels))



insurance_predictions = tree_reg.predict(insurance_prepared)

tree_mse = mean_squared_error(insurance_labels, insurance_predictions)

tree_rmse = np.sqrt(tree_mse)

print(tree_rmse)



from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, insurance_prepared, insurance_labels,

                         scoring="neg_mean_squared_error", cv=10)

rmse_scores = np.sqrt(-scores)

print(rmse_scores.mean())



from sklearn.ensemble import RandomForestRegressor 

forest_reg = RandomForestRegressor()

forest_reg.fit(insurance_prepared, insurance_labels)



forest_scores = cross_val_score(forest_reg, insurance_prepared, insurance_labels, scoring="neg_mean_squared_error", cv=10)

forest_rmse_scores = np.sqrt(-forest_scores)

print(forest_rmse_scores.mean())



from sklearn.model_selection import GridSearchCV

parameters = [{'n_estimators': [1, 2,3,4,5,], 'max_depth': [1,2,3,4,5], 'warm_start': [True] },

              {'bootstrap': [False], 'n_estimators': [1,2,3,4,5], 'max_depth': [1,2,3,4,5], 'warm_start': [True]}]

grid_search = GridSearchCV(estimator = forest_reg,

                           param_grid = parameters,

                           scoring = 'neg_mean_squared_error',

                           cv = 10,

                           n_jobs = -1)

grid_search = grid_search.fit(insurance_prepared, insurance_labels)



print(grid_search.best_params_)



grid_search.best_estimator_



final_model = grid_search.best_estimator_

final_predictions = final_model.predict(insurance_prepared)

final_mse = mean_squared_error(insurance_labels, final_predictions)

final_rmse = np.sqrt(final_mse)

final_rmse