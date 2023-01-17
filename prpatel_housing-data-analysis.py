import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.pipeline import Pipeline

import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score

from sklearn.tree import export_graphviz

# Import data

file_name = "../input/19062019 clean 44 3.txt"

l,fam,a,mm,fm,ba,ls,mgr,fgr,l_r,o_r,tf=np.loadtxt(file_name, unpack="True") #unpack the data
df = pd.DataFrame(zip(mgr, a, mm, fm, tf, ba, ls), columns=['mgr','a','mm','fm','tf','ba', 'ls'])



df.head()
df.describe()
df.info()
df.hist(figsize=(16,8))

plt.show()

corr = df.corr()

sns.heatmap(corr)

plt.show()



sns.pairplot(df)
train_set_original, test_set_original = train_test_split(df, test_size=0.3, random_state=42)
# Custom functions to include in the pipeline



class column_drop(BaseEstimator, TransformerMixin):

    def __init__(self, column_list=None):

        self.column_list = column_list

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        if self.column_list == None:

            return X

        else:

            X_reduced = X.drop(self.column_list, axis=1)

            return X_reduced.values

    

class custom_transform(BaseEstimator, TransformerMixin):

    def __init__(self, column_list=None, func=lambda x:x*1):

        self.column_list=column_list

        self.func=func

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        if column_list is not None:

            X_columns_calculated = self.func(X[:,self.column_list])

            X_reduced            = np.delete(X, self.column_list, axis=1)

            X_transformed        = np.c_[X_reduced, X_columns_calculated]

            

        else:

            X_columns_calculated = self.func(X)

            X_transformed        = X_columns_calculated

        return X_transformed
column_list = list(train_set_original.columns)

print(column_list) # ba is removed in the pipeline and mgr and ls are transformed(subsequently removed)



transformation_pipeline = Pipeline([('drop_column', column_drop(column_list=['ba'])), # Should be the first one 

                                    # as it can accept DataFrame and outputs numpy array

                                    ('scaling',StandardScaler(copy=True,with_mean=False)),

                                    ('custom_transform', custom_transform(column_list=[0,5], func=np.sqrt)),

    

]) 



train_set_transformed    = transformation_pipeline.fit_transform(train_set_original)



df_train_set_transformed = pd.DataFrame(train_set_transformed, columns=['a', 'mm', 'fm', 'tf'] + ['mgr_sqrt', 'ls_sqrt']) 

sns.pairplot(df_train_set_transformed)



df_train_set_transformed.corr()
transformation_pipeline = Pipeline([('drop_column', column_drop(column_list=['tf','ba', 'ls'])), # Should be the first one 

                                    # as it can accept DataFrame and outputs numpy array

                                    ('scaling',StandardScaler(copy=True,with_mean=False)),

                                    ('custom_transform', custom_transform(column_list=[0], func=np.sqrt)),

    

]) 



train_set_transformed    = transformation_pipeline.fit_transform(train_set_original)



df_train_set_transformed = pd.DataFrame(train_set_transformed, columns=['a', 'mm', 'fm'] + ['mgr_sqrt']) 

sns.pairplot(df_train_set_transformed)



df_train_set_transformed.corr()
df_train_set_transformed.plot(kind='scatter', x='a', y='mgr_sqrt', alpha=0.5,

                               c='mm', cmap=plt.get_cmap('jet'), colorbar=True)
df_train_set_transformed_features = df_train_set_transformed.drop('mgr_sqrt',axis=1)

df_train_set_transformed_label    = df_train_set_transformed['mgr_sqrt'].copy()



tree_reg = DecisionTreeRegressor(max_depth=12, min_weight_fraction_leaf=0.002, random_state=4)



scores_tree = cross_val_score(tree_reg, df_train_set_transformed_features, df_train_set_transformed_features, 

                         scoring='neg_mean_squared_error', cv=30)



print(-scores_tree, '\n', -scores_tree.mean(), scores_tree.std(), '\n')



tree_reg.fit(df_train_set_transformed_features, df_train_set_transformed_label)



# dot_data = export_graphviz(

#             tree_reg,

#             out_file=None,

#             feature_names=df_train_set_transformed_features.columns,

#             class_names=['mgr_sqrt'],

#             rounded=True,

#             filled=True

#         )



# graph = pydotplus.graph_from_dot_data(dot_data)



# graph.write_png('visualise_decision_tree.png')





tree_predictions_training_set = tree_reg.predict(df_train_set_transformed_features)

print("Goodness of fit on training Set", round(r2_score(tree_predictions_training_set, df_train_set_transformed_label),2)*100, '%')



# Save the model to re-use later



joblib.dump(tree_reg, 'tree_reg.joblib')

# and later re-use it

import joblib



tree_reg = joblib.load('tree_reg.joblib')

# test_set_transformed    = transformation_pipeline.transform(test_set_original)



# df_test_set_transformed = pd.DataFrame(test_set_transformed, columns=['a', 'mm', 'fm'] + ['mgr_sqrt'])



# df_test_set_transformed_features = df_test_set_transformed.drop('mgr_sqrt',axis=1)

# df_test_set_transformed_label    = df_test_set_transformed['mgr_sqrt'].copy()



# test_predict = tree_reg.predict(df_test_set_transformed_features)

# print(mean_squared_error(test_predict, df_test_set_transformed_label))



# print("Goodness of fit on test Set", round(r2_score(test_predict, df_test_set_transformed_label),2)*100, '%')