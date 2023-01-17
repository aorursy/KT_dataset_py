import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import Imputer

from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor



data = pd.read_csv('../input/data.csv',index_col=0)



data['shot_id_number'] = data.index + 1  # shot id number has null value



sample = pd.read_csv('../input/sample_submission.csv',index_col=0)



data_train_test_combined =  data[data['is_goal'].notnull()]

test_file = data[data['is_goal'].isnull()] #.dropna(axis=0,subset=['shot_id_number'])
data_train_test_combined.dropna(axis=0,subset=['is_goal'],inplace=True) #Not required

y = data_train_test_combined.is_goal

X = data_train_test_combined.drop(['is_goal'],axis =1 ) #.select_dtypes(exclude =['object'])
train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.25)

X_test = test_file.drop(['is_goal'],axis =1)
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()

imputed_X_train = my_imputer.fit_transform(train_X.select_dtypes(exclude='object'))

imputed_X_valid = my_imputer.transform(test_X.select_dtypes(exclude='object'))

imputed_final_test = my_imputer.transform(X_test.select_dtypes(exclude='object'))
# pd.DataFrame(imputed_X_train,columns=train_X.select_dtypes(exclude ='object').columns) #imputer return numpy ndarray

# pd.DataFrame(imputed_X_train)



# pd.DataFrame(imputed_X_train,columns=train_X.select_dtypes(exclude ='object').columns)
# All categorical columns

object_cols = [col for col in train_X.columns if train_X[col].dtype == "object"]



# Columns that can be safely label encoded

good_label_cols = [col for col in object_cols if 

                   set(train_X[col]) == set(X_test[col])]

        

# Problematic columns that will be dropped from the dataset

bad_label_cols =  list (set(object_cols) - set(good_label_cols))

        

print('Categorical columns that will be label encoded:', good_label_cols)

print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)
# Label Encoding 

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

# Drop categorical columns that will not be encoded

label_X_train = train_X.drop(bad_label_cols, axis=1)

label_X_valid = test_X.drop(bad_label_cols, axis=1)

label_final_X_test = X_test.drop(bad_label_cols,axis=1)



# Apply label encoder 

label_encoder = LabelEncoder()





for col in set(good_label_cols):

    label_X_train[col] = label_encoder.fit_transform(label_X_train[col].astype(str))

    label_X_valid[col] = label_encoder.transform(label_X_valid[col].astype(str))

    label_final_X_test[col] = label_encoder.transform(label_final_X_test[col].astype(str))
Numerical = pd.DataFrame(imputed_X_train,columns=train_X.select_dtypes(exclude ='object').columns)

Numerical_V = pd.DataFrame(imputed_X_valid,columns=test_X.select_dtypes(exclude='object').columns)

Numerical_T = pd.DataFrame(imputed_final_test,columns=X_test.select_dtypes(exclude='object').columns)

Numerical_T.columns
#pd.concat(label_X_train,pd.DataFrame(imputed_X_train,columns=train_X.select_dtypes(exclude ='object').columns))

Categorical = label_X_train[good_label_cols]

Categorical_V = label_X_valid[good_label_cols]

Categorical_T = label_final_X_test[good_label_cols]

Categorical_T.columns
#model_train_x = pd.concat([label_X_train[good_label_cols],imputed_X_train],ignore_index=True,sort=False)

# pd.concat([data1, data2], ignore_index=True, sort =False)

Model_train_X = pd.concat([Numerical.reset_index() ,Categorical.reset_index() ],ignore_index=True,sort=False,axis=1) # Number of rows were increasing

Model_valid_X = pd.concat([Numerical_V.reset_index(),Categorical_V.reset_index()],ignore_index=True,sort=False,axis=1) 

Model_test_X = pd.concat([Numerical_T.reset_index(),Categorical_T.reset_index()],ignore_index=True,sort=False,axis=1)
#Model_train_X.shape

#feature engineering

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestClassifier

sel = SelectFromModel(RandomForestClassifier(n_estimators = 100))

sel.fit(Model_train_X, train_y)

sel.get_support()

selected_feat= Model_train_X.columns[(sel.get_support())]

len(selected_feat)

print(selected_feat)





def plot_feature_importances(clf, X_train, y_train=None, 

                             top_n=10, figsize=(8,8), print_table=False, title="Feature Importances"):

    '''

    plot feature importances of a tree-based sklearn estimator

    

    Note: X_train and y_train are pandas DataFrames

    

    Note: Scikit-plot is a lovely package but I sometimes have issues

              1. flexibility/extendibility

              2. complicated models/datasets

          But for many situations Scikit-plot is the way to go

          see https://scikit-plot.readthedocs.io/en/latest/Quickstart.html

    

    Parameters

    ----------

        clf         (sklearn estimator) if not fitted, this routine will fit it

        

        X_train     (pandas DataFrame)

        

        y_train     (pandas DataFrame)  optional

                                        required only if clf has not already been fitted 

        

        top_n       (int)               Plot the top_n most-important features

                                        Default: 10

                                        

        figsize     ((int,int))         The physical size of the plot

                                        Default: (8,8)

        

        print_table (boolean)           If True, print out the table of feature importances

                                        Default: False

        

    Returns

    -------

        the pandas dataframe with the features and their importance

        

    Author

    ------

        George Fisher

    '''

    

    __name__ = "plot_feature_importances"

    

    import pandas as pd

    import numpy  as np

    import matplotlib.pyplot as plt

    

    from xgboost.core     import XGBoostError

    from lightgbm.sklearn import LightGBMError

    

    try: 

        if not hasattr(clf, 'feature_importances_'):

            clf.fit(X_train.values, y_train.values.ravel())



            if not hasattr(clf, 'feature_importances_'):

                raise AttributeError("{} does not have feature_importances_ attribute".

                                    format(clf.__class__.__name__))

                

    except (XGBoostError, LightGBMError, ValueError):

        clf.fit(X_train.values, y_train.values.ravel())

            

    feat_imp = pd.DataFrame({'importance':clf.feature_importances_})    

    feat_imp['feature'] = X_train.columns

    feat_imp.sort_values(by='importance', ascending=False, inplace=True)

    feat_imp = feat_imp.iloc[:top_n]

    

    feat_imp.sort_values(by='importance', inplace=True)

    feat_imp = feat_imp.set_index('feature', drop=True)

    feat_imp.plot.barh(title=title, figsize=figsize)

    plt.xlabel('Feature Importance Score')

    plt.show()

    

    if print_table:

        from IPython.display import display

        print("Top {} features in descending order of importance".format(top_n))

        display(feat_imp.sort_values(by='importance', ascending=False))

        

    return feat_imp



import pandas as pd

# X_train = pd.DataFrame(X)

# y_train = pd.DataFrame(y)



from xgboost              import XGBClassifier

from sklearn.ensemble     import ExtraTreesClassifier

from sklearn.tree         import ExtraTreeClassifier

from sklearn.tree         import DecisionTreeClassifier

from sklearn.ensemble     import GradientBoostingClassifier

from sklearn.ensemble     import BaggingClassifier

from sklearn.ensemble     import AdaBoostClassifier

from sklearn.ensemble     import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from lightgbm             import LGBMClassifier





clfs = [XGBClassifier(),              LGBMClassifier(), 

        ExtraTreesClassifier(),       ExtraTreeClassifier(),

        BaggingClassifier(),          DecisionTreeClassifier(),

        GradientBoostingClassifier(), LogisticRegression(),

        AdaBoostClassifier(),         RandomForestClassifier()]



for clf in clfs:

    try:

        _ = plot_feature_importances(clf, Model_train_X, train_y, top_n=Model_train_X.shape[1], title=clf.__class__.__name__)

    except AttributeError as e:

        print(e)



my_model = XGBRegressor(n_estimators=1000)

my_model.fit(Model_train_X,train_y)

from sklearn.metrics import mean_absolute_error

mean_absolute_error(my_model.predict(Model_valid_X),test_y)
my_model = XGBRegressor(n_estimators=1000)

my_model.fit(label_X_train[good_label_cols],train_y)

from sklearn.metrics import mean_absolute_error

mean_absolute_error(my_model.predict(label_X_valid[good_label_cols]),test_y)

#ans = my_model.predict(label_final_X_test[good_label_cols])

# ans[:5]
test_preds = my_model.predict(label_final_X_test[good_label_cols])

output = pd.DataFrame({'shot_id_number': X_test['shot_id_number'],

                       'is_goal': test_preds})

output.loc[output['is_goal'] < 0,'is_goal'] = 0

output.to_csv('submission.csv',index=False)
ans = pd.read_csv('submission.csv')




ans[ans['is_goal'] < 0] 

ans.describe()