# import module we'll need to import our custom module
from shutil import copyfile

# copy our file into the working directory (make sure it has .py suffix)
copyfile(src = "../input/supporting-files/housing_code.py", dst = "../working/housing_code.py")
copyfile(src = "../input/supporting-files/housing_imports.py", dst = "../working/housing_imports.py")
copyfile(src = "../input/supporting-files/housing_models.py", dst = "../working/housing_models.py")
copyfile(src = "../input/github-code/pandas_feature_union.py", dst = "../working/pandas_feature_union.py")
copyfile(src = "../input/github-code/__init__.py", dst = "../working/__init__.py")

# import all our functions
from housing_imports import *
from housing_code import *
from housing_models import *
from pandas_feature_union import *
from __init__ import *
# PRESERVE THE HOME IDS IN FROM THE TEST SET. MERGE THE TWO SETS FOR PROCESSING
csv_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv').drop_duplicates()
csv_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv').drop_duplicates()
y = csv_train.iloc[:, -1]
data = pd.concat([csv_train.iloc[:, 1:-1], csv_test.iloc[:, 1:]], axis=0)
data.reset_index(drop=True, inplace=True)
# print(data.shape)
house_id = pd.DataFrame(csv_test.Id)
# COUPLE DIFFERENT TYPES OF NULL VALUES
null_columns=data.columns[data.isnull().any()]
data[null_columns].isnull().sum().head() # ABBREVIATED
# NULL VALUES FOR GARAGE AND BASEMENT FOLLOW A TREND
import missingno as msno
cats = TypeSelector(np.object).fit_transform(data)
nulls = cats[cats.columns[cats.isnull().any()]]
msno.matrix(nulls);
# CREATE ABSOLUTE TIME FEATURES -> EASY ANSWER. TACKING ON A STRING ENSURES
# THAT THE NUMBER IS NOT ACCIDENTLY CONVERTED BACK TO A NUMERICAL LATER
# ALSO KEEP NUMERICALS FOR BINNING
# GarageYrBlt REMOVED

years = ['YearBuilt', 'YearRemodAdd', 'YrSold']

for colname in years:
        data[colname + '_cat'] = data[colname].apply(lambda x: x if np.isnan(x) else 'year_' + str(int(x)))

# PERMANENTLY TRANSFORM TO CAT - NOT NUMERICAL IN NATURE
data['MSSubClass'] = data.MSSubClass.apply(lambda x: 'class ' + str(x)) 

from tqdm import tqdm
tqdm.pandas()
# NOT MUCH DIFFERENCE BETWEEN GarageYrBlt AND YearBuilt. BASICALLY, 
# THE VALUES DIFFER IF A GARAGE WAS ADDED. A NEW INDICATER FEATURE IS 
# MADE AND THE COLUMN IS DROPPEED

for i in tqdm(range(0, data.shape[0])):
    if np.isnan(data.GarageYrBlt[i]):
        year = data.YearBuilt[i]
        data.GarageYrBlt[i] = year
        
data.GarageYrBlt = data.GarageYrBlt.apply(lambda x: int(x))

new_feat = []
for i in range(0, data.shape[0]):
    if data.GarageYrBlt[i] == data.YearBuilt[i]:
        new_feat.append(0)
    else:
        new_feat.append(1)

# CREATE AN INDICATOR DATAFRAME. THIS HELPS AVOID CONFUSION DURING FINAL
# PREPROCESSING

    
data['Garage_added'] = new_feat
data['Garage_added'] = data['Garage_added'].astype("bool")
data.drop(['GarageYrBlt'], axis=1, inplace=True)
# CREATE AN INDICTATOR FOR REMODEL
new_feat = []
for i in range(0, data.shape[0]):
    if data.YearBuilt[i] == data.YearRemodAdd[i]:
        new_feat.append(0)
    else:
        new_feat.append(1)
        
data['Remodeled'] = new_feat
data['Remodeled'] = data['Remodeled'].astype("bool")
# CONVERT A FEW MORE BOOLS FOR FUN
data['paved_street'] = data.Street.apply(lambda x: 1 if x == 'Pave' else 0).astype('bool')
data['central_air'] = data.CentralAir.apply(lambda x: 1 if x == 'Y' else 0).astype('bool')
data.drop(['Street', 'CentralAir'], axis=1, inplace=True)
# WE'LL LEAVE YearRemodAdd FOR FEATURE SELECTION 
# ADDING RELATIVE REFACTORED TIME FEATURES
import datetime
current = datetime.date.today()
# print(current.year)
data['sold_delta'] = current.year - data['YrSold'] 
data['built_delta'] = current.year - data['YearBuilt'] 
data['remodel_delta'] = current.year - data['YearRemodAdd'] 
# GIVEN THE NUMBER OF NULLS, DISTRIBUTION OF VALUES, AND RELATIONSHIP
# BETWEEN FEATURES, SOME NULLS WILL BE FILLED WITH MODE AND SOME WILL
# BE CONVERTED TO A NEW CATEGORY, "NONE"

none_list = ['Alley', 
       'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
       'BsmtFinType2', 
       'FireplaceQu', 'GarageType', 'GarageFinish',
       'GarageQual', 'GarageCond', 'Fence', 'MiscFeature', 'PoolQC']

for colname in none_list:
    data[colname].fillna('None', inplace=True)
    
# LOOKS READY FOR THE PIPELINES
null_columns=data.columns[data.isnull().any()]
print(data[null_columns].isnull().sum())

# I'VE ABSTRACTED AWAY MOST OF THE PREPROCESSING AND BASIC ENGINEERING. THE
# TRANSFORMERS ARE VIEWABLE IN ATTACHMENTS. SOME OF THE TRANSFORMERS ARE SIMPLY
# WRAPPERS THAT ALLOW THE TRANSFORMERS TO FUNCTION IN AN SKLEARN PIPELINE.

transformer_list=[
        ("binned_features", make_pipeline(
                        TypeSelector(np.number),
                        StandardScalerDf(),
                        SoftImputeDf(),
                        SelectFeatures(),
                        KBins()
        )),
        ("numeric_features", make_pipeline(
                            TypeSelector(np.number),
                            StandardScalerDf(),
                            SoftImputeDf()
        )),
        ("categorical_features", make_pipeline(
                             TypeSelector(np.object),
                             RegImpute() 
        )),
        ("frequency_features", make_pipeline(
                         TypeSelector(np.object),
                         RegImpute(),
                         SelectFeatures(val_count=15, categorical=True),
                         FreqFeatures()
        )),
        ("boolean_features", make_pipeline(
                         TypeSelector(np.bool_),
                         RegImpute(regex=False) 
        ))  ]
preprocess_pipeline = make_pipeline(
    PandasFeatureUnion(transformer_list),
    QuickPipeline_mod()  )
    
X = preprocess_pipeline.fit_transform(data)
# I TRY TO KEEP THINGS IN PANDAS MOST OF THE TIME. THIS COMES IN HANDY WHEN 
# ANALYZING FEATURES LATER ON. 
X.head(2)
# TAKING THE LOG OF THE TARGET CORRECTS FOR A RIGHT SKEW. HOWEVER, THE LOGNORMAL
# CREATES A SLIGHT LEFT SKEW
# NORMALIZATION IS FOR COMPARISON PURPOSES ONLY
# SEPARATED FOR COMPARISON ONLY

y_norm = y.apply(lambda x: (x - y.mean()) / y.std())
y_box, lambda_ = boxcox(y) # need the lambda to eventually reverse the transformation
y_box_norm = pd.DataFrame(y_box).apply(lambda x: (x - y_box.mean()) / y_box.std())
sns.kdeplot(y_norm, label='normal')
sns.kdeplot(np.log(y_norm) + 4, label='lognormal')
sns.kdeplot(np.ravel(y_box_norm) - 4, label='box-cox')
plt.legend();

# THE LEAST SKEWED IS THE BOX COX TRANSFORMATION
y, lambda_ = boxcox(y)

# MAKE SURE THE BOX COX IS REVERSIBLE. THE POWER TRANSFORM AVAILABLE 
# THROUGH SKLEARN DOES NOT SEEM TO PROVIDE A LAMBDA AND IS NOT EASILY REVERSIBLE

def invboxcox(y,ld):
    if ld == 0:
        return(np.exp(y))
    else:
        return(np.exp(np.log(ld*y+1)/ld))

#  LITTLE TEST TO MAKE SURE
# test = csv_train.iloc[:, -1][:100]; print(test[:3])
# y_box_test, lambda_test = boxcox(test); print(pd.Series(y_box_test)[:3])
# # Add 1 to be able to transform 0 values
# test_rev = invboxcox(y_box_test, lambda_test);print(pd.Series(test_rev).apply(lambda x: np.int64(x))[:3])

X_test = X.iloc[1460:, :]
X_ = X.iloc[:1460, :]
print(X_.shape);print(X_test.shape);print(y.shape)
X_train = X_
y_train = y
xgb_params
# REGRESSORS
lgb = BayesLGBMRegressor()
# LGBM WOULD NOT ACCEPT A DICT AS A HYPERPARAMETER - I'D NEED TO EXPLORE THIS MORE, USE **kwargs
svr = BayesSVR(intervals=svr_params)
rf = BayesRandomForest(intervals=rf_params)
regressors = [lgb, svr, rf]

# META-REGRESSOR
meta = BayesXGBRegressor(intervals=xgb_params)
# RELATIVELY STRAIGHTFORWARD. EASY TO DISSECT AND UNDERSTAND THE RELATIONSHIP
# BETWEEN THE META-MODEL AND BASE MODELS.

ensemble = StackingCVRegressorAveraged(regressors=regressors, 
                                       meta_regressor=meta)
ensemble.fit(X_train, y_train)
y_pred = ensemble.predict(X_test)
# REVERSE BOX COX TRANSFORMATION
labels = pd.DataFrame(invboxcox(y_pred, lambda_)).apply(lambda x: np.float64(round(x, 2)))
submit = pd.concat([house_id, labels], axis=1)

submit.set_index('Id', inplace=True)

submit.rename(columns={0: 'SalePrice'}, inplace=True)
submit.head()
# submit.to_csv('preds/predictions5.csv', index=True)
# SOME OF THE CATEGORICALS CAN ALSO BE REPRESENTED AS ORDINALS
# data['BsmtQual'].value_counts(dropna=False)
# cat_to_ordinal_1 = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 
#                     'Ex': 5}

# cat_to_ordinal_2 = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 
#                     'ALQ': 5, 'GLQ': 6}

# Functionality = {'Sal Salvage': 0, 'Sev Severely': 1, 'Maj2': 2, 'Maj1': 3,
#                 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7}

# exposure = {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4}

# cat_to_num_1 = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
#                'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond',
#                'PoolQC']

# cat_to_num_2 = ['BsmtFinType1', 'BsmtFinType2']

# for feat in cat_to_num_1:
#     data[feat].fillna(data[feat].mode()[0], inplace=True)
#     data[feat + '_num'] = data[feat].map(cat_to_ordinal_1)

# for feat in cat_to_num_2:
#     data[feat].fillna(data[feat].mode()[0], inplace=True)
#     data[feat + '_num'] = data[feat].map(cat_to_ordinal_2)

# data['bsmt_exp_num'] =  data['BsmtExposure'].map(exposure)

