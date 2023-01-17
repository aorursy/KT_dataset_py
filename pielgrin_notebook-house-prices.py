#import some necessary librairies



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting

import seaborn as sns

from IPython.display import display

color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)





from scipy import stats

from scipy.stats import norm, skew #for some statistics





pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points





from subprocess import check_output

print(check_output(["ls", "../input/house-prices-advanced-regression-techniques"]).decode("utf8")) #check the files available in the directory
#Now let's import and put the train and test datasets in  pandas dataframe

def get_data():

    train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

    test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

    display(train.head(5))

    display(test.head(5))

    return train, test
def split_id_column(train, test):

    #check the numbers of samples and features

    print("The train data size before dropping Id feature is : {} ".format(train.shape))

    print("The test data size before dropping Id feature is : {} ".format(test.shape))



    #Save the 'Id' column

    train_ID = train['Id']

    test_ID = test['Id']



    #Now drop the  'Id' colum since it's unnecessary for  the prediction process.

    train.drop("Id", axis = 1, inplace = True)

    test.drop("Id", axis = 1, inplace = True)



    #check again the data size after dropping the 'Id' variable

    print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 

    print("The test data size after dropping Id feature is : {} ".format(test.shape))

    

    return train_ID, test_ID, train, test
#descriptive statistics summary

def describe_SalePrice():

    display(train['SalePrice'].describe())
#histogram

def plot_histogram_SalePrice():

    sns.distplot(train['SalePrice']);
def plot_scatter_relation_between_feature_and_target_variable(feature, targetVariable, preferences=None):

    data = pd.concat([train[targetVariable], train[feature]], axis=1)

    data.plot.scatter(x=feature, y=targetVariable, ylim=(0,800000));
def plot_box_relation_between_feature_and_target_variable(feature, targetVariable, preferences=None):

    data = pd.concat([train[targetVariable], train[feature]], axis=1)

    

    if preferences.get('subplots', {}).get('figsize'):

        height = preferences.get('subplots', {}).get('figsize')[0]

        width = preferences.get('subplots', {}).get('figsize')[1]

        f, ax = plt.subplots(figsize=(height, width))

    

    fig = sns.boxplot(x=feature, y=targetVariable, data=data)

    fig.axis(ymin=0, ymax=800000);

    

    if preferences.get('xticks', {}).get('rotation'):

        plt.xticks(rotation=90);
#correlation matrix

def plot_correlation_matrix():

    corrmat = train.corr()

    f, ax = plt.subplots(figsize=(12, 9))

    sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix

def plot_SalePrice_zoom_correlation_matrix(k = 10):

    corrmat = train.corr()

    cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

    cm = np.corrcoef(train[cols].values.T)

    sns.set(font_scale=1.25)

    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

    plt.show()
#scatterplot

def plot_scatter_relations(cols):

    sns.set()

#cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

    sns.pairplot(train[cols], size = 2.5)

    plt.show();
def delete_outliers(train):

    train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

    return train
def plot_feature_distribution(train, feature='SalePrice'):

    sns.distplot(train[feature] , fit=norm);



    # Get the fitted parameters used by the function

    (mu, sigma) = norm.fit(train[feature])

    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



    #Now plot the distribution

    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

                loc='best')

    plt.ylabel('Frequency')

    plt.title(feature + ' distribution')



    #Get also the QQ-plot

    fig = plt.figure()

    res = stats.probplot(train[feature], plot=plt)

    plt.show()
def apply_log_transformation_on_feature(train, feature='SalePrice', visualize = True):



    #We use the numpy fuction log1p which  applies log(1+x) to all elements of the column

    train[feature] = np.log1p(train[feature])



    if visualize == True:

        plot_feature_distribution(train, feature)

    

    return train
def transform_totalbasement(all_data):

    #create column for new variable (one is enough because it's a binary categorical feature)

    #if area>0 it gets 1, for area==0 it gets 0

    all_data['HasBsmt'] = pd.Series(len(all_data['TotalBsmtSF']), index=all_data.index)

    all_data['HasBsmt'] = 0 

    all_data.loc[all_data['TotalBsmtSF']>0,'HasBsmt'] = 1

    

    all_data.loc[all_data['HasBsmt']==1,'TotalBsmtSF'] = np.log(all_data['TotalBsmtSF'])

    

    all_data.drop(['HasBsmt'], axis=1, inplace=True)



    #histogram and normal probability plot

    sns.distplot(all_data[all_data['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);

    fig = plt.figure()

    res = stats.probplot(all_data[all_data['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)

    

    return all_data
def concatenate_data(train, test):

    ntrain = train.shape[0]

    ntest = test.shape[0]

    y_train = train.SalePrice.values

    all_data = pd.concat((train, test)).reset_index(drop=True)

    all_data.drop(['SalePrice'], axis=1, inplace=True)

    print("all_data size is : {}".format(all_data.shape))

    

    return ntrain, ntest, y_train, all_data
def print_missing_data(all_data):

    #missing data

    all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

    all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]

    missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})

    display(missing_data.head(25))
def transform_missing_data(all_data, transform_type='mean'):

    if transform_type == 'none':

        all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

        all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

        all_data["Alley"] = all_data["Alley"].fillna("None")

        all_data["Fence"] = all_data["Fence"].fillna("None")

        all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

    

        #Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

        all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

            lambda x: x.fillna(x.median()))

    

        for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

            all_data[col] = all_data[col].fillna('None')

        

        for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

            all_data[col] = all_data[col].fillna(0)

    

        for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

            all_data[col] = all_data[col].fillna(0)

        

        for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

            all_data[col] = all_data[col].fillna('None')

        

        all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

        all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

    

        all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])



        all_data = all_data.drop(['Utilities'], axis=1)

    

        all_data["Functional"] = all_data["Functional"].fillna("Typ")

    

        all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

    

        all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

    

        all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

        all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

    

        all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

    

        all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

    

    if transform_type=='mean':

        print('transform_type = mean')

        all_data = all_data.fillna(all_data.mean())

    

    return all_data
def transform_numerical_variables_to_categorical(all_data):

    #MSSubClass=The building class

    all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)



    #Changing OverallCond into a categorical variable

    all_data['OverallCond'] = all_data['OverallCond'].astype(str)



    #Year and month sold are transformed into categorical features.

    all_data['YrSold'] = all_data['YrSold'].astype(str)

    all_data['MoSold'] = all_data['MoSold'].astype(str)



    return all_data
def encode_label_to_categorical_features(all_data):



    from sklearn.preprocessing import LabelEncoder

    cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

            'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

            'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

            'YrSold', 'MoSold')

    # process columns, apply LabelEncoder to categorical features

    for c in cols:

        lbl = LabelEncoder() 

        lbl.fit(list(all_data[c].values)) 

        all_data[c] = lbl.transform(list(all_data[c].values))



    # shape        

    print('Shape all_data: {}'.format(all_data.shape))

    

    return all_data
def create_square_feet_feature(all_data):

    # Adding total sqfootage feature 

    all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']

    return all_data
def get_skewness_in_numerical_features(all_data):

    numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



    # Check the skew of all numerical features

    skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

    print("\nSkew in numerical features: \n")

    skewness = pd.DataFrame({'Skew' :skewed_feats})

    display(skewness.head(10))

    

    return skewness
def apply_box_cox_transformation_for_skew_features(all_data, skewness):

    print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



    from scipy.special import boxcox1p

    skewed_features = skewness.index

    lam = 0.15

    for feat in skewed_features:

        #all_data[feat] += 1

        all_data[feat] = boxcox1p(all_data[feat], lam)

    

    return all_data 

#all_data[skewed_features] = np.log1p(all_data[skewed_features])
def end_processing(all_data, ntrain):

    # Convert categorical variable into dummy/indicator variables.

    all_data = pd.get_dummies(all_data)

    print(all_data.shape)

    

    all_data = transform_missing_data(all_data, transform_type='mean')

    #all_data = all_data.fillna(all_data.mean())

    print_missing_data(all_data)

    

    # Split data

    train = all_data[:ntrain]

    test = all_data[ntrain:]

    

    return train, test
from sklearn.metrics import mean_squared_error, accuracy_score

from sklearn import linear_model, svm

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.model_selection import KFold, cross_val_score

import xgboost as xgb

import lightgbm as lgb

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from keras.models import Model

from keras.layers import Input, Dense

from keras.optimizers import Adam
#Validation functions



def rmse_cv(model, train, y_train, n_folds = 5):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)



def mae_cv(model, train, y_train, n_folds = 5):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    mae= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_absolute_error", cv = kf))

    return(mae)



def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):

        self.base_models = base_models

        self.meta_model = meta_model

        self.n_folds = n_folds

   

    # We again fit the data on clones of the original models

    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        

        # Train cloned base models then create out-of-fold predictions

        # that are needed to train the cloned meta-model

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(X, y):

                instance = clone(model)

                self.base_models_[i].append(instance)

                instance.fit(X[train_index], y[train_index])

                y_pred = instance.predict(X[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

                

        # Now train the cloned  meta-model using the out-of-fold predictions as new feature

        self.meta_model_.fit(out_of_fold_predictions, y)

        return self

   

    #Do the predictions of all base models on the test data and use the averaged predictions as 

    #meta-features for the final prediction which is done by the meta-model

    def predict(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)

            for base_models in self.base_models_ ])

        return self.meta_model_.predict(meta_features)
def create_models(model_list):

    models = {}

    if 'linear_regression' in model_list:

        LinRegr = linear_model.LinearRegression()

        models['linear_regression'] = LinRegr

        

    if 'random_forest' in model_list:

        RFRegr = RandomForestRegressor(max_depth=10, random_state=0)

        models['random_forest'] = RFRegr

        

    if 'svmr' in model_list:

        SVMR = svm.SVR()

        models['svmr'] = SVMR

    

    if 'lasso' in model_list:

        lasso = make_pipeline(RobustScaler(), linear_model.LassoCV(alphas =[1, 0.1, 0.001, 0.0005], random_state=1))

        models['lasso'] = lasso



    if 'enet' in model_list:

        enet = make_pipeline(RobustScaler(), linear_model.ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

        models['enet'] = enet



    if 'krr' in model_list:

        krr = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

        models['krr'] = krr

    

    if 'gboost' in model_list:

        gboost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

        models['gboost'] = gboost

    

    if 'xgboost' in model_list:

        xgboost = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1)

        models['xgboost'] = xgboost

        

    if 'lightgbm' in model_list:

        lightgbm = lgb.LGBMRegressor(objective='regression',num_leaves=5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

        models['lightgbm'] = lightgbm

        

    if 'neural_network' in model_list:

        input_layer = Input(shape=(220,))

        first_hidden_layer = Dense(128)(input_layer)

        second_hidden_layer = Dense(32)(first_hidden_layer)

        output_layer = Dense(1, activation='linear')(second_hidden_layer)



        model = Model(inputs=input_layer, outputs=output_layer)

        adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

        model.compile(loss='mean_squared_error', metrics=['mse', 'mae'], optimizer=adam)

        model.fit(x=train, y=y_train, epochs=20, validation_split=0.2)

        

        models['neural_network'] = model

    

    return models
def tune_hyperparameters_for_lasso(alphas):

    models = {}

    

    if not alphas:

        alphas = [0.1]

    

    for a in alphas:

        lasso = linear_model.Lasso(alpha=a)

        models[str(a)] = lasso

    

    return models
def tune_hyperparameters_for_random_forest(max_depths):

    models = {}

    

    if not max_depths:

        max_depths = [2]

    

    for m in max_depths:

        RFRegr = RandomForestRegressor(max_depth=m, random_state=0)

        models[str(m)] = RFRegr

    

    return models
def get_predictions_by_model(train, y_train, model, test):

    print(model)

    model.fit(train, y_train)

    train_pred = model.predict(train)

    pred = np.expm1(model.predict(test.values))

    print(rmsle(y_train, train_pred))

    if model == models['neural_network']:

        pred = pred[:,0]

    return pred
train, test = get_data()
train_ID, test_ID, train, test = split_id_column(train, test)
describe_SalePrice()
plot_histogram_SalePrice()
#scatter plot grlivarea/saleprice



plot_scatter_relation_between_feature_and_target_variable('GrLivArea', 'SalePrice')
#scatter plot totalbsmtsf/saleprice



plot_scatter_relation_between_feature_and_target_variable('TotalBsmtSF', 'SalePrice')
preferences = {'xticks':{'rotation':90}, 'subplots':{'figsize':[8, 6]}}

plot_box_relation_between_feature_and_target_variable('OverallQual', 'SalePrice', preferences)
preferences = {'xticks':{'rotation':90}, 'subplots':{'figsize':[16, 8]}}

plot_box_relation_between_feature_and_target_variable('YearBuilt', 'SalePrice', preferences)
plot_correlation_matrix()
plot_SalePrice_zoom_correlation_matrix(k = 10)
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

plot_scatter_relations(cols)
plot_scatter_relation_between_feature_and_target_variable('GrLivArea', 'SalePrice', preferences)
train = delete_outliers(train)

plot_scatter_relation_between_feature_and_target_variable('GrLivArea', 'SalePrice', preferences)
train = apply_log_transformation_on_feature(train, feature='SalePrice', visualize = True)
ntrain, ntest, y_train, all_data = concatenate_data(train, test)
#all_data = apply_log_transformation_on_feature(all_data, feature='GrLivArea', visualize = True)

all_data = transform_totalbasement(all_data)
skewness = get_skewness_in_numerical_features(all_data)

all_data = apply_box_cox_transformation_for_skew_features(all_data, skewness)

skewness = get_skewness_in_numerical_features(all_data)
print_missing_data(all_data)
#all_data = transform_missing_data(all_data, transform_type='mean')

#all_data = all_data.fillna(all_data.mean())

#print_missing_data(all_data)
all_data = transform_numerical_variables_to_categorical(all_data)
all_data = encode_label_to_categorical_features(all_data)
all_data = create_square_feet_feature(all_data)
train, test = end_processing(all_data, ntrain)
#models = create_models(['linear_regression', 'random_forest', 'svmr', 'lasso'])

models = create_models(['lasso', 'neural_network'])



if ('enet' in models) and ('gboost' in models) and ('krr' in models) and ('lasso' in models):

    stacked_averaged_models = StackingAveragedModels(base_models = (models['enet'], models['gboost'], models['krr']),

                                                 meta_model = models['lasso'])

    #models['stacked'] = stacked_averaged_models



#models = tune_hyperparameters_for_lasso([0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009])



#models = tune_hyperparameters_for_random_forest([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])



#for key, value in models.items():

#    rmse = rmse_cv(value, train, y_train, 5)

#    mae = mae_cv(value, train, y_train, 5)

#    

#    print("\n" + key +" rmse score: {:.4f} ({:.4f})\n".format(rmse.mean(), rmse.std()))

#    print("\n" + key +" mae score: {:.4f} ({:.4f})\n".format(mae.mean(), mae.std()))


#my_model = models[model_to_use]

#my_model.fit(train, y_train)

#predicted_prices = my_model.predict(test)

#print(predicted_prices)



#lasso_preds = get_predictions_by_model(train, y_train, models['lasso'], test)

#xgb_preds = get_predictions_by_model(train, y_train, models['xgboost'])

neural_network_preds = get_predictions_by_model(train, y_train, models['neural_network'], test)



#predicted_prices = 0.7*lasso_preds + 0.3*xgb_preds



my_submission = pd.DataFrame({'Id': test_ID, 'SalePrice': neural_network_preds})

# you could use any filename. We choose submission here

file_name = 'submission_' + 'neural_network' + '.csv'

my_submission.to_csv(file_name, index=False)
#ensemble = get_predictions_by_model(train, y_train, stacked_averaged_models)*0.70 

#+ get_predictions_by_model(train, y_train, models['xgboost'])*0.15 

#+ get_predictions_by_model(train, y_train, models['lightgbm'])*0.15

#        

#my_submission = pd.DataFrame({'Id': test_ID, 'SalePrice': ensemble})

#my_submission.to_csv('stack_submission.csv',index=False)