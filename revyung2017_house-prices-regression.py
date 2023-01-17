import pandas as pd

import numpy as np

pd.set_option('max_rows', 10000)

pd.set_option('max_columns', 20000)

from IPython.display import display

# reading data

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

display(train.head())

display(train.describe())

print(train.shape)

train.info()
# create copy before messing

train_cleaned = train.copy()

test_cleaned = test.copy()

# after inspection, we can notice high std condition like square feet, mostly areas, while some need manual tuning like years

# additionally, log transform Sale Price might help since its variance is huge

# but rmb to transform back when prediction is done

rescale_SF = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 

               'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 

              'GrLivArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 

              'ScreenPorch', 'PoolArea', 'MiscVal']

def SF_rescale(data):

    for col in rescale_SF:

        data[col] = data[col].apply(lambda x: np.log(1+x))

    return data

def train_SalePrice_rescale(data):

    data['SalePrice'] = data['SalePrice'].apply(lambda x: np.log(1+x))

    return data

# minus 1900 to data.GarageYrBlt since min is 1900

# minus 1872 to data.YearBuilt 

# minus 1950 to data.YearRemodAdd

# minus 2006 to data.YrSold

def year_rescale(data):

    data['GarageYrBlt'] = data.GarageYrBlt - 1900

    data['YearBuilt'] = data.YearBuilt - 1872

    data['YearRemodAdd'] = data.YearRemodAdd - 1950

    data['YrSold'] = data.YrSold - 2006

    return data

# auto data cleaning since data contains too many columns

def data_cleaning(data):

    for col_n, row_ind in data.iteritems():

        # convert categorical to numeric

        if data[col_n].dtypes == 'object':

            data[col_n] = data[col_n].factorize()[0]

        # check nos of NAN to decide what to fill

        if data[col_n].count() / data.shape[0] > 0.1:

            data[col_n].fillna('-1', inplace=True)

        else:

            median = data[col_n].median()

            data[col_n].fillna(median, inplace=True)

        # one hot encode non-ordinal and non-binary categorical columns

        # however, its better to use groupby and plot to manually extract signal before creating

        # too much dimensionality 

    print(data.shape)

    return data



train_cleaned = SF_rescale(train_cleaned)

train_cleaned = train_SalePrice_rescale(train_cleaned)

test_cleaned = SF_rescale(test_cleaned)



train_cleaned = year_rescale(train_cleaned)

test_cleaned = year_rescale(test_cleaned)



train_cleaned = data_cleaning(train_cleaned)

test_cleaned = data_cleaning(test_cleaned)



train_cleaned = train_cleaned.astype(dtype=np.float64)

test_cleaned = test_cleaned.astype(dtype=np.float64)
train_cleaned.info()
# lets apply minMax scaler to train & test before PCA

for col in train_cleaned.columns.tolist()[1:-1]:

    maxi = max(max(train_cleaned[col]), max(test_cleaned[col]))

    mini = min(min(train_cleaned[col]), min(test_cleaned[col]))

    RANGE = maxi- mini

    train_cleaned[col] = (train_cleaned[col] - mini) / RANGE

    test_cleaned[col] = (test_cleaned[col] - mini) / RANGE
test_cleaned.describe()
# apply PCA to reduce dimensionality, since nos of rows is much less than that

from sklearn.decomposition import PCA

bins = 5

nos_col = train_cleaned.shape[1]

n_trials = np.arange(nos_col//5, nos_col-2, nos_col//5)

n_explained_variance = []

# rmb pca is applied to both train and test set so make sure u fit both

all_data = np.concatenate((train_cleaned.drop(['SalePrice', 'Id'], axis=1), test_cleaned.drop(['Id'], axis=1)))

for n_components in n_trials:

    pca = PCA(n_components=n_components)

    pca.fit(all_data)

    n_explained_variance.append(sum(pca.explained_variance_ratio_)*100)

# plot pca against n_components

import matplotlib.pyplot as plt

plt.plot(n_trials, n_explained_variance)

print(n_trials)

print(n_explained_variance)
# apply PCA

pca = PCA(n_components=40)

all_data = np.concatenate((train_cleaned.drop(['SalePrice', 'Id'], axis=1), test_cleaned.drop(['Id'], axis=1)))

pca.fit(all_data)

train_PCA = pca.transform(train_cleaned.drop(['SalePrice', 'Id'], axis=1))

test_PCA = pca.transform(test_cleaned.drop(['Id'], axis=1))

train_PCA = pd.DataFrame(train_PCA)

test_PCA = pd.DataFrame(test_PCA)

# add back SalePrice after PCA

train_PCA['SalePrice'] = train_cleaned.SalePrice

# lets check for outliners and scale of our data set after PCA

print(train_PCA.shape)

train_PCA.describe()
# create validation from train set for performance testing

from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(train_cleaned.iloc[:, 1:-1], train_cleaned.iloc[:, -1], test_size=0.3, random_state=42, shuffle=True) 

# print(X_train.shape, X_val.shape)

# lets get a baseline accuracy and get feature importance for GBT 

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error as mse

GBR = GradientBoostingRegressor()

GBR.fit(X_train, Y_train)

train_mse_error = mse(Y_train, GBR.predict(X_train))

val_mse_error = mse(Y_val, GBR.predict(X_val))

print('Inverse-transform back it to the unit of Price since we used log(1+x).\n')

print('original_training RMSE: {}'.format(train_mse_error**0.5))

print('original_validation RMSE: {}'.format(val_mse_error**0.5))
# plot feature importance for feature engineering

import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(20,10))

plt.title('feature importance of GBR')

plt.bar(train_cleaned.columns.tolist()[1:-1], GBR.feature_importances_)

plt.show()
X_train, X_val, Y_train, Y_val = train_test_split(train_PCA.iloc[:, :-1], train_PCA.iloc[:, -1], test_size=0.3, random_state=42, shuffle=True) 

# print(X_train.shape, X_val.shape)

# lets get a baseline accuracy and get feature importance for GBT 

GBR_PCA = GradientBoostingRegressor()

GBR_PCA.fit(X_train, Y_train)

train_mse_error = mse(Y_train, GBR_PCA.predict(X_train))

val_mse_error = mse(Y_val, GBR_PCA.predict(X_val))

print('pca_training RMSE: {}'.format(train_mse_error**0.5))

print('pca_validation RMSE: {}'.format(val_mse_error**0.5))
# plot feature importance for feature engineering

import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(20,10))

plt.title('feature importance of GBR_PCA')

plt.bar(train_PCA.columns.tolist()[:-1], GBR_PCA.feature_importances_)

plt.show()
picked_PCA_features = [0, 3, 24]

train_PCA[picked_PCA_features].hist()
# also prepare a set of input that its SalePrice is not scaled

train_NS = train_cleaned.copy()

train_NS.SalePrice = train_NS.SalePrice.apply(lambda x: np.exp(x)-1)
# 1st level: GBT_R, RF_R, SVR, LR. KNN_R 

# 2nd level: Linear Model

import sklearn.ensemble as ensemble

import sklearn.neighbors as neighbors

import sklearn.linear_model as linear

import sklearn.svm as svm

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn import tree

from sklearn import neural_network



models = [[ensemble.GradientBoostingRegressor(),

           ensemble.RandomForestRegressor(),

           svm.SVR(gamma='auto'),

           linear.BayesianRidge(),

           GaussianProcessRegressor(),

           tree.DecisionTreeRegressor(),

           neural_network.MLPRegressor(),

           neighbors.KNeighborsRegressor()], 

         linear.BayesianRidge()]



# creates dataframe to store first level result

first_level_B = pd.DataFrame()

first_level_C = pd.DataFrame()

i = 0

for AB_C in [(train_PCA[picked_PCA_features], test_PCA[picked_PCA_features]),

            (train_PCA.iloc[:,:-1], test_PCA.iloc[:,:]),

            (train_cleaned.iloc[:,1:-1], test_cleaned.iloc[:,1:]),

            (train_NS.iloc[:,1:-1], test_cleaned.iloc[:,1:])]:

    AB, C = AB_C[0], AB_C[1]

    A_x, B_x, A_y, B_y = train_test_split(AB, train_PCA.iloc[:,-1], test_size=0.1, random_state=42)

    for model in models[0]:

        model.fit(A_x, A_y)

        first_level_B[type(model).__name__+'_'+str(i)] = model.predict(B_x)

        first_level_C[type(model).__name__+'_'+str(i)] = model.predict(C)

    i += 1

    

first_level_B['B_y'] = B_y.to_numpy()

display(first_level_B.head())

first_level_C.head()
# perform second level stacking

final_model = models[1]

final_model.fit(first_level_B.iloc[:,:-1], first_level_B.iloc[:,-1])

# mse

training_prediction = final_model.predict(first_level_B.iloc[:,:-1])

training_mse = mse(first_level_B.iloc[:,-1], training_prediction)

print('2nd level model RMSE: {}'.format(training_mse**0.5))



submission = final_model.predict(first_level_C.iloc[:,:])

# transform it back

submission = np.exp(submission) - 1

submission_df = pd.DataFrame({'Id':test_cleaned.Id.astype(np.int32) , 'SalePrice':submission})

submission_df.to_csv('./submission.csv', index=False)

from IPython.display import FileLink

FileLink(r'./submission.csv')