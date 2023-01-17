# Import

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



# Loading

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')



# Datasets & Labels

y_train = df_train['SalePrice'].values

id_train = df_train['Id'].values

id_test = df_test['Id'].values

all_data = pd.concat([df_train.drop(['Id', 'SalePrice'], axis=1), df_test.drop(['Id'], axis=1)])



# Shape

print('Training Label Shape:\t{}'.format(y_train.shape))

print('Training Data Shape:\t{}'.format(df_train.shape))

print('Testing Data Shape:\t{}'.format(df_test.shape))

print('All Data Shape:\t{}'.format(all_data.shape))
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

all_data["Alley"] = all_data["Alley"].fillna("None")

all_data["Fence"] = all_data["Fence"].fillna("None")

all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")



#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

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



df = all_data



# Columns with non-numeric data

non_numeric_columns = [col for col in df.columns if all_data[col].dtype=='object']

numeric_columns = [col for col in df.columns if all_data[col].dtype!='object']
# Create LabelEncoder for each non-numeric column

from sklearn.preprocessing import LabelEncoder

encoder = {col:LabelEncoder().fit(np.hstack([df[col].values, np.array(['Missing'])])) for col in non_numeric_columns}



# Use LabelEncoders for each non-numeric column

for col in non_numeric_columns:

    df[col] = encoder[col].transform(df[col].values)

    

    

# Create StandardScaler for each numeric column

from sklearn.preprocessing import StandardScaler

scaler = {col:StandardScaler().fit(df[col].values.reshape(-1, 1)) for col in numeric_columns}



# Use StandardScaler for each numeric column

for col in numeric_columns:

    df[col] = scaler[col].transform(df[col].values.reshape(-1, 1))

    

    

# Index of the non-numeric columns

index_non_numeric_columns = [df.columns.get_loc(col) for col in non_numeric_columns]



# One-Hot-Encoding for non-numeric columns

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features=index_non_numeric_columns)

X = ohe.fit_transform(df.values).toarray()



X_all = X[:df_train.shape[0]]

X_pred = X[df_train.shape[0]:]
# Train-Test-Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_all, y_train, test_size=0.3, random_state=0)
# Cross-Validation

from sklearn.model_selection import KFold

skf = KFold(n_splits=2, random_state=1, shuffle=True)





# Start values for the evolution

loc_C, loc_gamma = 0, 0

scale_C, scale_gamma = 1, 1



# Evolution

data = []

last = -10000

from sklearn.svm import SVR

for i in range(1, 151):

    

    # n Random new values for the hyperparameters

    n = 25

    all_C = np.abs(np.random.normal(loc=loc_C, scale=scale_C, size=n))

    all_gamma = np.abs(np.random.normal(loc=loc_gamma, scale=scale_gamma, size=n))

    

    tests = []

    # Iterate over the population

    for C, gamma in zip(all_C, all_gamma):

        svr_rbf = SVR(kernel='rbf', C=C, gamma=gamma)

    

        scores = []

        # Cross-Validation

        for train_index, test_index in skf.split(X_train, y_train):

            svr_rbf.fit(X_train[train_index], y_train[train_index])

            rbf_score = svr_rbf.score(X_train[test_index], y_train[test_index])



            scores.append(rbf_score)

        tests.append([np.mean(scores), C, gamma])

    # Rank the population 

    tests.sort(key=lambda x: x[0], reverse=True)

    data.append(tests[0])

    # Prepare the next generation

    best = np.array(tests)[:3, 1:3]

    # Update values with momentum

    flex = 0.25

    if tests[0][0]>last:

        loc_C = (1-flex)*loc_C + flex*best.mean(axis=0)[0]

        loc_gamma = (1-flex)*loc_gamma + flex*best.mean(axis=0)[1]

        scale_C =(1-flex)*scale_C +  flex*np.abs(np.diff(best, axis=0).mean(axis=0))[0]*5

        scale_gamma =(1-flex)*scale_gamma +  flex*np.abs(np.diff(best, axis=0).mean(axis=0))[1]*5

        last = tests[0][0]

    print(i, tests[0])

    # epoch-count ['cross-validation-score', 'C', 'gamma']
# Top 5

unordered_data = data.copy()

data.sort(key=lambda x: x[0], reverse=True)

data[:5]
# Retrain the best parameters

svr_rbf = SVR(kernel='rbf', C=data[0][1], gamma=data[0][2])

    

scores = []

for train_index, test_index in skf.split(X_train, y_train):

    svr_rbf.fit(X_train[train_index], y_train[train_index])

    rbf_score = svr_rbf.score(X_train[test_index], y_train[test_index])

    scores.append(rbf_score)

print(np.mean(scores))
# Predict and submit (Best Score: 0.13030)

predictions = svr_rbf.predict(X_pred)



ids = pd.read_csv('../input/test.csv')['Id'].values



df = pd.DataFrame()

df['Id'] = ids

df['SalePrice'] = predictions

df.to_csv('submission.csv', index=False)
import matplotlib.pyplot as plt

plt.style.use('ggplot')

score, C, gamma = zip(*unordered_data)



f, axarr = plt.subplots(3, sharex=True, figsize=(12,8))

axarr[0].set_title('Score - Evolution')

axarr[0].plot(range(len(score)), score)

axarr[1].set_yscale('log')

axarr[1].set_title('C - Evolution')

axarr[1].plot(range(len(C)), C)

axarr[2].set_yscale('log')

axarr[2].set_title('Gamma - Evolution')

axarr[2].plot(range(len(gamma)), gamma)

plt.tight_layout()

plt.show()