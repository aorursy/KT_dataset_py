#DataLoader
import pandas as pd
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')

# train_df.columns
!pip install category_encoders
import category_encoders as ce

to_transform = train_df.select_dtypes(include = 'object').columns
target_enc = ce.TargetEncoder(cols = to_transform)

target_enc.fit(train[to_transform],train['SalePrice'])

train = train.join(target_enc.transform(train[to_transform]).add_suffix('_target'))
test = test.join(target_enc.transform(test[to_transform]).add_suffix('_target'))
train = train.select_dtypes(exclude = 'object')
test = test.select_dtypes(exclude = 'object')

train.fillna(value = 0, inplace = True)
test.fillna(value = 0, inplace = True)


from sklearn.preprocessing import StandardScaler

stdize = StandardScaler()

feature_cols = train.columns.drop('SalePrice')
Xtrain,ytrain = train[feature_cols],train['SalePrice']
Xtest = test

Xtrain = stdize.fit_transform(Xtrain)
Xtest = stdize.transform(Xtest)



from sklearn.linear_model import RidgeCV
alphas = tuple([0.01*(3**x) for x in range(8)])
estimator = RidgeCV(alphas = alphas)

estimator.fit(Xtrain, ytrain)

predictions = estimator.predict(Xtest)
predictions = pd.DataFrame(predictions, index = test.index)
predictions.rename(columns = {0:'SalePrice'}, inplace = True)
predictions.reset_index()
predictions.to_csv('predict.csv')