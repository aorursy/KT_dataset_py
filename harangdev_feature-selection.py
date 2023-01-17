import numpy as np

import pandas as pd



from lightgbm import LGBMRegressor

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



from sklearn.ensemble import RandomForestRegressor

from boruta import BorutaPy

import shap

import eli5

from eli5.sklearn import PermutationImportance



import matplotlib.pyplot as plt



import warnings

import gc

warnings.filterwarnings("ignore")
data = pd.read_csv('../input/train.csv', index_col='id')

test = pd.read_csv('../input/test.csv', index_col='id')

df = pd.concat([data, test])
today = pd.to_datetime('2016-01-01')

    

# datetime features

df['was_renovated'] = (df['yr_renovated'] != 0).astype('uint8')

not_renovated = df[df['was_renovated'] == 0].index

df.loc[not_renovated, 'yr_renovated'] = df.loc[not_renovated, 'yr_built']

df['date'] = pd.to_datetime(df['date'].str[:8])

df['yr_built'] = pd.to_datetime({'year': df['yr_built'], 'month': [1]*len(df), 'day': [1]*len(df)})

df['yr_renovated'] = pd.to_datetime({'year': df['yr_renovated'], 'month': [1]*len(df), 'day': [1]*len(df)}, errors='coerce')

df['today-D-date'] = (today - df['date']).dt.days

df['today-D-yr_renovated'] = (today - df['yr_renovated']).dt.days

df['today-D-yr_built'] = (today - df['yr_built']).dt.days

df['date-D-yr_built'] = (df['date'] - df['yr_built']).dt.days

df['yr_renovated-D-yr_built'] = (df['yr_renovated'] - df['yr_built']).dt.days

df = df.drop(['date', 'yr_built', 'yr_renovated'], axis=1)



# feature interactions

df['room_count'] = df['bedrooms'] + df['bathrooms']

df['sqft_living_per_rooms'] = df['sqft_living'] / (df['room_count']+1)

df['sqft_lot_per_rooms'] = df['sqft_lot'] / (df['room_count']+1)

df['room_per_floors'] = df['room_count'] / df['floors']

df['sqft_living_per_floors'] = df['sqft_living'] / df['floors']

df['sqft_lot_per_floors'] = df['sqft_lot'] / df['floors']

df['sqft_living_per_bedrooms'] = df['sqft_living'] / (df['bedrooms']+1)

df['sqft_lot_per_bedrooms'] = df['sqft_lot'] / (df['bedrooms']+1)

df['bedroom_per_floors'] = df['bedrooms'] / df['floors']

df['sqft_lot-D-sqft_living'] = df['sqft_lot'] - df['sqft_living']

df['sqft_lot-R-sqft_living'] = df['sqft_lot'] / df['sqft_living']

df['sqft_living15-D-sqft_living'] = df['sqft_living15'] - df['sqft_living']

df['sqft_living15-R-sqft_living'] = df['sqft_living15'] / df['sqft_living']

df['sqft_lot15-D-sqft_lot'] = df['sqft_lot15'] - df['sqft_lot']

df['sqft_lot15-R-sqft_lot'] = df['sqft_lot15'] / df['sqft_lot']

df['rooms_mul']=df['bedrooms']*df['bathrooms']

df['total_score']=df['condition']+df['grade']+df['view']



# binary features

df['has_basement'] = (df['sqft_basement']>0).astype('uint8')

df['has_attic'] = ((df['floors'] % 1) != 0).astype('uint8')



# one hot encode

df['zipcode'] = df['zipcode'].astype('str')

df = pd.get_dummies(df)



# log transform target

df['price'] = np.log1p(df['price'])



data = df.loc[data.index]

test = df.loc[test.index]



del df; gc.collect();
data.head()
def rmse_expm1(pred, true):

    return -np.sqrt(np.mean((np.expm1(pred)-np.expm1(true))**2))



def evaluate(x_data, y_data):

    model = LGBMRegressor(objective='regression', num_iterations=10**5)

    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, random_state=0)

    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=100, verbose=False)

    val_pred = model.predict(x_val)

    score = rmse_expm1(val_pred, y_val)

    return score



def rfe(x_data, y_data, method, ratio=0.9, min_feats=40):

    feats = x_data.columns.tolist()

    archive = pd.DataFrame(columns=['model', 'n_feats', 'feats', 'score'])

    while True:

        model = LGBMRegressor(objective='regression', num_iterations=10**5)

        x_train, x_val, y_train, y_val = train_test_split(x_data[feats], y_data, random_state=0)

        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=100, verbose=False)

        val_pred = model.predict(x_val)

        score = rmse_expm1(val_pred, y_val)

        n_feats = len(feats)

        print(n_feats, score)

        archive = archive.append({'model': model, 'n_feats': n_feats, 'feats': feats, 'score': score}, ignore_index=True)

        if method == 'basic':

            feat_imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)

        elif method == 'perm':

            perm = PermutationImportance(model, random_state=0).fit(x_val, y_val)

            feat_imp = pd.Series(perm.feature_importances_, index=feats).sort_values(ascending=False)

        elif method == 'shap':

            explainer = shap.TreeExplainer(model)

            shap_values = explainer.shap_values(x_data[feats])

            feat_imp = pd.Series(np.abs(shap_values).mean(axis=0), index=feats).sort_values(ascending=False)

        next_n_feats = int(n_feats * ratio)

        if next_n_feats < min_feats:

            break

        else:

            feats = feat_imp.iloc[:next_n_feats].index.tolist()

    return archive



feats = [col for col in data.columns if col != 'price']

len(feats)
model = LGBMRegressor(objective='regression', num_iterations=10**5)

x_data = data[feats]

y_data = data['price']

x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, random_state=0)

model.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=100, verbose=False)

val_pred = model.predict(x_val)

score = rmse_expm1(val_pred, y_val)

score
%%time

basic_archive = rfe(x_data, y_data, 'basic')
feat_imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)

for i in range(40, 90, 5):

    print(i, evaluate(data[feat_imp.iloc[:i].index], data['price']))
%%time

perm_archive = rfe(x_data, y_data, 'perm')
perm = PermutationImportance(model, random_state=0).fit(x_val, y_val)

perm_feat_imp = pd.Series(perm.feature_importances_, index=feats).sort_values(ascending=False)

eli5.show_weights(perm)
for i in range(40, 90, 5):

    print(i, evaluate(data[perm_feat_imp.iloc[:i].index], data['price']))
%%time

shap_archive = rfe(x_data, y_data, 'shap')
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(x_data)

shap_feat_imp = pd.Series(np.abs(shap_values).mean(axis=0), index=feats).sort_values(ascending=False)

shap.summary_plot(shap_values, x_data)
for i in range(40, 90, 5):

    print(i, evaluate(data[shap_feat_imp.iloc[:i].index], data['price']))
feat_imp_archive = pd.DataFrame(index=feats, columns=['basic', 'perm', 'shap', 'mean'])

feat_imp_archive['basic'] = feat_imp.rank(ascending=False)

feat_imp_archive['perm'] = perm_feat_imp.rank(ascending=False)

feat_imp_archive['shap'] = shap_feat_imp.rank(ascending=False)

feat_imp_archive['mean'] = feat_imp_archive[['basic', 'perm', 'shap']].mean(axis=1)

feat_imp_archive = feat_imp_archive.sort_values('mean')

feat_imp_archive[feat_imp_archive['mean']<20].plot(kind='bar', figsize=(20, 10), title='feature importance rankings');
for i in range(40, 90, 5):

    print(i, evaluate(data[feat_imp_archive.iloc[:i].index], data['price']))
%%time

rf = RandomForestRegressor(n_jobs=-1, n_estimators=200, max_depth=5)

feat_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=0)

feat_selector.fit(data[feats].values, data['price'].values)
evaluate(data[np.array(feats)[feat_selector.support_]], data['price'])