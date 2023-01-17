import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import scale 

from sklearn.linear_model import ElasticNet

from sklearn.preprocessing import scale

from sklearn import metrics

from tqdm import tqdm

from bisect import bisect_right

from sklearn.externals.six import StringIO  

from IPython.display import Image  

from sklearn.tree import export_graphviz

from IPython.display import Image, display

from scipy import stats

!pip install pydotplus

import pydotplus

import os 

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor



pd.options.display.max_rows = 4000

pd.options.display.max_seq_items = 2000

pd.options.display.float_format = '{:.4f}'.format
def draw_feature(selected_cols):

        

        clf = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, criterion = 'mae')

        clf.fit(X[selected_cols], Y)



        pred_te = clf.predict(X_test[selected_cols])

        print("MAE Test Model: %.4f" % metrics.mean_absolute_error(Y_test, pred_te))

        print("MAE Test Naive: %.4f" % metrics.mean_absolute_error(Y_test, np.repeat(np.mean(Y_train), len(Y_test))))

        

        dot_data = StringIO()

        export_graphviz(clf, out_file=dot_data, feature_names = selected_cols,

                        filled=True, rounded=True, proportion = True, impurity = False,

                        special_characters=True,

                        class_names = ['GOOD', 'BAD']

                       )

        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

        graph.write_png('original_tree.png')

        graph.set_size('"20,!"')

        plt = Image(graph.create_png())

        display(plt)



def scale_tr_te(X_train_enc, X_test_enc):



    X_train_enc = scale(X_train_enc)

    X_test_enc = scale(X_test_enc)



    return X_train_enc, X_test_enc



def calc_rank_table_and_predict(data, zero_reg, cols):

   

    data_prep = data[(data['inf_rate'].notnull())]

    all_cols = list(data_prep.columns)

    all_cols.remove('has_metro')

    model_data1 = data_prep.groupby('subject')[all_cols].median()

    model_data1['has_metro'] = data_prep.groupby('subject')['has_metro'].max()

    app = ['ivl_per_100k',	'ekmo_per_100k']

    model_data1[app] = model_data1[app].mask(model_data1[app]==0).fillna(model_data1[app].mean())

    model_data1['pred'] = 0



    Y = model_data1['inf_rate']

    X = model_data1.drop(['inf_rate'], axis = 1)

    X = X.fillna(X.mean())



    clf = ElasticNet(alpha=0.25, l1_ratio=0.5, random_state = 0)

    clf.fit(scale(X[cols]), Y.values)



    data_prep = data[(data['inf_rate'].isnull())]

    all_cols = list(data_prep.columns)

    all_cols.remove('has_metro')

    model_data2 = data_prep.groupby('subject')[all_cols].median()

    model_data2['has_metro'] = data_prep.groupby('subject')['has_metro'].max()

    app = ['ivl_per_100k',	'ekmo_per_100k']

    model_data2[app] = model_data2[app].mask(model_data2[app]==0).fillna(model_data2[app].mean())

    model_data2[cols] = model_data2[cols].fillna(model_data2.mean())



    zero_cols = ['urban_65-69_years', 'urban_80-84_years', 'urban_70-74_years', 

          'urban_85-89_years', 'urban_60-64_years', 'urban_75-79_years', 'humidity_max']

    model_data2[zero_cols]=zero_reg.set_index('subject')[zero_cols]

    model_data2['inf_rate'] = clf.predict(scale(model_data2[cols]))

    model_data2['pred'] = 1



    model_data = pd.concat((model_data1, model_data2), axis = 0)

    df = pd.DataFrame()

    df = model_data[['inf_rate', 'epirank_avia', 'epirank_train','ivl_per_100k', 'ekmo_per_100k', 'public_transport', 'shops_and_malls', 'has_metro', 'pred']]

    df = df.fillna(df.median())

    df['inf_rate_rank'] = df['inf_rate'].rank(method='dense', ascending=False).astype(int)

    df['epirank_avia_rank'] = df['epirank_avia'].rank(method='dense', ascending=False).astype(int)

    df['epirank_train_rank'] = df['epirank_train'].rank(method='dense', ascending=False).astype(int)

    df['ivl_per_100k_rank'] = df['ivl_per_100k'].rank(method='dense', ascending=True).astype(int)

    df['ekmo_per_100k_rank'] = df['ekmo_per_100k'].rank(method='dense', ascending=True).astype(int)

    df['public_transport_rank'] = df['public_transport'].rank(method='dense', ascending=False).astype(int)

    df['shops_and_malls_rank'] = df['shops_and_malls'].rank(method='dense', ascending=False).astype(int)

    df['joint_rank']=df.filter(regex='_rank').iloc[:,1:].median(axis=1) + df['inf_rate_rank']

    

    return df
path = '/kaggle/input/'

data = pd.read_csv(path + 'reproduction-rate-russia-regions/data_v4.csv')

zero_reg = pd.read_csv(path + 'reproduction-rate-add-data-for-prediction/gks_metrics_zero_reg.csv')
data.head(20)
use_cols = ['population', 'density', 

        'lat', 'lng',

       'life_quality_place_rating', 'ecology', 'cleanness', 'public_services',

       'neighbourhood', 'children_places', 'sport_and_outdoor',

       'inf_rate',

       'avg_temp_min', 'avg_temp_max', 'avg_temp_std', 'avg_temp_median',

       'humidity_min', 'humidity_max', 'humidity_std', 'humidity_median',

       'pressure_min', 'pressure_max', 'pressure_std', 'pressure_median',

       'wind_speed_ms_min', 'wind_speed_ms_max', 'wind_speed_ms_std',

       'wind_speed_ms_median', 

       'urban_50-54_years', 'urban_55-59_years', 'urban_60-64_years',

       'urban_65-69_years', 'urban_70-74_years', 'urban_75-79_years',

       'urban_80-84_years', 'urban_85-89_years', 'urban_90-94_years',

       'rural_50-54_years', 'rural_55-59_years', 'rural_60-64_years',

       'rural_65-69_years', 'rural_70-74_years', 'rural_75-79_years',

       'rural_80-84_years', 'rural_85-89_years', 'rural_90-94_years',

       'work_ratio_15-72_years',

       'work_ratio_55-64_years', 'work_ratio_15-24_years',

       'work_ratio_15-64_years', 'work_ratio_25-54_years', 

      #  'num_patients_tubercul_1992-2017_per_year',

      #  'num_patients_tubercul_1992', 'num_patients_tubercul_1993',

      #  'num_patients_tubercul_1994', 'num_patients_tubercul_1995',

      #  'num_patients_tubercul_1996', 'num_patients_tubercul_1997',

      #  'num_patients_tubercul_1998', 'num_patients_tubercul_1999',

      #  'num_patients_tubercul_2000', 'num_patients_tubercul_2001',

      #  'num_patients_tubercul_2002', 'num_patients_tubercul_2003',

      #  'num_patients_tubercul_2004', 'num_patients_tubercul_2005',

      #  'num_patients_tubercul_2006', 'num_patients_tubercul_2007',

      #  'num_patients_tubercul_2008', 'num_patients_tubercul_2009',

      #  'num_patients_tubercul_2010', 'num_patients_tubercul_2011',

      #  'num_patients_tubercul_2012', 'num_patients_tubercul_2013',

      #  'num_patients_tubercul_2014', 'num_patients_tubercul_2015',

      #  'num_patients_tubercul_2016', 'num_patients_tubercul_2017',

      #  'volume_serv_household_2017',

      #  'volume_serv_chargeable_2017', 'volume_serv_transport_2017',

      #  'volume_serv_post_2017', 'volume_serv_accommodation_2017',

      #  'volume_serv_telecom_2017', 'volume_serv_others_2017',

      #  'volume_serv_veterinary_2017', 'volume_serv_housing_2017',

      #  'volume_serv_education_2017', 'volume_serv_medicine_2017',

      #  'volume_serv_disabled_2017', 'volume_serv_culture_2017',

      #  'volume_serv_sport_2017', 'volume_serv_hotels_2017',

      #  'volume_serv_tourism_2017', 'volume_serv_sanatorium_2017',

       'num_phones_rural_2018', 'num_phones_urban_2019',

       'bus_march_travel_18', 'bus_april_travel_18',

       'epirank_avia', 'epirank_bus', 'epirank_train', 'epirank_avia_cat',

       'epirank_bus_cat', 'epirank_train_cat',

       'whole_population', 'urban', 'rural', 'has_metro']    
data['num_patients_tubercul_1992-2017_per_year'] = data.filter(regex='tubercul').sum(axis=1)/26
data_prep = data[(data['inf_rate'].notnull())]

data_prep = data_prep[(data_prep['inf_rate']>0)]

data_prep[use_cols] = data_prep[use_cols].astype('float')

model_data = data_prep.groupby('subject')[use_cols[:-1]].median()

model_data['has_metro'] = data_prep.groupby('subject')[use_cols[-1:]].max()
Y = model_data['inf_rate']

X = model_data.drop(['inf_rate'], axis = 1)

X = X.fillna(X.mean())

use_cols = list(X.columns)
from sklearn.preprocessing import scale

X_train, X_test, Y_train, Y_test = train_test_split(X, Y.values, test_size=10, random_state=42)

X_train_enc_sc, X_test_enc_sc = scale_tr_te(X_train, X_test)
from sklearn.model_selection import RepeatedKFold

from sklearn import metrics

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet

import lightgbm as lgb

from sklearn.svm import LinearSVR

import xgboost as xgb

from sklearn import ensemble



NFOLDS = 3

RANDOM_STATE = 42

N_REPEATS = 40



clfs = []

mae_mod = []

mae_bas = []

folds = RepeatedKFold(n_splits=NFOLDS, n_repeats = N_REPEATS, random_state=RANDOM_STATE)

oof_preds_mod = np.zeros((len(X_train), 1))

test_preds_mod = np.zeros((len(X_test), 1))

oof_preds_bas = np.zeros((len(X_train), 1))

test_preds_bas = np.zeros((len(X_test), 1))



for fold_, (trn_, val_) in tqdm(enumerate(folds.split(X_train, Y_train))):

    

    X_trn, Y_trn = X_train.iloc[trn_, :], Y_train[trn_]

    X_val, Y_val = X_train.iloc[val_, :], Y_train[val_]



    X_trn_enc_sc, X_val_enc_sc = scale_tr_te(X_trn, X_val)

    clf = ElasticNet(alpha=0.25, l1_ratio=0.5, random_state = 42)

    clf.fit(X_trn_enc_sc, Y_trn)

    

    val_pred = clf.predict(X_val_enc_sc)



    X_test_enc_sc = pd.DataFrame(X_test_enc_sc, columns = use_cols)

    test_fold_pred = clf.predict(X_test_enc_sc)



    mae_mod.append(metrics.mean_absolute_error(Y_val, val_pred))

    mae_bas.append(metrics.mean_absolute_error(Y_val, np.repeat(np.mean(Y_trn), len(val_))))

    oof_preds_mod[val_, :] = val_pred.reshape((-1, 1))

    oof_preds_bas[val_, :] = np.repeat(np.mean(Y_trn), len(val_)).reshape((-1, 1))

    test_preds_mod += test_fold_pred.reshape((-1,1))

    test_preds_bas += np.repeat(np.mean(Y_trn), len(X_test_enc_sc)).reshape((-1, 1))



test_preds_mod /= NFOLDS * N_REPEATS

test_preds_bas /= NFOLDS * N_REPEATS

print('')



mae_score_cv_mod = np.round(metrics.mean_absolute_error(Y_train, oof_preds_mod.ravel()),4)

print('')

print("MAE OOF Model = {}".format(np.round(mae_score_cv_mod, 4)))

print("MAE CV Model = {}".format(np.round(np.mean(mae_mod), 4)))

print("MAE STD Model = {}".format(np.round(np.std(mae_mod),4)))



mae_score_cv_bas = np.round(metrics.mean_absolute_error(Y_train, oof_preds_bas.ravel()),4)

print('')

print("MAE OOF Baseline = {}".format(np.round(mae_score_cv_bas, 4)))

print("MAE CV Baseline = {}".format(np.round(np.mean(mae_bas), 4)))

print("MAE STD Baseline = {}".format(np.round(np.std(mae_bas),4)))



print('')

mae_score_test_mod = np.round(metrics.mean_absolute_error(Y_test, test_preds_mod),4)

print("MAE Test Model = {}".format(mae_score_test_mod))

mae_score_test_bas = np.round(metrics.mean_absolute_error(Y_test, test_preds_bas),4)

print("MAE Test Baseline = {}".format(mae_score_test_bas))
from sklearn.preprocessing import scale 

from sklearn.linear_model import ElasticNet

from sklearn import metrics



clf = ElasticNet(alpha=0.25, l1_ratio=0.5, random_state = 42)

clf.fit(X_train_enc_sc, Y_train)

pred_te = clf.predict(X_test_enc_sc)

print("MAE Test Naive: %.4f" % metrics.mean_absolute_error(Y_test, np.repeat(np.mean(Y_train), len(Y_test))))

print("MAE Test Model: %.4f" % metrics.mean_absolute_error(Y_test, pred_te))



print(80*'=')
coefficients = pd.concat([pd.DataFrame(X_train.columns),pd.DataFrame(np.transpose(clf.coef_))], axis = 1)

coefficients.columns = ['variable', 'weight']

coefficients['percent'] = np.abs(coefficients['weight'])

coefficients['percent'] /= coefficients['percent'].sum()

coefficients.sort_values(by = 'percent', ascending = False).head(18)
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(20, 10))

sns.barplot(x="percent", y="variable", data=coefficients.sort_values(by="percent", ascending=False).head(15))

plt.title('Feature Importance')

plt.tight_layout()
# cols = coefficients.sort_values(by="percent", ascending=False).head(10)['variable'].unique()

cols = ['urban_65-69_years', 'urban_80-84_years', 'urban_70-74_years', 

          'urban_85-89_years', 'urban_60-64_years', 'urban_75-79_years', 'humidity_max']
draw_feature(cols)
cols = ['urban_65-69_years', 'urban_80-84_years', 'urban_70-74_years', 'lat',

          'urban_85-89_years', 'urban_60-64_years', 'urban_75-79_years', 'humidity_max']



rank_table = calc_rank_table_and_predict(data, zero_reg, cols)



rank_table[rank_table['pred']==0].sort_values(by='joint_rank', ascending = True).head(20)
rank_table[rank_table['pred']==1].sort_values(by='joint_rank', ascending = True)[['inf_rate']]
data_prep = data[(data['inf_rate'].notnull())]

use_cols = ['lat', 'urban', 'inf_rate', 'humidity_max', 'population', 'density', 'epirank_avia', 'epirank_train', 'urban_65-69_years']

data_prep[use_cols] = data_prep[use_cols].astype('float')

model_data = data_prep.groupby('subject')[use_cols].median()
rvs1 = model_data[(model_data['lat']>=55.442) & (model_data['urban']>300000) & (model_data['inf_rate'].notnull())]['inf_rate']

print(rvs1)
rvs2 = model_data[((model_data['lat']<55.442) & (model_data['urban']>300000)) &  (model_data['inf_rate'].notnull())]['inf_rate']

print(rvs2)
stats.mannwhitneyu(rvs1, rvs2)
rvs1 = model_data[(model_data['lat']>=55.442) & (model_data['urban']>300000) & (model_data['inf_rate'].notnull())]['humidity_max']

rvs2 = model_data[((model_data['lat']<55.442) & (model_data['urban']>300000)) &  (model_data['inf_rate'].notnull())]['humidity_max']

print(rvs1)

print('')

print(rvs2)

print('')

stats.mannwhitneyu(rvs1, rvs2)
rvs1 = model_data[(model_data['lat']>=55.442) & (model_data['urban']>300000) & (model_data['inf_rate'].notnull())]['density']

rvs2 = model_data[((model_data['lat']<55.442) & (model_data['urban']>300000)) &  (model_data['inf_rate'].notnull())]['density']

stats.mannwhitneyu(rvs1, rvs2)
rvs1 = model_data[(model_data['lat']>=55.442) & (model_data['urban']>300000) & (model_data['inf_rate'].notnull())]['epirank_train']

rvs2 = model_data[((model_data['lat']<55.442) & (model_data['urban']>300000)) &  (model_data['inf_rate'].notnull())]['epirank_train']

print(stats.mannwhitneyu(rvs1, rvs2))



rvs1 = model_data[(model_data['lat']>=55.442) & (model_data['urban']>300000) & (model_data['inf_rate'].notnull())]['epirank_avia']

rvs2 = model_data[((model_data['lat']<55.442) & (model_data['urban']>300000)) &  (model_data['inf_rate'].notnull())]['epirank_avia']

print(stats.mannwhitneyu(rvs1, rvs2))
rvs1 = model_data[(model_data['lat']>=55.442) & (model_data['urban']>300000) & (model_data['inf_rate'].notnull())]['urban_65-69_years']

rvs2 = model_data[((model_data['lat']<55.442) & (model_data['urban']>300000)) &  (model_data['inf_rate'].notnull())]['urban_65-69_years']

print(stats.mannwhitneyu(rvs1, rvs2))