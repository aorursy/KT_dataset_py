# Imports
%matplotlib inline
%reload_ext autoreload
%autoreload 2
import xgboost as xgb
import lightgbm as lgb
from fastai.structured import *
from fastai.column_data import *
np.set_printoptions(threshold=50, edgeitems=20)
from IPython.display import HTML, display
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
from sklearn.model_selection import StratifiedKFold
# Load the csv
PATH = '../input/'
train_df = pd.read_csv(PATH+'train.csv')
test_df = pd.read_csv(PATH+'test.csv')
test_df.shape
# Classify our labels:
cat_vars = ['hacdor',
            'hacapo',
            'v14a',
            'refrig',
            'v18q',
            'paredblolad',
            'paredzocalo',
            'paredpreb',
            'pareddes',
            'paredmad',
            'paredzinc',
            'paredfibras',
            'paredother',
            'pisomoscer',
            'pisocemento',
            'pisoother',
            'pisonatur',
            'pisonotiene',
            'pisomadera',
            'techozinc',
            'techoentrepiso',
            'techocane',
            'techootro',
            'cielorazo',
            'abastaguadentro',
            'abastaguafuera',
            'abastaguano',
            'public',
            'planpri',
            'noelec',
            'coopele',
            'sanitario1',
            'sanitario2',
            'sanitario3',
            'sanitario5',
            'sanitario6',
            'energcocinar1',
            'energcocinar2',
            'energcocinar3',
            'energcocinar4',
            'elimbasu1',
            'elimbasu2',
            'elimbasu3',
            'elimbasu4',
            'elimbasu5',
            'elimbasu6',
            'epared1',
            'epared2',
            'epared3',
            'etecho1',
            'etecho2',
            'etecho3',
            'eviv1',
            'eviv2',
            'eviv3',
            'dis',
            'estadocivil1',
            'estadocivil2',
            'estadocivil3',
            'estadocivil4',
            'estadocivil5',
            'estadocivil6',
            'estadocivil7',
            'parentesco1',
            'parentesco2',
            'parentesco3',
            'parentesco4',
            'parentesco5',
            'parentesco6',
            'parentesco7',
            'parentesco8',
            'parentesco9',
            'parentesco10',
            'parentesco11',
            'parentesco12',
            'dependency',
            'edjefe',
            'edjefa',
            'instlevel1',
            'instlevel2',
            'instlevel3',
            'instlevel4',
            'instlevel5',
            'instlevel6',
            'instlevel7',
            'instlevel8',
            'instlevel9',
            'tipovivi1',
            'tipovivi2',
            'tipovivi3',
            'tipovivi4',
            'tipovivi5',
            'computer',
            'television',
            'mobilephone',
            'lugar1',
            'lugar2',
            'lugar3',
            'lugar4',
            'lugar5',
            'lugar6',
            'area1',
            'area2']

contin_vars = ['v2a1',
               'rooms',
               'v18q1',
               'r4h1',
               'r4h2',
               'r4h3',
               'r4m1',
               'r4m2',
               'r4m3',
               'r4t1',
               'r4t2',
               'r4t3',
               'tamhog',
               'escolari',
               'rez_esc',
               'hhsize',
               'male',
               'female',
               'hogar_nin',
               'hogar_adul',
               'hogar_mayor',
               'hogar_total',
               'meaneduc',
               'bedrooms',
               'overcrowding',
               'qmobilephone',
               'age',
               'SQBescolari',
               'SQBage',
               'SQBhogar_total',
               'SQBedjefe',
               'SQBhogar_nin',
               'SQBovercrowding',
               'SQBdependency',
               'SQBmeaned',
               'agesq']

objective= ['Target']

ids = ['Id']
# Apply categorical type:
for v in cat_vars: train_df[v] = train_df[v].astype('category').cat.as_ordered()

apply_cats(test_df, train_df)
# Contin_vars as floats:
for v in contin_vars:
    train_df[v] = train_df[v].fillna(0).astype('float32')
    test_df[v] = test_df[v].fillna(0).astype('float32')
# Create new features per individuals:
# Credits to this notebook in R for the features:
# https://www.kaggle.com/taindow/predicting-poverty-levels-with-r
for dataframe in (train_df, test_df):
    dataframe['Rent_per_individual'] = dataframe['v2a1']/dataframe['r4t3']
    dataframe['Rent_per_child'] = dataframe['v2a1']/dataframe['r4t1']
    dataframe['Rent_per_over_65'] = dataframe['v2a1']/dataframe['r4t3']
    dataframe['Rent_per_room'] = dataframe['v2a1']/dataframe['rooms']
    dataframe['Rent_per_bedrooms'] = dataframe['v2a1']/dataframe['bedrooms']
    dataframe['Proportion_under_12'] = dataframe['r4t1']/dataframe['r4t3']
    dataframe['Proportion_under_12_male'] = dataframe['r4h1']/dataframe['r4t3']
    dataframe['Proportion_under_12_female'] = dataframe['r4m1']/dataframe['r4t3']
    dataframe['Proportion_male'] = dataframe['r4h3']/dataframe['r4t3']
    dataframe['Proportion_female'] = dataframe['r4m3']/dataframe['r4t3']
    dataframe['Rooms_per_individual'] = dataframe['rooms']/dataframe['r4t3']
    dataframe['Rooms_per_child'] = dataframe['rooms']/dataframe['r4t1']
    dataframe['Tablets_per_individual'] = dataframe['v18q1']/dataframe['r4t3']
    dataframe['Tablets_per_child'] = dataframe['v18q1']/dataframe['r4t1']
    dataframe['Years_schooling_per_individual'] = dataframe['escolari']/dataframe['r4t3']
    dataframe['Years_schooling_per_adult'] = dataframe['escolari']/(dataframe['r4t3']-dataframe['r4t1'])
    dataframe['Years_schooling_per_child'] = dataframe['escolari']/dataframe['r4t1']
    dataframe['Proportion_under_19'] = dataframe['hogar_nin']/dataframe['r4t3']
    dataframe['Proportion_over_19'] = dataframe['hogar_adul']/dataframe['r4t3']
    dataframe['Proportion_under_65'] = (dataframe['hogar_total']-dataframe['hogar_mayor'])/dataframe['r4t3']
    dataframe['Proportion_over_65'] = dataframe['hogar_mayor']/dataframe['r4t3']
    dataframe['Bedrooms_per_individual'] = dataframe['bedrooms']/dataframe['r4t3']
    dataframe['Bedrooms_per_child'] = dataframe['bedrooms']/dataframe['r4t1']
    dataframe['Bedrooms_per_over_65'] = dataframe['bedrooms']/dataframe['r4t3']
    dataframe['Extreme_conditions_flag'] = (dataframe['abastaguano'] & dataframe['noelec'] & dataframe['sanitario1'] & dataframe['energcocinar1'])
    dataframe['bedrooms_to_rooms'] = dataframe['bedrooms']/dataframe['rooms']
    dataframe['tamhog_to_rooms'] = dataframe['tamhog']/dataframe['rooms']
    dataframe['tamhog_to_bedrooms'] = dataframe['tamhog']/dataframe['bedrooms']
    dataframe['r4t3_to_tamhog'] = dataframe['r4t3']/dataframe['tamhog']
    dataframe['hhsize_to_rooms'] = dataframe['hhsize']/dataframe['rooms']
    dataframe['hhsize_to_bedrooms'] = dataframe['hhsize']/dataframe['bedrooms']
    dataframe['rent_to_hhsize'] = dataframe['v2a1']/dataframe['hhsize']
    dataframe['qmobilephone_to_r4t3'] = dataframe['qmobilephone']/dataframe['r4t3']
    dataframe['qmobilephone_to_v18q1'] = dataframe['qmobilephone']/dataframe['v18q1']
# Features per Household on the training set:
groupper = train_df.groupby('idhogar')
interactions = (pd.DataFrame(dict(
                head_partner_escolari=train_df.parentesco2.astype(int) * train_df.escolari.astype(int)))
                .groupby(train_df.idhogar)
                .max())
my_features = (groupper.mean()[['escolari', 'age', 'male', 'meaneduc', 'SQBdependency']]
                .join(groupper.std()[['escolari', 'age']], 
                      rsuffix='_std')
                .join(groupper[['escolari', 'age']].min(), rsuffix="_min")
                .join(groupper[['escolari', 'age']].max(), rsuffix="_max")
                .join(groupper[['dis']].sum())
                .join(interactions)
                .join(groupper[['computer', 'television', 'qmobilephone', 'v18q1']].mean().sum(axis=1).rename('technics')))
my_features.rename(columns={'escolari': 'escolari_mean', 'age': 'age_mean', 'male': 'male_mean', 'dis': 'dis_sum', 
                            'meaneduc': 'meaneduc_mean', 'SQBdependency': 'SQBdependency_mean'}, inplace=True)
train_df = train_df.merge(my_features, how='left', left_on='idhogar', right_on='idhogar')
# Features per Household on the testing set:
groupper = test_df.groupby('idhogar')
interactions = (pd.DataFrame(dict(
                head_partner_escolari=test_df.parentesco2.astype(int) * test_df.escolari.astype(int)))
                .groupby(test_df.idhogar)
                .max())
my_features = (groupper.mean()[['escolari', 'age', 'male', 'meaneduc', 'SQBdependency']]
                .join(groupper.std()[['escolari', 'age']], 
                      rsuffix='_std')
                .join(groupper[['escolari', 'age']].min(), rsuffix="_min")
                .join(groupper[['escolari', 'age']].max(), rsuffix="_max")
                .join(groupper[['dis']].sum())
                .join(interactions)
                .join(groupper[['computer', 'television', 'qmobilephone', 'v18q1']].mean().sum(axis=1).rename('technics')))
my_features.rename(columns={'escolari': 'escolari_mean', 'age': 'age_mean', 'male': 'male_mean', 'dis': 'dis_sum',
                            'meaneduc': 'meaneduc_mean', 'SQBdependency': 'SQBdependency_mean'}, inplace=True)
test_df = test_df.merge(my_features, how='left', left_on='idhogar', right_on='idhogar')
list(train_df)
train_df.drop('idhogar', axis=1, inplace=True)
test_df.drop('idhogar', axis=1, inplace=True)
new_features = ['Rent_per_individual',
                'Rent_per_child',
                'Rent_per_over_65',
                'Rent_per_room',
                'Rent_per_bedrooms',
                'Proportion_under_12',
                'Proportion_under_12_male',
                'Proportion_under_12_female',
                'Proportion_male',
                'Proportion_female',
                'Rooms_per_individual',
                'Rooms_per_child',
                'Tablets_per_individual',
                'Tablets_per_child',
                'Years_schooling_per_individual',
                'Years_schooling_per_adult',
                'Years_schooling_per_child',
                'Proportion_under_19',
                'Proportion_over_19',
                'Proportion_under_65',
                'Proportion_over_65',
                'Bedrooms_per_individual',
                'Bedrooms_per_child',
                'Bedrooms_per_over_65',
                'Extreme_conditions_flag',
                'bedrooms_to_rooms',
                'tamhog_to_rooms',
                'tamhog_to_bedrooms',
                'r4t3_to_tamhog',
                'hhsize_to_rooms',
                'hhsize_to_bedrooms',
                'rent_to_hhsize',
                'qmobilephone_to_r4t3',
                'qmobilephone_to_v18q1',
                'escolari_mean',
                'age_mean',
                'male_mean',
                'meaneduc_mean',
                'SQBdependency_mean',
                'escolari_std',
                'age_std',
                'escolari_min',
                'age_min',
                'escolari_max',
                'age_max',
                'dis_sum',
                'head_partner_escolari',
                'technics']
# Treat the new features as contin_vars:
for v in new_features:
    train_df[v] = train_df[v].replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')
    test_df[v] = test_df[v].replace([np.inf, -np.inf], np.nan).fillna(0).astype('float32')
# Process the training data using the awesome fastai function proc_df:
train_df = train_df.set_index('Id')
train_df = train_df[cat_vars+contin_vars+new_features+objective]

df, y, nas, mapper = proc_df(train_df, 'Target', do_scale=True)
# Process the testing data using the awesome fastai function proc_df:
test_df = test_df.set_index('Id')

# Just a dummy column so that the column exists.
test_df['Target'] = 0
test_df = test_df[cat_vars+contin_vars+new_features+['Target']]

df_test, _, nas, mapper = proc_df(test_df, 'Target', do_scale=True,
                                  mapper=mapper, na_dict=nas)
# Create a K-fold instance
k = 5
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=1)
# Train/validate on 5 different training/validation sets thanks to K-folds and predict on the testing set:
predicts = []
test_X = np.array(df_test)
for train_index, test_index in kf.split(df, y):
    print("###")
    X_train, X_val = np.array(df)[train_index], np.array(df)[test_index]
    y_train, y_val = y[train_index], y[test_index]
    
    xgb_params = {
        'learning_rate': 0.1,
        'n_estimators': 1000,
        'objective': 'multi:softmax',
        'eval_metric': 'merror',
        'silent': 1,
        'num_class': 5,
        'seed': 27}
    
    d_train = xgb.DMatrix(X_train, y_train)
    d_valid = xgb.DMatrix(X_val, y_val)
    d_test = xgb.DMatrix(test_X)
    
    model = xgb.train(xgb_params, d_train, num_boost_round = 10000, evals=[(d_valid, 'eval')], verbose_eval=100, 
                     early_stopping_rounds=50)
                        
    xgb_pred = model.predict(d_test)
    predicts.append(list(xgb_pred))
# Average the 5 predictions sets:
preds=[]
for i in range(len(predicts[0])):
    sum=0
    for j in range(k):
        sum+=predicts[j][i]
    preds.append(sum / k)
# Create the results datframe:
df_test['Target'] = np.array(preds).astype(int)
sub = df_test[['Target']].copy()
sub.reset_index(inplace=True, drop=False)
sub.head()
sub.to_csv('sample_submission.csv', index=False)