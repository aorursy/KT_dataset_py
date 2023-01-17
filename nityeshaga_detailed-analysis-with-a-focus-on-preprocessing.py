import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline
# do this to make Pandas show all the columns of a DataFrame, otherwise it just shows a summary
pd.set_option('display.max_columns', None) 
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

train_id = df_train['Id']
test_id = df_test['Id']

train_idhogar = df_train['idhogar']
test_idhogar = df_train['idhogar']

df_train.drop(columns=['Id'], inplace=True)
df_test.drop(columns=['Id'], inplace=True)

print("Shape of train data: ", df_train.shape)
print("Shape of test data: ", df_test.shape)

ntrain = df_train.shape[0]
ntest = df_test.shape[0]

all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
print("A glimpse at the columns of training data:")
df_train.head()
print("The feature that we need to predict: ", set(df_train.columns) - set(df_test.columns))
df_train['Target'].describe()
def barplot_with_anotate(feature_list, y_values, plotting_space=plt, annotate_vals=None):
    x_pos = np.arange(len(feature_list))
    plotting_space.bar(x_pos, y_values);
    plotting_space.xticks(x_pos, feature_list, rotation=270);
    if annotate_vals == None:
        annotate_vals = y_values
    for i in range(len(feature_list)):
        plotting_space.text(x=x_pos[i]-0.3, y=y_values[i]+1.0, s=annotate_vals[i]);
df_train_heads = df_train.loc[df_train['parentesco1'] == 1]
poverty_label_sizes = list(df_train_heads.groupby('Target').size())

barplot_with_anotate(['extreme', 'moderate', 'vulnerable', 'non-vulnerable'], poverty_label_sizes,
                     annotate_vals = [str(round((count/df_train_heads.shape[0])*100, 2))+'%' 
                                      for count in poverty_label_sizes]);
plt.rcParams["figure.figsize"] = [6, 6];
plt.xlabel('Poverty Label');
plt.ylabel('No. of people');
def plot_dwelling_property(property_df):
    _, axarr = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row', figsize=(16, 16))

    target_idx = 0
    for row in range(2):
        for col in range(2):
            percentage_list = [round((count/poverty_label_sizes[target_idx])*100, 2)
                                 for count in list(property_df.iloc[target_idx, :])]
            x_pos = list(range(len(property_df.columns)))
            
            axarr[row, col].bar(x_pos, 
                                percentage_list, 
                                color='y')
            
            axarr[row, col].set_title('For individuals in Poverty group=' + str(target_idx+1))
            
            xtick_labels = list(property_df.columns)
            xtick_labels.insert(0, '') # insert a blank coz `set_xticklabels()` skips the 1st element ##why??
            axarr[row, col].set_xticklabels(xtick_labels, rotation=300)
            
            axarr[row, col].set_ylim(bottom=0, top=100)
            #axarr[row, col].set_xlim(left=0, right=len(property_df.columns))
            
            for i in range(len(property_df.columns)):
                axarr[row, col].annotate(xy=(x_pos[i]-0.3, percentage_list[i]+1.0), s=percentage_list[i]);
            
            axarr[0, 0].set_ylabel("Percentage of the total in this poverty group");
            axarr[1, 0].set_ylabel("Percentage of the total in this poverty group");
            axarr[1, 0].set_xlabel("Types");
            axarr[1, 1].set_xlabel("Types");

            axarr[row, col].autoscale(enable=True, axis='x')
            target_idx+=1
outside_wall_material_df = df_train_heads.groupby('Target').sum()[['paredblolad', 'paredzocalo', 'paredpreb', 'pareddes', 'paredmad', 
                                  'paredzinc', 'paredfibras', 'paredother']]
outside_wall_material_df
plot_dwelling_property(outside_wall_material_df)
floor_material_df = df_train_heads.groupby('Target').sum()[['pisomoscer', 'pisocemento', 'pisoother',
                                                      'pisonatur', 'pisonotiene', 'pisomadera']]
floor_material_df
plot_dwelling_property(floor_material_df)
toilet_df = df_train_heads.groupby('Target').sum()[['sanitario1', 'sanitario2', 'sanitario3', 'sanitario5',
                                              'sanitario6']]
toilet_df
plot_dwelling_property(toilet_df)
rubbish_disposal_df = df_train_heads.groupby('Target').sum()[['elimbasu1', 'elimbasu2', 'elimbasu3',
                                                        'elimbasu4', 'elimbasu5', 'elimbasu6']]
rubbish_disposal_df
plot_dwelling_property(rubbish_disposal_df)
roof_material_df = df_train_heads.groupby('Target').sum()[['techozinc', 'techoentrepiso', 'techocane', 'techootro']]
roof_material_df
plot_dwelling_property(roof_material_df)
water_provision_df = df_train_heads.groupby('Target').sum()[['abastaguadentro', 'abastaguafuera', 'abastaguano']]
water_provision_df
plot_dwelling_property(water_provision_df)
electricity_df = df_train_heads.groupby('Target').sum()[['public', 'planpri', 'noelec', 'coopele']]
electricity_df
plot_dwelling_property(electricity_df)
cooking_energy_df = df_train_heads.groupby('Target').sum()[['energcocinar1', 'energcocinar2', 'energcocinar3',
                                                      'energcocinar4']]
cooking_energy_df
plot_dwelling_property(cooking_energy_df)
avg_household_size_df = df_train_heads.groupby('Target').mean()['hhsize']
avg_household_size_df
df_train.groupby('Target').mean().head()
urban_rural_df = df_train_heads.groupby('Target').sum()[['area1', 'area2']]
urban_rural_df['UrbanPercentage'] = urban_rural_df['area1'] * round((100/sum(urban_rural_df['area1'])), 6)
urban_rural_df['RuralPercentage'] = urban_rural_df['area2'] * round((100/sum(urban_rural_df['area2'])), 6)
urban_rural_df
region_df = df_train_heads.groupby('Target').sum()[['lugar1', 'lugar2', 'lugar3', 'lugar4', 'lugar5', 'lugar6']]
region_df
plot_dwelling_property(region_df)
region_df.T
round(((all_data.shape[0] - sum(all_data['v2a1'].value_counts())) / all_data.shape[0] ) * 100, 2)
sns.boxplot(x='Target', y='escolari', data=all_data.loc[:ntrain]);
all_data.drop(columns=['sanitario1', 'sanitario6',
                       'elimbasu4', 'elimbasu5', 'elimbasu6',
                       'techozinc', 'techoentrepiso', 'techocane', 'techootro',
                       'abastaguadentro', 'abastaguafuera', 'abastaguano',
                       'public', 'planpri', 'noelec', 'coopele'], inplace=True)
num_features = all_data._get_numeric_data().columns
num_features_length = len(num_features)

categ_features = pd.Index(list(set(all_data.columns) - set(num_features)))
categ_features_length = len(categ_features)

print("Number of numerical features: ", num_features_length)
print("Number of categorical features: ", categ_features_length)

labels = ['numeric', 'categorical']
colors = ['y', 'r']
plt.figure(figsize=(8, 8))
plt.pie([num_features_length, categ_features_length], 
        labels=labels, 
        autopct='%1.1f%%', 
        shadow=True, 
        colors=colors);
all_data[categ_features].head()
_, axarr = plt.subplots(nrows=1, ncols=3, sharey='row', figsize=(12, 6))

for idx, feature in enumerate(['dependency', 'edjefe', 'edjefa']):
    sns.countplot(x=feature, data=all_data[all_data[feature].isin(['yes', 'no'])], ax=axarr[idx])
yes_no_map = {'no': 0, 'yes': 1}
    
all_data['dependency'] = all_data['dependency'].replace(yes_no_map).astype(np.float32)
all_data['edjefe'] = all_data['edjefe'].replace(yes_no_map).astype(np.float32)
all_data['edjefa'] = all_data['edjefa'].replace(yes_no_map).astype(np.float32)
num_binary_features = []

for feature in all_data.columns:
    if sorted(df_train[feature].unique()) in [[0, 1], [0], [1]]:
        num_binary_features.append(feature)
        
print("Total number of binary-numerical features: ", len(num_binary_features))
print("Binary-numerical features: ")
num_binary_features
num_non_binary_features = [feature for feature in all_data.columns if feature not in num_binary_features]

print("Total number of non-binary-numerical features: ", len(num_non_binary_features))
print("Non-binary numerical features: ")

num_non_binary_features_dict = {feature: len(all_data[feature].unique()) for feature in num_non_binary_features}

num_non_binary_features_sorted = sorted(num_non_binary_features_dict, 
                                        key=lambda feature: num_non_binary_features_dict[feature], 
                                        reverse=True)

num_non_binary_features_len_sorted = [num_non_binary_features_dict[feature] for feature in num_non_binary_features_sorted]

plt.figure(figsize=(16, 16))
barplot_with_anotate(num_non_binary_features_sorted, num_non_binary_features_len_sorted);
plt.ylabel("No. of unique values");
plt.xlabel("Non-binary numerical features");
all_data[num_binary_features].describe()
num_conti_features = pd.Index(['v2a1', 'meaneduc', 'dependency', 'SQBmeaned', 'SQBdependency'])
all_data[num_conti_features].describe()
num_discrete_features = pd.Index([feature for feature in num_non_binary_features if feature not in num_conti_features])
all_data[num_discrete_features].describe()
def missing_features(data, column_set):
    incomplete_features = {feature: data.shape[0]-sum(data[feature].value_counts())
                                   for feature in column_set
                                   if not sum(data[feature].value_counts()) == data.shape[0]}
    incomplete_features_sorted = sorted(incomplete_features, key=lambda feature: incomplete_features[feature], reverse=True)
    incompleteness = [round((incomplete_features[feature]/data.shape[0])*100, 2) for feature in incomplete_features_sorted]
    plt.figure(figsize=(12, 6))
    barplot_with_anotate(incomplete_features_sorted, incompleteness)
    plt.ylabel("Percentage (%) of values that are missing")
    #plt.rcParams["figure.figsize"] = [12, 6]
    
    for feature, percentage in zip(incomplete_features_sorted, incompleteness):
        print("Feature:", feature)
        print("No. of NaNs:", incomplete_features[feature], "(", percentage, ")")
missing_features(all_data, all_data.columns)
# entries which have both v2a1 as NaN and tipovivi3 as 0
all_data[['v2a1', 'tipovivi3']][all_data['tipovivi3'] == 0][all_data['v2a1'].isnull()].shape
# handling v2a1
all_data.loc[:, 'v2a1'].fillna(0, inplace=True)
# entries which have v18q as 0 and v18q1 as NaN
all_data[['v18q1', 'v18q']][all_data['v18q'] == 0][all_data['v18q1'].isnull()].shape
# handling v18q1
all_data.loc[:, 'v18q1'].fillna(0, inplace=True)
# handling meaneduc and SQBmeaned
all_data.loc[:, 'meaneduc'].fillna(all_data['meaneduc'].mean(), inplace=True)
all_data.loc[:, 'SQBmeaned'].fillna(all_data['SQBmeaned'].mean(), inplace=True)
all_data.drop(columns=['rez_esc'], inplace=True)
all_data['WallQual'] = all_data['epared1'] + 2*all_data['epared2'] + 3*all_data['epared3']

all_data['RoofQual'] = all_data['etecho1'] + 2*all_data['etecho2'] + 3*all_data['etecho3']

all_data['FloorQual'] = all_data['eviv1'] + 2*all_data['eviv2'] + 3*all_data['eviv3']

all_data['EducationLevel'] = all_data['instlevel1'] + 2*all_data['instlevel2'] + 3*all_data['instlevel3'] + \
    4*all_data['instlevel4'] + 5*all_data['instlevel5'] + 6*all_data['instlevel6'] + 7*all_data['instlevel7'] + \
    8*all_data['instlevel8'] + 9*all_data['instlevel9']
all_data.drop(columns=['epared1', 'epared2', 'epared3',
                       'etecho1', 'etecho2', 'etecho3',
                       'eviv1', 'eviv2', 'eviv3',
                       'instlevel1', 'instlevel2', 'instlevel3', 'instlevel4', 'instlevel5',
                       'instlevel6', 'instlevel7', 'instlevel8', 'instlevel9'], inplace=True)
redundant_features = ['r4t1', 'r4t2', 'r4t3', 'tamhog', 'tamviv', 'hhsize', 'r4t3', 'v18q', 'mobilephone']
all_data.drop(columns=redundant_features, inplace=True)
all_data['RentPerRoom'] = all_data['v2a1'] / all_data['rooms']

all_data['AdultsPerRoom'] = all_data['hogar_adul'] / all_data['rooms']

all_data['AdultsPerBedroom'] = all_data['hogar_adul'] / all_data['bedrooms']
# individual level boolean features
ind_bool = ['dis', 'male', 'female', 'estadocivil1', 'estadocivil2', 'estadocivil3', 
            'estadocivil4', 'estadocivil5', 'estadocivil6', 'estadocivil7', 
            'parentesco1', 'parentesco2',  'parentesco3', 'parentesco4', 'parentesco5', 
            'parentesco6', 'parentesco7', 'parentesco8',  'parentesco9', 'parentesco10', 
            'parentesco11', 'parentesco12', 'EducationLevel']

# individual level ordered features
ind_ordered = ['escolari', 'age']
f = lambda x: x.std(ddof=0)
f.__name__ = 'std_0'
ind_agg = all_data.groupby('idhogar')[ind_ordered + ind_bool].agg(['mean', 'max', 'min', 'sum', f])

new_cols = []
for col in ind_agg.columns.levels[0]:
    for stat in ind_agg.columns.levels[1]:
        new_cols.append(f'{col}-{stat}')

ind_agg.columns = new_cols
ind_agg.head()
print("Original number of features:", all_data.shape[1])

all_data = all_data.merge(ind_agg, on = 'idhogar', how = 'left')

print("Number of features after merging transformed individual level features", all_data.shape[1])

all_data.drop(columns=ind_bool+ind_ordered, inplace=True)

print("Number of features after dropping the individual level features", all_data.shape[1])
from sklearn.metrics import f1_score, make_scorer
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
# drop the idhogar column
all_data.drop(columns=['idhogar'], inplace=True)
df_train = all_data[:ntrain][:]
df_test = all_data[ntrain:][:]
df_test = df_test.drop('Target', axis=1)
print(df_train.shape)
print(df_test.shape)
X_train= df_train.drop('Target', axis= 1)
Y_train= df_train['Target']

X_test= df_test
validation_scores = {}
scorer = make_scorer(f1_score, greater_is_better=True, average = 'macro')
skf = StratifiedKFold(n_splits=5)
lightgbm = lgb.LGBMClassifier(class_weight='balanced', boosting_type='dart',
                         drop_rate=0.9, min_data_in_leaf=100, 
                         max_bin=255,
                         n_estimators=500,
                         bagging_fraction=0.01,
                         min_sum_hessian_in_leaf=1,
                         importance_type='gain',
                         learning_rate=0.1, 
                         max_depth=-1, 
                         num_leaves=31)

#validation_scores['LightGBM'] = cross_val_score(lightgbm, X_train, Y_train, cv=3, scoring=scorer).mean()
#print(validation_scores['LightGBM'])
predicts_lgb = []
for train_index, test_index in skf.split(X_train, Y_train):
    X_t, X_v = X_train.iloc[train_index], X_train.iloc[test_index]
    y_t, y_v = Y_train.iloc[train_index], Y_train.iloc[test_index]
    
    lightgbm.fit(X_t, y_t, eval_set=[(X_v, y_v)], early_stopping_rounds=50)
    predicts_lgb.append(lightgbm.predict(X_test))
lightgbm_pred = np.array(predicts_lgb).mean(axis=0).round().astype(int)

submission_lgb = pd.DataFrame({'Id': test_id,
                           'Target': lightgbm_pred})
submission_lgb.to_csv('submissionLGB.csv', index=False)
xgboost = xgb.XGBClassifier()

#validation_scores['XGBoost'] = cross_val_score(xgboost, X_train, Y_train, cv=3, scoring=scorer).mean()
#print(validation_scores['XGBoost']);
predicts_xgb = []
for train_index, test_index in skf.split(X_train, Y_train):
    X_t, X_v = X_train.iloc[train_index], X_train.iloc[test_index]
    y_t, y_v = Y_train.iloc[train_index], Y_train.iloc[test_index]
    
    xgboost.fit(X_t, y_t, eval_set=[(X_v, y_v)], early_stopping_rounds=50)
    predicts_xgb.append(xgboost.predict(X_test))
xgboost_pred = np.array(predicts_xgb).mean(axis=0).round().astype(int)

submission_xgb = pd.DataFrame({'Id': test_id,
                           'Target': xgboost_pred})
submission_xgb.to_csv('submissionXGB.csv', index=False)
'''models_with_scores = pd.DataFrame({
    'Model': list(validation_scores.keys()),
    'Validation Score': list(validation_scores.values())})

models_with_scores.sort_values(by='Validation Score', ascending=False)'''
submission_model_lgb_old = lgb.LGBMClassifier(class_weight='balanced', boosting_type='dart',
                         drop_rate=0.9, min_data_in_leaf=100, 
                         max_bin=255,
                         n_estimators=500,
                         bagging_fraction=0.01,
                         min_sum_hessian_in_leaf=1,
                         importance_type='gain',
                         learning_rate=0.1, 
                         max_depth=-1, 
                         num_leaves=31)
submission_model_lgb_old.fit(X_train, Y_train);
final_pred_lgb_old = submission_model_lgb_old.predict(X_test)
final_pred_lgb_old = final_pred_lgb_old.astype(int)
submission_lgb_old = pd.DataFrame({'Id': test_id,
                           'Target': final_pred_lgb_old})
submission_lgb_old.to_csv('submissionLGBold.csv', index=False)
'''submission_model_xgboost = lgb.LGBMClassifier()
submission_model_xgboost.fit(X_train, Y_train);
final_pred_xgb = submission_model_xgboost.predict(X_test)
final_pred_xgb = final_pred_xgb.astype(int)'''
'''submission_xgb = pd.DataFrame({'Id': test_id,
                           'Target': final_pred_xgb})
submission_xgb.to_csv('submissionXGB.csv', index=False)'''
'''final_pred_stacked = ((final_pred_lgb + final_pred_xgb) / 2).astype(int)
submission_stacked = pd.DataFrame({'Id': test_id,
                           'Target': final_pred_stacked})
submission_stacked.to_csv('submissionStacked.csv', index=False)'''