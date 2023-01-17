import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

from sklearn.metrics import roc_auc_score, roc_curve, auc



from bayes_opt import BayesianOptimization



from IPython.display import display



# Suppr warning

import warnings

warnings.filterwarnings("ignore")



%matplotlib inline  
plt.style.use('ggplot')



pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 1000)

pd.options.mode.use_inf_as_na = True
#Load data

data_path = '../input/learn-together/'



train_df = pd.read_csv(data_path + 'train.csv', index_col = 'Id')

test_df = pd.read_csv(data_path + 'test.csv', index_col = 'Id')
features = [f for f in list(train_df) if f!='Cover_Type']

target = 'Cover_Type'
stony_level = np.array([3,2,0,0,0,1,0,0,2,0,0,1,0,0,0,0,0,2,0,0,0,0,0,3,3,2,3,3,3,3,3,3,3,3,0,3,3,3,3,3])
def feat_eng(df):

    

    df['Dist_To_Hydro'] = np.sqrt(df['Horizontal_Distance_To_Hydrology']**2 + df['Vertical_Distance_To_Hydrology']**2)

    df['Log_Dist_To_Hydro'] = np.log( np.sqrt(df['Horizontal_Distance_To_Hydrology']**2 + df['Vertical_Distance_To_Hydrology']**2) +1)

    

    df['Hydro_Fire_p'] = np.abs(df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Fire_Points'])

    df['Hydro_Fire_n'] = np.abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Fire_Points'])



    df['Hydro_Road_p'] = np.abs(df['Horizontal_Distance_To_Hydrology'] + df['Horizontal_Distance_To_Roadways'])

    df['Hydro_Road_n'] = np.abs(df['Horizontal_Distance_To_Hydrology'] - df['Horizontal_Distance_To_Roadways'])



    df['Fire_Road_p'] = np.abs(df['Horizontal_Distance_To_Fire_Points'] + df['Horizontal_Distance_To_Roadways'])

    df['Fire_Road_n'] = np.abs(df['Horizontal_Distance_To_Fire_Points'] - df['Horizontal_Distance_To_Roadways'])

    

    horiz_grp = [f for f in features if f.startswith('Horizontal')]

    df['Horiz_Dist_Mean'] = df[horiz_grp].mean(axis = 1).round(2)

    df['Horiz_Dist_Std'] = df[horiz_grp].std(axis = 1).round(2)

    

    df['Is_Overwater'] = df['Vertical_Distance_To_Hydrology'] > 0

    

    hill_grp = [f for f in features if f.startswith('Hill')]

    df['Hillshade_Mean'] = df[hill_grp].mean(axis = 1).round(2)

    df['Hillshade_Std'] = df[hill_grp].std(axis = 1).round(2)

    

    soil_grp = [f for f in features if f.startswith('Soil_Type')]

    df['Stony_Level'] =  df[soil_grp]@stony_level

    

    df['Elevation_Adjusted'] = df['Elevation'] - df['Vertical_Distance_To_Hydrology']

    df['Elevation'] = np.log(df['Elevation'])

    

    df['Aspect'] = df['Aspect'].astype(int) % 360

    df['Sen_Aspect'] = np.sin(np.radians(df['Aspect']))

    df['Cos_Aspect'] = np.cos(np.radians(df['Aspect']))

    

    df['Sen_Slope'] = np.sin(np.radians(df['Slope']))

    df['Cos_Slope'] = np.cos(np.radians(df['Slope']))

    return df

    

train_df = feat_eng(train_df)

test_df = feat_eng(test_df)
features = [f for f in list(train_df) if f!='Cover_Type']

target = 'Cover_Type'
X_train, X_val, Y_train, Y_val = train_test_split(train_df[features], train_df[target],

                                                  test_size = 0.2, 

                                                  random_state = 42, 

                                                  stratify = train_df[target])

train_idx = X_train.index

val_idx = X_val.index
def Bayes_ExtraTrees(n_estimators,

                     max_depth,

                     min_samples_split,

                     min_samples_leaf,

                     max_features,

                     bootstrap):

    

    n_estimators = int(n_estimators)

    max_depth = int(max_depth)

    min_samples_split = int(min_samples_split)

    min_samples_leaf = int(min_samples_leaf)

    bootstrap = bootstrap > 0.5

    

    assert type(n_estimators) == int

    assert type(max_depth) == int

    assert type(min_samples_split) == int

    assert type(min_samples_leaf) == int

    

    preds = np.zeros(len(X_val))



    etc = ExtraTreesClassifier(n_estimators=n_estimators,

                               max_depth=max_depth,

                               min_samples_split=min_samples_split,

                               min_samples_leaf=min_samples_leaf,

                               max_features=max_features,

                               bootstrap=bootstrap,

                               oob_score=bootstrap,

                               n_jobs=6,

                               random_state=42,

                               verbose=0)

    etc.fit(X_train, Y_train)

    

    preds = etc.predict(X_val)

    

    score = accuracy_score(Y_val, preds)

    

    return score

params = {'n_estimators' : (40, 1000),

          'max_depth' : (10, 200),

          'min_samples_split': (2,15),

          'min_samples_leaf' : (2,15),

          'max_features' : (.2,.8),

          'bootstrap':(0,1)} 



ExtraTreesBO = BayesianOptimization(Bayes_ExtraTrees, params, random_state = 42)



print(ExtraTreesBO.space.keys)

init_points = 15

n_iter = 20
print('--' * 100)



with warnings.catch_warnings():

    warnings.filterwarnings('ignore')

    ExtraTreesBO.maximize(init_points = init_points,

                            n_iter = n_iter,

                            acq = 'ucb',

                            xi = 0.0,

                            alpha = 1e-6)

print(f'Best Accuracy Achieved: {ExtraTreesBO.max["target"]}')

X_train, X_val0, Y_train, Y_val0 = train_test_split(train_df[features], train_df[target],

                                                  test_size = 0.05, 

                                                  random_state = 111, 

                                                  stratify = train_df[target])
nfold = 11

skf = StratifiedKFold(n_splits = nfold, shuffle = True, random_state = 42)



oof = np.zeros(len(train_df))

oof_probs = np.zeros((len(train_df),7))



accuracies = []



y_prob_et = np.zeros((len(test_df),7))

y_val_prob = np.zeros((len(X_val0),7))



feature_importance_et = pd.DataFrame()
for fold, (trn_idx, val_idx) in enumerate(skf.split(X_train, Y_train)):

    print(f'Computing fold {fold+1}...')

    

    etc = ExtraTreesClassifier(n_estimators = int(ExtraTreesBO.max['params']['n_estimators']),

                               max_depth = int(ExtraTreesBO.max['params']['max_depth']),

                               min_samples_split = int(ExtraTreesBO.max['params']['min_samples_split']),  

                               min_samples_leaf = int(ExtraTreesBO.max['params']['min_samples_leaf']),

                               max_features =  ExtraTreesBO.max['params']['max_features'],                 

                               bootstrap = ExtraTreesBO.max['params']['bootstrap'] > 0.5,

                               oob_score = ExtraTreesBO.max['params']['bootstrap'] > 0.5,

                               n_jobs=6)

   

 

    X_trn, X_val = X_train.iloc[trn_idx, :], X_train.iloc[val_idx, :]

    Y_trn, Y_val = Y_train.iloc[trn_idx], Y_train.iloc[val_idx]

    etc.fit(X_trn, Y_trn)

    

    oof[val_idx] = etc.predict(X_val)

    oof_probs[val_idx] = etc.predict_proba(X_val)

    

    accuracies.append(accuracy_score(Y_val, oof[val_idx]))

    

    print(f'Accuracy on Validation: {round(accuracies[fold],4)}')

    if hasattr(etc, 'oob_score_'):

        print(f'oob score: {round(etc.oob_score_,4)}')

    

    y_prob_et += etc.predict_proba(test_df[features])/ skf.n_splits 

    y_val_prob += etc.predict_proba(X_val0)/ skf.n_splits 

    

    # Features imp

    fold_importance_df = pd.DataFrame({'feature': features, 'importance': etc.feature_importances_, 'fold': nfold+1})

    feature_importance_et = pd.concat([feature_importance_et, fold_importance_df], axis=0)

    

    del X_trn, X_val, Y_trn, Y_val



print(f"Mean accuracy score: {round(np.mean(accuracies),4)}")
y_pred = np.argmax(y_prob_et, axis = 1)+1

y_val_pred = np.argmax(y_val_prob, axis = 1)+1
print(pd.crosstab(Y_val0, y_val_pred, rownames=['Actual'], colnames=['Prediction']))

print(f"accuracy: {accuracy_score(Y_val0, y_val_pred)}")

print(f"\n {classification_report(Y_val0, y_val_pred)}")
sns.set_style('whitegrid')



cols = (feature_importance_et[["feature", "importance"]]

    .groupby("feature")

    .mean()

    .sort_values(by="importance", ascending=False)[:30].index)



best_features = feature_importance_et.loc[feature_importance_et['feature'].isin(cols)]



plt.figure(figsize=(10,10))

sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance",ascending=False),

        edgecolor=('black'), linewidth=2, palette="colorblind")

plt.title('ExtraTreesClassifier Features Importance (averaged/folds)', fontsize=15)

plt.tight_layout()
test_df.reset_index(inplace=True)

idx = test_df.Id



preds = pd.DataFrame({'Id': idx,

              'Cover_Type': y_pred})

preds.to_csv('submission.csv', index = False)

preds.head(10)