"""

Created on Thu Feb  6 19:15:03 2020



@author: Radu

"""





import numpy as np

import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from category_encoders import TargetEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from datetime import datetime



import warnings; warnings.simplefilter(action='ignore', category=FutureWarning)



def read_data():

    print(f'Reading data')

    path = r"C:\Users\Radu\Desktop\ML Projects\Categorical Feature Encoding Challenge II/"

    train_df = pd.read_csv(f'{path}train.csv')

    test_df = pd.read_csv(f'{path}test.csv')

    sample_submission_df = pd.read_csv(f'{path}sample_submission.csv')



    return train_df, test_df, sample_submission_df



train_df, test_df, sample_submission_df = read_data()



y_train = train_df['target'].copy()

X_train = train_df.drop(['target', 'id'], axis=1)

X_test = test_df.copy()



############################################# Missing Variable Imputation ###############################################



def replace_nan(data):

    for column in data.columns:

        print(column)

        if data[column].isna().sum() > 0:

            data[column] = data[column].fillna(data[column].mode()[0])





replace_nan(X_train)

replace_nan(X_test)



###################################################### Varaible encoding ###############################################



see = X_train.head(1000)



X_train.loc[X_train['bin_3'] ==  'F', 'bin_3'] = 0

X_train.loc[X_train['bin_3'] ==  'T', 'bin_3'] = 1



X_train.loc[X_train['bin_4'] ==  'N', 'bin_4'] = 0

X_train.loc[X_train['bin_4'] ==  'Y', 'bin_4'] = 1



X_train.loc[X_train['day'] ==  1, 'day'] = 'Monday'

X_train.loc[X_train['day'] ==  2, 'day'] = 'Thusday'

X_train.loc[X_train['day'] ==  3, 'day'] = 'wednesday'

X_train.loc[X_train['day'] ==  4, 'day'] = 'Thursday'

X_train.loc[X_train['day'] ==  5, 'day'] = 'Friday'

X_train.loc[X_train['day'] ==  6, 'day'] = 'Saturday'

X_train.loc[X_train['day'] ==  7, 'day'] = 'Sunday'



X_train.loc[X_train['month'] ==  1, 'month'] = 'Jan'

X_train.loc[X_train['month'] ==  2, 'month'] = 'Feb'

X_train.loc[X_train['month'] ==  3, 'month'] = 'Mar'

X_train.loc[X_train['month'] ==  4, 'month'] = 'Apr'

X_train.loc[X_train['month'] ==  5, 'month'] = 'May'

X_train.loc[X_train['month'] ==  6, 'month'] = 'Jun'

X_train.loc[X_train['month'] ==  7, 'month'] = 'Jul'

X_train.loc[X_train['month'] ==  8, 'month'] = 'Aug'

X_train.loc[X_train['month'] ==  9, 'month'] = 'Sep'

X_train.loc[X_train['month'] ==  10, 'month'] = 'Oct'

X_train.loc[X_train['month'] ==  11, 'month'] = 'Nov'

X_train.loc[X_train['month'] ==  12, 'month'] = 'Dec'



X_test.loc[X_test['bin_3'] ==  'F', 'bin_3'] = 0

X_test.loc[X_test['bin_3'] ==  'T', 'bin_3'] = 1



X_test.loc[X_test['bin_4'] ==  'N', 'bin_4'] = 0

X_test.loc[X_test['bin_4'] ==  'Y', 'bin_4'] = 1



X_test.loc[X_test['day'] ==  1, 'day'] = 'Monday'

X_test.loc[X_test['day'] ==  2, 'day'] = 'Thusday'

X_test.loc[X_test['day'] ==  3, 'day'] = 'wednesday'

X_test.loc[X_test['day'] ==  4, 'day'] = 'Thursday'

X_test.loc[X_test['day'] ==  5, 'day'] = 'Friday'

X_test.loc[X_test['day'] ==  6, 'day'] = 'Saturday'

X_test.loc[X_test['day'] ==  7, 'day'] = 'Sunday'



X_test.loc[X_test['month'] ==  1, 'month'] = 'Jan'

X_test.loc[X_test['month'] ==  2, 'month'] = 'Feb'

X_test.loc[X_test['month'] ==  3, 'month'] = 'Mar'

X_test.loc[X_test['month'] ==  4, 'month'] = 'Apr'

X_test.loc[X_test['month'] ==  5, 'month'] = 'May'

X_test.loc[X_test['month'] ==  6, 'month'] = 'Jun'

X_test.loc[X_test['month'] ==  7, 'month'] = 'Jul'

X_test.loc[X_test['month'] ==  8, 'month'] = 'Aug'

X_test.loc[X_test['month'] ==  9, 'month'] = 'Sep'

X_test.loc[X_test['month'] ==  10, 'month'] = 'Oct'

X_test.loc[X_test['month'] ==  11, 'month'] = 'Nov'

X_test.loc[X_test['month'] ==  12, 'month'] = 'Dec'





bin_features = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4']



ohe_features = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'day', 'month']



target_features = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_0', 'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']







# Label encoding columns 

col_list = []

from sklearn.preprocessing import LabelEncoder

label_x_train = X_train[ohe_features]

label_x_test = X_test[ohe_features]

for j in label_x_train.columns.values:

    print(j)

    le = LabelEncoder()

    ### fit with the desired col, col in position 0 for this ###example

    fit_by = pd.Series([i for i in label_x_train[j].unique() if type(i) == str])

    le.fit(fit_by)

    ### Set transformed col leaving np.NaN as they are

    label_x_train[j] = label_x_train[j].apply(lambda x: le.transform([x])[0] if type(x) == str else x)

    label_x_test[j] = label_x_test[j].apply(lambda x: le.transform([x])[0] if type(x) == str else x)

    col_list.extend([ j + "_" + s  for s in le.classes_])

    



# Variable OH encoding 

ohe = OneHotEncoder( handle_unknown="ignore", sparse = False )

ohe_x_train = ohe.fit_transform(label_x_train)

ohe_x_test = ohe.transform(label_x_test)

#ohe_x_test = ohe.transform(X_test[ohe_features])





# Target encoding columns 'nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'

def transform(transformer, x_train, y_train, cv):

    oof = pd.DataFrame(index=x_train.index, columns=x_train.columns)

    for train_idx, valid_idx in cv.split(x_train, y_train):

        x_train_train = x_train.loc[train_idx]

        y_train_train = y_train.loc[train_idx]

        x_train_valid = x_train.loc[valid_idx]

        transformer.fit(x_train_train, y_train_train)

        oof_part = transformer.transform(x_train_valid)

        oof.loc[valid_idx] = oof_part

    return oof



target = TargetEncoder(drop_invariant=True, smoothing=0.2)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



target_x_train = transform(target, X_train[target_features], y_train, cv).astype('float')



target.fit(X_train[target_features], y_train)

target_x_test = target.transform(X_test[target_features]).astype('float')



# Putting it all toghether

X_train = np.hstack([X_train[bin_features], ohe_x_train, target_x_train])

X_test = np.hstack([X_test[bin_features], ohe_x_test, target_x_test])



see = pd.DataFrame(X_train[:1000,:])



##################################################### Rule Fit Model ##################################################



from sklearn.ensemble import GradientBoostingClassifier

from rulefit import RuleFit



mia_label_x_train = X_train[:200000,:]

mia_y_train = y_train[:200000]



feature_names = []

feature_names.extend(bin_features)

feature_names.extend(col_list)

feature_names.extend(target_features)





print("Training started at {}".format(datetime.now()))

clf = GradientBoostingClassifier(n_estimators=10, learning_rate=0.1, max_depth=3, random_state=0)

rf = RuleFit(tree_generator= clf, rfmode= 'classify')

rf.fit(mia_label_x_train, mia_y_train, feature_names=feature_names)

print("Training ended at {}".format(datetime.now()))



rules = rf.get_rules()



#rules.to_csv('rules_RuleFit.csv')



# save the model to disk

import pickle

filename = 'RuleFit_model.sav'

#pickle.dump(rf, open(filename, 'wb'))

 

# load the model from disk

loaded_model = pickle.load(open(filename, 'rb'))



mia_label_x_test = X_train[200000:,:]

mia_y_test = y_train[200000:]



y_pred = rf.predict(mia_label_x_test)

print( 'Accuracy score is: {}'.format(roc_auc_score(mia_y_test, y_pred)))



##################################################### Recreate Rules ##################################################



X_train_rule = pd.DataFrame(X_train, columns = feature_names )



see1= X_train_rule.head(1000)



# nom_8 <= 0.2058642879128456 & ord_3 <= 0.2574392706155777

X_train_rule['rule1'] = np.where((X_train_rule['nom_8']<= 0.2058642879128456) & (X_train_rule['ord_3'] <= 0.2574392706155777), 1, 0)

# nom_8 <= 0.15247249603271484

X_train_rule['rule2'] = np.where(X_train_rule['nom_8']<= 0.15247249603271484, 1, 0)

# nom_8 > 0.2058642879128456 & ord_3 <= 0.2574392706155777

X_train_rule['rule3'] = np.where((X_train_rule['nom_8']> 0.2058642879128456) & (X_train_rule['ord_3'] <= 0.2574392706155777), 1, 0)

# day_Saturday > 0.5 & ord_1 <= 0.18961423635482788 & day_Monday <= 0.5

X_train_rule['rule4'] = np.where((X_train_rule['day_Saturday'] == 1) & (X_train_rule['ord_1'] <= 0.18961423635482788), 1, 0)

# ord_5 > 0.11862009018659592 & ord_0 > 2.5 & nom_5 > 0.17176222056150436

X_train_rule['rule5'] = np.where((X_train_rule['ord_5'] > 0.11862009018659592) & (X_train_rule['ord_0'] > 2.5) & (X_train_rule['nom_5'] > 0.17176222056150436), 1, 0)

# ord_3 > 0.13002829253673553 & nom_8 > 0.15247249603271484 & bin_0 <= 0.5

X_train_rule['rule6'] = np.where((X_train_rule['ord_3'] > 0.13002829253673553) & (X_train_rule['nom_8'] > 0.15247249603271484) & (X_train_rule['bin_0']  <= 0.5), 1, 0)

# nom_9 <= 0.17295265197753906

X_train_rule['rule7'] = np.where(X_train_rule['nom_9']<= 0.17295265197753906, 1, 0)

# month_Sep <= 0.5 & nom_9 <= 0.17295265197753906

X_train_rule['rule8'] = np.where((X_train_rule['month_Sep'] < 1) & (X_train_rule['nom_9'] <= 0.17295265197753906), 1, 0)

# nom_9 > 0.13000915944576263 & month_Jul > 0.5

X_train_rule['rule9'] = np.where((X_train_rule['nom_9'] > 0.13000915944576263) & (X_train_rule['month_Jul'] == 1 ), 1, 0)

# nom_8 > 0.2521960437297821 & nom_6 > 0.24674799293279648 & ord_4 > 0.2339029535651207

X_train_rule['rule10'] = np.where((X_train_rule['nom_8'] > 0.2521960437297821) & (X_train_rule['nom_6'] > 0.24674799293279648) & (X_train_rule['ord_4']  > 0.2339029535651207), 1, 0)

# month_Apr > 0.5 & ord_1 > 0.15924452245235443 & nom_8 <= 0.2626989334821701

X_train_rule['rule11'] = np.where((X_train_rule['month_Apr'] > 0.5) & (X_train_rule['ord_1'] > 0.15924452245235443) & (X_train_rule['nom_8']  <= 0.2626989334821701), 1, 0)

# nom_5 > 0.2712095379829407 & nom_7 > 0.2467493638396263 & bin_1 > 0.5

X_train_rule['rule12'] = np.where((X_train_rule['nom_5'] > 0.2712095379829407) & (X_train_rule['nom_7'] > 0.2467493638396263 ) & (X_train_rule['bin_1']  == 1), 1, 0)





print("Training started at {}".format(datetime.now()))

oof = pd.DataFrame(index=train_df.index, columns= ['y_prob'])

cv = StratifiedKFold(n_splits=5, random_state = 62)

for train_idx, valid_idx in cv.split(X_train_rule, y_train):

    x_train_train = X_train_rule.iloc[train_idx, :]

    y_train_train = y_train[train_idx]

    x_train_valid = X_train_rule.iloc[valid_idx, :]

    y_train_valid = y_train[valid_idx]

        

    # Fitting Logistic Regression to the Training set

    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression(solver='sag', random_state = 0)

    classifier.fit(x_train_train, y_train_train)

    

    # Predicting the Test set results

    #y_pred = classifier.predict(X_train)

    y_prob_train=classifier.predict_proba(x_train_train)   

    y_prob_valid=classifier.predict_proba(x_train_valid)         

    oof.loc[valid_idx] = y_prob_valid[:,1:]

    

    # AUROC and AR

    ROC_Train = roc_auc_score(y_train_train, y_prob_train[:,1])

    ROC_Valid = roc_auc_score(y_train_valid, y_prob_valid[:,1])



    print("AUROC Train Set : {:2.2f} %".format(ROC_Train*100))

    print("AUROC Valid Set : {:2.2f} %".format(ROC_Valid*100))

#y_prob_test=classifier.predict_proba(X_test) 

print("Training ended at {}".format(datetime.now()))



coef = pd.merge(pd.DataFrame(X_train_rule.columns), pd.DataFrame(np.transpose(classifier.coef_)), left_index = True, right_index = True)
