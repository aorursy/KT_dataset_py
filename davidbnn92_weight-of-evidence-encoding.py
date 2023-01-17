import numpy as np 

import pandas as pd 

import os



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve



import category_encoders as ce



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



test_features = pd.read_csv("../input/cat-in-the-dat-ii/test.csv")

train_set = pd.read_csv("../input/cat-in-the-dat-ii/train.csv")



train_targets = train_set.target

train_features = train_set.drop(['target'], axis=1)

percentage = train_targets.mean() * 100

print("The percentage of ones in the training target is {:.2f}%".format(percentage))

train_features.head()
columns = [col for col in train_features.columns if col != 'id']

woe_encoder = ce.WOEEncoder(cols=columns)

woe_encoded_train = woe_encoder.fit_transform(train_features[columns], train_targets).add_suffix('_woe')

train_features = train_features.join(woe_encoded_train)



woe_encoded_cols = woe_encoded_train.columns
df = train_features.copy()

df['target'] = train_targets



overall_number_of_ones = train_targets.sum()

overall_number_of_zeroes = 600000 - overall_number_of_ones

print("There are {} ones and {} zeroes in the training set".format(

    overall_number_of_ones, overall_number_of_zeroes

))



grouped = pd.DataFrame()

grouped['Total'] = df.groupby('nom_0').id.count()

grouped['number of ones'] = df.groupby('nom_0').target.sum()

grouped['number of zeroes'] = grouped['Total'] - grouped['number of ones']



grouped['percentage of ones'] = grouped['number of ones'] / overall_number_of_ones

grouped['percentage of zeroes'] = grouped['number of zeroes'] / overall_number_of_zeroes

grouped['(% ones) > (% zeroes)'] = grouped['percentage of ones'] > grouped['percentage of zeroes']



grouped['weight of evidence'] = df.groupby('nom_0').nom_0_woe.mean()



grouped
grouped = pd.DataFrame()

grouped['Total'] = df.groupby('month').id.count()

grouped['number of ones'] = df.groupby('month').target.sum()

grouped['number of zeroes'] = grouped['Total'] - grouped['number of ones']



grouped['percentage of ones'] = grouped['number of ones'] / overall_number_of_ones

grouped['percentage of zeroes'] = grouped['number of zeroes'] / overall_number_of_zeroes

grouped['(% ones) > (% zeroes)'] = grouped['percentage of ones'] > grouped['percentage of zeroes']



grouped['weight of evidence'] = df.groupby('month').month_woe.mean()



grouped
# Define helper function

def logreg_test(cols, encoder):

    df = train_features[cols]

    auc_scores = []

    acc_scores = []

    

    skf = StratifiedKFold(n_splits=6, shuffle=True).split(df, train_targets)

    for train_id, valid_id in skf:

        enc_tr = encoder.fit_transform(df.iloc[train_id,:], train_targets.iloc[train_id])

        enc_val = encoder.transform(df.iloc[valid_id,:])

        regressor = LogisticRegression(solver='lbfgs', max_iter=1000, C=0.6)

        regressor.fit(enc_tr, train_targets.iloc[train_id])

        acc_scores.append(regressor.score(enc_val, train_targets.iloc[valid_id]))

        probabilities = [pair[1] for pair in regressor.predict_proba(enc_val)]

        auc_scores.append(roc_auc_score(train_targets.iloc[valid_id], probabilities))

        

    acc_scores = pd.Series(acc_scores)

    mean_acc = acc_scores.mean() * 100

    print("Mean accuracy score: {:.3f}%".format(mean_acc))

    

    auc_scores = pd.Series(auc_scores)

    mean_auc = auc_scores.mean() * 100

    print("Mean AUC score: {:.3f}%".format(mean_auc))



##########################################

print("Using Weight of Evidence Encoder")

woe_encoder = ce.WOEEncoder(cols=columns)

logreg_test(columns, woe_encoder)



##########################################

print("\nUsing Target Encoder")

targ_encoder = ce.TargetEncoder(cols=columns, smoothing=0.2)

logreg_test(columns, targ_encoder)



##########################################

print("\nUsing CatBoost Encoder")

cb_encoder = ce.CatBoostEncoder(cols=columns)

logreg_test(columns, cb_encoder)
# Encode again, this time on the whole training set. WOEE was done above.

encoder = ce.TargetEncoder(cols=columns, smoothing=0.2)

encoded_train = encoder.fit_transform(train_features[columns], train_targets).add_suffix('_targ_enc')

train_features = train_features.join(encoded_train)



encoder = ce.CatBoostEncoder(cols=columns)

encoded_train = encoder.fit_transform(train_features[columns], train_targets).add_suffix('_catboost')

train_features = train_features.join(encoded_train)



training_set = train_features.copy()

training_set['target'] = train_targets

corrmat = training_set.corr()

plt.subplots(figsize=(20,20))

sns.heatmap(corrmat, vmax=0.9, square=True)
corr_with_target = corrmat['target'].apply(abs).sort_values(ascending=False)

corr_with_target.drop(['target'], inplace=True)

df = pd.DataFrame(data={'features': corr_with_target.index, 'target': corr_with_target.values})

plt.figure(figsize=(20, 20))

sns.barplot(x="target", y="features", data=df)

plt.title('Correlation with target')

plt.tight_layout()

plt.show()
# Encoding training data

df = train_features[columns]

train_encoded = pd.DataFrame()

skf = StratifiedKFold(n_splits=5,shuffle=True).split(df, train_targets)

for tr_in,fold_in in skf:

    encoder = ce.WOEEncoder(cols=columns)

    encoder.fit(df.iloc[tr_in,:], train_targets.iloc[tr_in])

    train_encoded = train_encoded.append(encoder.transform(df.iloc[fold_in,:]),ignore_index=False)



train_encoded = train_encoded.sort_index()



# Encoding test data

encoder = ce.WOEEncoder(cols=columns)

encoder.fit(df, train_targets)

test_encoded = encoder.transform(test_features[columns])



# Fitting

regressor = LogisticRegression(solver='lbfgs', max_iter=1000, C=0.6)

regressor.fit(train_encoded, train_targets)



# Predicting

probabilities = [pair[1] for pair in regressor.predict_proba(test_encoded)]

output = pd.DataFrame({'id': test_features['id'],

                       'target': probabilities})

output.to_csv('submission.csv', index=False)

output.describe()