import pandas as pd

import numpy as np
import datetime as dt

from sklearn.impute import SimpleImputer, MissingIndicator

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.sparse import csr_matrix, save_npz, vstack
raw_train_df = pd.read_csv('./data/train.csv')

raw_train_df['source'] = "train"

raw_train_df.rename(str.lower, axis='columns', inplace=True)
raw_test_df = pd.read_csv('./data/test.csv')

raw_test_df['source'] = "test"

raw_test_df.rename(str.lower, axis='columns', inplace=True)
# Remove samples from train that exists in both datasets
tz_in_both = set(raw_train_df['tz']).intersection(raw_test_df['tz'])

raw_train_df = raw_train_df[~raw_train_df['tz'].isin(tz_in_both)]
train_df = raw_train_df.drop("nesher", axis=1)
test_df = raw_test_df.copy()
raw_df = pd.concat([train_df, raw_test_df], axis=0, ignore_index=True)
df = raw_df.copy()
df.drop(["kahas_two_months_before_giyus", "year"], axis = 1, inplace = True) #column already exists in dummy format
df['mahzor_acharon'] = pd.to_datetime(df['mahzor_acharon'], format='%d/%m/%Y')
df['mahzor_month'] = df['mahzor_acharon'].dt.month
df['mahzor_march'] = np.where(df['mahzor_month']==3, 1, 0)

df['mahzor_august'] = np.where(df['mahzor_month']==8, 1, 0)

df['mahzor_november'] = np.where(df['mahzor_month']==11, 1, 0)
df.drop("mahzor_month", axis=1, inplace=True)
manila_columns = [col for col in df.columns if 'will' in col]
manila_columns.remove('destination_will')

manila_columns.remove('lohem_will')
manila = df.melt(id_vars="tz", value_vars=manila_columns, var_name="unit", value_name="will")
#treat 0 as Null

zero_mask = manila["will"] != 0

manila = manila[zero_mask].copy()
manila_stats = manila[["tz", "will"]].groupby("tz").agg(["mean", "median", "min", "max", "sum", "count", np.std])
manila_stats.columns = manila_stats.columns.droplevel()
manila = manila.merge(df[["tz", "destination_will"]], how="inner", on="tz")
# finding and removing up to one unit where the value of destination will is equal to the unit will

# this allows calculating the statistics without it, thus enabling to find patterns

# such as the difference between max() before and after the drop
indices_df = manila[manila["will"] == manila["destination_will"]].groupby('tz').idxmax()

indices_to_drop = indices_df['destination_will'].values

manila_no_dest = manila.drop(index = indices_to_drop, axis=0)
manila_stats_no_dest = manila_no_dest[["tz", "will"]].groupby("tz").agg(["mean", "median", "min", "max", "sum", np.std])
manila_stats_no_dest.columns = manila_stats_no_dest.columns.droplevel()
manila_stats_no_dest.reset_index(inplace=True)
combined_manila_stats = manila_stats.merge(manila_stats_no_dest, how="left", on="tz", suffixes=('_all', '_no_dest'))
#Create "Difference" columns

funcs = ["mean", "median", "min", "max", "sum", "std"]

for func in funcs:

    column_name = func + "_diff"

    combined_manila_stats[column_name] = combined_manila_stats[func + "_all"] - combined_manila_stats[func + "_no_dest"]
combined_manila_stats.set_index("tz", inplace=True)
#adding column prefix to assist debugging and for inspection in feature importances (next notebook)

combined_manila_stats.columns = ["manila_" + column for column in combined_manila_stats.columns]
df = df.merge(combined_manila_stats, how="left", left_on="tz", right_index=True)
# "count" states how many units have been filled by the user

# this ensures that empty manilas get 0 count
df["manila_count"].fillna(0, inplace=True)
app_raw_df = pd.read_csv('./data/applications.csv')
#filter application only to those in train/test

tz_in_df_mask = app_raw_df['CustomerID'].isin(df['tz'])

app_raw_df = app_raw_df[tz_in_df_mask].copy().reset_index(drop=True)
app_raw_df['Sub1'].fillna(value="unknown", inplace=True)

app_raw_df['Sub2'].fillna(value="unknown", inplace=True)
#concatenate Sub1 & Sub2

app_raw_df['Sub1_Sub2'] = app_raw_df[['Sub1', 'Sub2']].apply(lambda x: '_'.join(x), axis=1)
mask = app_raw_df['CustomerID'].isin(train_df['tz'])

app_train_df = app_raw_df[mask].copy()
mask = app_raw_df['CustomerID'].isin(test_df['tz'])

app_test_df = app_raw_df[mask].copy()
sub1_mask = app_raw_df['Sub1'].isin(app_train_df['Sub1'])

sub2_mask = app_raw_df['Sub2'].isin(app_train_df['Sub2'])

sub_mix_mask = app_raw_df['Sub1_Sub2'].isin(app_train_df['Sub1_Sub2'])
sub1_mask_2 = app_raw_df['Sub1'].isin(app_test_df['Sub1'])

sub2_mask_2 = app_raw_df['Sub2'].isin(app_test_df['Sub2'])

sub_mix_mask_2 = app_raw_df['Sub1_Sub2'].isin(app_test_df['Sub1_Sub2'])
clean_df = app_raw_df[sub1_mask & sub2_mask & sub_mix_mask & sub1_mask_2 & sub2_mask_2 & sub_mix_mask_2].copy()
df_sub1 = pd.crosstab(clean_df['CustomerID'],clean_df['Sub1'])
df_sub1.columns = ["sub1_" + column for column in df_sub1.columns]
df_sub2 = pd.crosstab(clean_df['CustomerID'],clean_df['Sub2'])

df_sub2.columns = ["sub2_" + column for column in df_sub2.columns]
df_submix = pd.crosstab(clean_df['CustomerID'],clean_df['Sub1_Sub2'])

df_submix.columns = ["submix_" + column for column in df_submix.columns]
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(lowercase=False, ngram_range=(1,3), binary=True, norm=None, use_idf=False, min_df = 10)
train_tfidf = app_train_df.dropna(subset=["Sub3"], axis=0)
# concatenate all strings from customer ID into one cell

train_tfidf = train_tfidf.groupby(['CustomerID'])['Sub3'].apply(lambda x: ' '.join(x)).reset_index()
tfidf = tfidf.fit(train_tfidf['Sub3'])
df_all_tfidf = app_raw_df.dropna(subset=["Sub3"], axis=0).copy()
df_all_tfidf = df_all_tfidf.groupby(['CustomerID'])['Sub3'].apply(lambda x: ' '.join(x)).reset_index()
tfidf_all = tfidf.transform(df_all_tfidf['Sub3'])
tfidf_all
df_sub3 = pd.DataFrame(tfidf_all.todense())
df_sub3.columns = tfidf.vocabulary_
df_sub3['CustomerID'] = df_all_tfidf['CustomerID']
df_sub3.set_index("CustomerID", inplace=True)
df_sub3.columns = ["sub3_" + column for column in df_sub3.columns]
app_counts_df = app_raw_df[['Mispar_Pnia', 'CustomerID', 'ArotzPnia', 'Incident_direct', 'Begin_date']].copy()
app_counts_df['Begin_date'] = pd.to_datetime(app_counts_df['Begin_date'], format='%Y/%m/%d').dt.normalize()
app_counts_df = app_counts_df.merge(df[['tz', 'mahzor_acharon']], 

                                    left_on="CustomerID", right_on="tz").drop("tz", axis=1)
app_counts_df['days_to_giyus'] = app_counts_df['mahzor_acharon'] - app_counts_df['Begin_date']
app_counts_df['days_to_giyus'] = app_counts_df['days_to_giyus'].dt.days
app_counts_df['days_to_giyus'] = pd.cut(app_counts_df['days_to_giyus'], 

                                        bins=[-10000, 0, 14, 30, 60, 100, 100000],

                                        labels=["after_giyus", "0-14days", "14-30days", 

                                                "30-60days", "60-100days", "100+days"])
app_counts_df['days_to_giyus'] = app_counts_df['days_to_giyus'].astype("object")
arotz_train_mask = app_counts_df['ArotzPnia'].isin(app_train_df['ArotzPnia'])

arotz_test_mask = app_counts_df['ArotzPnia'].isin(app_test_df['ArotzPnia'])

arotz_na = app_counts_df["ArotzPnia"].isna()
arotz_df = app_counts_df[~arotz_na & arotz_train_mask & arotz_test_mask].copy()
app_arotz = pd.crosstab(arotz_df['CustomerID'],arotz_df['ArotzPnia'])
app_arotz.columns = ["application_count_artoz_" + column for column in app_arotz.columns]
direction_df = app_counts_df[app_counts_df['Incident_direct'].isin(["נכנס", "יוצא"])]
app_direction = pd.crosstab(direction_df['CustomerID'],direction_df['Incident_direct'])
app_direction.columns = ["application_count_direction_" + column for column in app_direction.columns]
app_timebucket = pd.crosstab(app_counts_df['CustomerID'],app_counts_df['days_to_giyus'])
app_timebucket.columns = ["application_count_days_before_giyus_" + column for column in app_timebucket.columns]
app_total = app_counts_df[['Mispar_Pnia', 

                           'CustomerID']].groupby(['CustomerID']).count()
app_total.rename(columns = {"Mispar_Pnia": "applications_count"}, inplace=True)
data_frames = [df_sub1, df_sub2, df_submix, df_sub3, app_arotz, app_direction, app_timebucket, app_total]
from functools import reduce
app_merged = reduce(lambda  left,right: pd.merge(left,right,on=['CustomerID'],

                                                how='outer'), data_frames).fillna(0)
train_mask = df['source']=="train"

train_df = df[train_mask].copy()
test_mask = df['source']=="test"

test_df = df[test_mask].copy()
na_columns = train_df.columns[train_df.isna().any()]
missing_indicator = MissingIndicator().fit(train_df[na_columns])
indicator_column_names = [x + "_is_missing" for x in na_columns]
missing_indicator_df = pd.DataFrame(missing_indicator.transform(df[na_columns]), columns = indicator_column_names)
#convert True, False to 0 & 1

missing_indicator_df = missing_indicator_df.astype(int)
df = pd.concat([df, missing_indicator_df], axis=1)
imputer = SimpleImputer(missing_values = np.nan, strategy = 'median').fit(train_df[na_columns])
df[na_columns] = imputer.transform(df[na_columns])
ohe = OneHotEncoder(sparse=False, dtype=np.int).fit(train_df[['destination']])
n_destinations = len(ohe.categories_[0])
dest_column_names = ["is_destination_" + str(x+1) for x in range(n_destinations)]
destination_dummies = pd.DataFrame(ohe.transform(df[['destination']]), columns=dest_column_names)
df = pd.concat([df, destination_dummies], axis=1)
final_df = df.merge(app_merged, how="left", left_on="tz", right_index=True).fillna(0)
train_df = final_df[final_df["source"]=="train"].copy()
#add nesher column

train_df = train_df.merge(raw_train_df[["tz", "nesher"]], how="inner", on="tz")
non_numeric_columns = ['mispar_ishi', 'tz', 'destination', 'mahzor_acharon', 'nesher', 'source']

train_non_numeric = train_df[non_numeric_columns].copy()

train_df.drop(non_numeric_columns, axis=1, inplace=True)
df_columns = pd.Series(train_df.columns)
train_df = csr_matrix(train_df)
test_df = final_df[final_df["source"]=="test"].copy()
non_numeric_columns = ['mispar_ishi', 'tz', 'destination', 'mahzor_acharon', 'source']

test_non_numeric = test_df[non_numeric_columns].copy()

test_df.drop(non_numeric_columns, axis=1, inplace=True)
test_df = csr_matrix(test_df)
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import PredefinedSplit, cross_val_score

from sklearn.feature_selection import mutual_info_classif

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics
from xgboost import XGBClassifier
SEED = 42
#true_test_mode == True means Training on all "train_df" and predicting on "test_df" without knowing the real labels.

true_test_mode = True
#convert mahzor to code from 0-4 for later use in Cross Validation

mahzor = LabelEncoder().fit_transform(train_non_numeric['mahzor_acharon'])
y = np.int32(train_non_numeric["nesher"])
if true_test_mode:

    X_train = train_df

    y_train = train_non_numeric["nesher"].copy()

    X_test = test_df

    train_mahzor = mahzor

else:

    # Creating Inner Test set by predicting last mahzor results

    X_train = train_df[mahzor < 4]

    y_train = y[mahzor<4]

    X_test = train_df[mahzor == 4]

    y_test = y[mahzor == 4]

    train_mahzor = mahzor[mahzor<4]
model = XGBClassifier(max_depth=6, n_jobs=-1, random_state=SEED)
X = vstack([X_train, X_test], format="csr")
# this creates a list of "0"s and "1"s according to the number of rows in each dataset

y = (X_train.shape[0] * [0]) + (X_test.shape[0] * [1])
imp = mutual_info_classif(X,y)
imp_df = pd.DataFrame({"importance" : imp})

imp_df['feature_name'] = df_columns.values

imp_df.sort_values(by="importance", ascending=False, inplace=True)
imp_df.head(20)
similar_distribution_feature_indices = (imp_df[imp_df['importance']<=0.01]).index
# If there are "informative" features - this means that these features have different distributions between train and test.

# Therefore we will keep only those who got low scores in the feature_importance score
X_train = X_train[:,similar_distribution_feature_indices]
X_test = X_test[:,similar_distribution_feature_indices]
df_columns = df_columns[similar_distribution_feature_indices].copy().reset_index(drop=True)
model = model.fit(X_train, y_train)

imp_df = pd.DataFrame({"importance" : model.feature_importances_})

imp_df['feature_name'] = df_columns.values

imp_df.sort_values(by="importance", ascending=False, inplace=True)
imp_df.head(20)
informative_feature_indices = (imp_df[imp_df['importance']>0]).index
X_train = X_train[:,informative_feature_indices].copy()
X_test = X_test[:,informative_feature_indices].copy()
df_columns = df_columns[informative_feature_indices].copy().reset_index(drop=True)
def create_predefined_splits(cv_splits, X, y):

    test_folds_idx = np.zeros(len(y), dtype=np.int)

    i = 0

    for train_index, test_index in cv_splits.split(X, y):

        test_folds_idx[test_index] = i

        i+=1

    return PredefinedSplit(test_folds_idx)
cv_splits = PredefinedSplit(train_mahzor)
scores = cross_val_score(model, X_train, y_train, scoring="roc_auc", cv = cv_splits, n_jobs=-1)
print("We got " + str(round(scores.mean(), 3)) + " AUC with CV on the training set!")
model = model.fit(X_train, y_train)
test_probs = model.predict_proba(X_test)

test_probs = test_probs[:,1]
if true_test_mode == False:

    val_auc = metrics.roc_auc_score(y_test, test_probs)

    print("We got " + str(round(val_auc, 3)) + " AUC on the validation set!")
import time

timestr = time.strftime("%Y%m%d-%H%M%S")
if true_test_mode:

    submission = pd.DataFrame({"TZ": test_non_numeric['tz'], 'NESHER' : test_probs})

    submission.to_csv('./submissions/sub' + timestr + '.csv', index=False)

    

    solution = pd.read_csv('./data/solution.csv')

    solution = solution.merge(submission, on="TZ")

    test_auc = metrics.roc_auc_score(solution["NESHER_x"], solution["NESHER_y"])

    print("We got " + str(round(test_auc, 3)) + " AUC on the test set!")