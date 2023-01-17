%load_ext autoreload
%autoreload 2

%matplotlib inline
from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display

from sklearn import metrics
#PATH = "data/"
#!ls {PATH}
# Read in the data
#df_raw = pd.read_csv(f'{PATH}Interview.csv', parse_dates = ['Date of Interview'])

#Alt read in for Kaggle kernal
df_raw = pd.read_csv('../input/Interview.csv')
df_raw.head()
# Removing empty variables
# I'll go ahead and put all this work in a new df so I have an original copy if I need to go back for any reason.
interview_df = df_raw.drop(['Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25', 'Unnamed: 26', 'Unnamed: 27'], axis = 1)

# Renaming variables to strings that are a little easier to work with.
interview_df.columns = ['Date', 'Client', 'Industry', 'Location', 'Position', 'Skillset',
                        'Interview_Type', 'ID', 'Gender', 'Cand_Loc', 'Job_Loc', 'Venue',
                        'Native_Loc', 'Permission', 'Hope', 'Three_hour_call', 'Alt_phone',
                        'Resume_Printout', 'Clarify_Venue', 'Shared_Letter', 'Expected', 
                        'Attended', 'Martial_Status']
interview_df.shape
for c in interview_df.columns:
    print(c)
    print(interview_df[c].unique())
print(interview_df['Date'].unique())
def clean_date(date):
    date = date.str.strip()
    date = date.str.split("&").str[0]
    date = date.str.replace('–', '/')
    date = date.str.replace('.', '/')
    date = date.str.replace('Apr', '04')
    date = date.str.replace('-', '/')
    date = date.str.replace(' ', '/')
    date = date.str.replace('//+', '/')
    return date
interview_df['Date'] = clean_date(interview_df['Date'])
# Through exploration I discovered row 1233 had a ton of missign values, including the date. Since I can't parse 
# a missing date (and I'm going to create a bunch of new features from that date variable) I'm just removing the row
interview_df.drop(interview_df.index[[1233]], inplace = True)
# Create my new date variables
interview_df['year'] = interview_df['Date'].str.split("/").str[2]
interview_df['day'] = interview_df['Date'].str.split("/").str[0]
interview_df['month'] = interview_df['Date'].str.split("/").str[1]

# This will find the short years and replace with long years
interview_df['year'].replace(['16', '15'], ['2016', '2015'], inplace = True)

# Finally I create the new Date column
interview_df['date'] = pd.to_datetime(pd.DataFrame({'year': interview_df['year'],
                                            'month': interview_df['month'],
                                            'day': interview_df['day']}), format = '%Y-%m-%d')
interview_df.head()
interview_df.drop(['Date', 'year', 'month', 'day'], axis = 1, inplace = True)
interview_df = pd.concat([interview_df[c].astype(str).str.lower() for c in interview_df.columns], axis = 1)
interview_df = pd.concat([interview_df[c].astype(str).str.strip() for c in interview_df.columns], axis = 1)
interview_df['Client'].value_counts()
interview_df['Client'].replace(['standard chartered bank chennai', 'aon hewitt gurgaon', 'hewitt'], 
                              ['standard chartered bank', 'aon hewitt', 'aon hewitt'], inplace = True)
interview_df['Industry'].replace(['it products and services', 'it services'], 
                              ['it', 'it'], inplace = True)

interview_df['Location'].replace(['- cochin-'], 
                              ['cochin'], inplace = True)

# I'm really not sure about this Interview_Type variable. I'd ask if given the chance. For now I'm just going
# to use 'scheduled walkin', 'walkin' and 'scheduled'
interview_df['Interview_Type'].replace(['scheduled walk in', 'sceduled walkin'],
                                       ['scheduled walkin', 'scheduled walkin'], inplace = True)
# I wonder why  cochin is always messed up?
interview_df['Cand_Loc'].replace(['- cochin-'], 
                              ['cochin'], inplace = True)
interview_df['Job_Loc'].replace(['- cochin-'], 
                              ['cochin'], inplace = True)
interview_df['Venue'].replace(['- cochin-'], 
                              ['cochin'], inplace = True)
# I'm assuming all these native locations are actual places. I didn't check them all
interview_df['Native_Loc'].replace(['- cochin-'], 
                              ['cochin'], inplace = True)
# I don't know if nan's and 'not yet' are actually different, but I'm treating them like they are
interview_df['Permission'].replace(['na', 'not yet', 'yet to confirm'], 
                              ['nan', 'tbd', 'tbd'], inplace = True)
interview_df['Hope'].replace(['na', 'not sure', 'cant say'], 
                              ['nan', 'unsure', 'unsure'], inplace = True)
interview_df['Three_hour_call'].replace(['na', 'no dont'], 
                              ['nan', 'no'], inplace = True)
interview_df['Hope'].replace(['na', 'not sure', 'cant say'], 
                              ['nan', 'unsure', 'unsure'], inplace = True)
interview_df['Alt_phone'].replace(['na', 'no i have only thi number'], 
                              ['nan', 'no'], inplace = True)
interview_df['Resume_Printout'].replace(['na', 'not yet', 'no- will take it soon'], 
                              ['nan', 'ny', 'ny'], inplace = True)
interview_df['Clarify_Venue'].replace(['na', 'no- i need to check'], 
                              ['nan', 'no'], inplace = True)
interview_df['Shared_Letter'].replace(['na', 'not sure', 'need to check', 'not yet', 'yet to check',
                                       'havent checked'],
                                      ['nan', 'unsure', 'unsure', 'unsure', 'unsure', 'unsure'], inplace = True)
interview_df['Expected'].replace(['na', '11:00 am', '10.30 am'], 
                              ['nan', 'nan', 'nan'], inplace = True)
interview_df['Attended'].replace(['yes', 'no'],
                                 [1, 0], inplace = True)
modeling_df = interview_df.drop(['Skillset'], axis = 1)
modeling_df = modeling_df[modeling_df['date'] < '2018-01-01']
add_datepart(modeling_df, 'date')
modeling_df.columns
# Handling categorical variables
train_cats(modeling_df)
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        display(df)
display_all(modeling_df.tail().T)
df, y, nas = proc_df(modeling_df, "Attended")
display_all(df.tail().T)
display_all(y.T)
def split_vals(a, n):
    return a[:n].copy(), a[n:].copy()
n_valid = int(len(modeling_df) * .2)
n_trn = len(modeling_df) - n_valid
raw_train, raw_valid = split_vals(modeling_df, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)
m_base = RandomForestClassifier(n_jobs = -1, oob_score=True)
%time m_base.fit(X_train, y_train)
print("Training Acc:", round(m_base.score(X_train, y_train),5)),
print("Validation Acc:", round(m_base.score(X_valid, y_valid), 5)),
print("Out-of-Bag Acc:", round(m_base.oob_score_, 5))
probs = m_base.predict_proba(X_valid)
probs = [p[1] for p in probs]
fpr, tpr, thresholds = metrics.roc_curve(y_valid, probs)
roc_auc = metrics.roc_auc_score(y_valid, probs)
plt.plot(fpr, tpr, color = 'darkorange',
        label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.legend(loc="lower right")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
draw_tree(m_base.estimators_[0], df, precision = 3)
m = RandomForestClassifier(min_samples_leaf=5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print("Training Acc:", round(m.score(X_train, y_train),5)),
print("Validation Acc:", round(m.score(X_valid, y_valid), 5))
print("Out-of-Bag Acc:", round(m.oob_score_, 5))
draw_tree(m.estimators_[0], df, precision = 3)
m = RandomForestClassifier(min_samples_leaf=5, max_features = 'log2',n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print("Training Acc:", round(m.score(X_train, y_train),5)),
print("Validation Acc:", round(m.score(X_valid, y_valid), 5))
print("Out-of-Bag Acc:", round(m.oob_score_, 5))
feature_imp = rf_feat_importance(m, df)
feature_imp.plot('cols', 'imp', figsize = (10, 6), legend = False)
def plot_fi(fi):
    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend = False)
plot_fi(feature_imp)
keep = feature_imp[feature_imp.imp > 0.0075].cols
len(keep)
df_keep = df[keep].copy()
X_train, X_valid = split_vals(df_keep, n_trn)
m = RandomForestClassifier(min_samples_leaf = 5,
                           max_features='log2', n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print("Training Acc:", round(m.score(X_train, y_train),5)),
print("Validation Acc:", round(m.score(X_valid, y_valid), 5))
print("Out-of-Bag Acc:", round(m.oob_score_, 5))
fi = rf_feat_importance(m, df_keep)
plot_fi(fi)
df_trn, y_trn, nas = proc_df(modeling_df, 'Attended', max_n_cat = 7)
X_train, X_valid = split_vals(df_trn, n_trn)
m = RandomForestClassifier(min_samples_leaf = 5,
                           max_features='log2', n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print("Training Acc:", round(m.score(X_train, y_train),5)),
print("Validation Acc:", round(m.score(X_valid, y_valid), 5))
print("Out-of-Bag Acc:", round(m.oob_score_, 5))
fi = rf_feat_importance(m, df_trn)
plot_fi(fi[:25])
from scipy.cluster import hierarchy as hc
corr = np.round(scipy.stats.spearmanr(df_keep).correlation,4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method = 'average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels = df_keep.columns, orientation='left',
                          leaf_font_size =16)
plt.show()
def get_oob(df):
    m = RandomForestClassifier(n_estimators=40, min_samples_leaf=8, max_features='sqrt', n_jobs=-1, oob_score=True)
    x, _ = split_vals(df, n_trn)
    m.fit(x, y_train)
    return m.oob_score_
get_oob(df_keep)
for c in ('Week', 'Dayofyear'):
    print(c, get_oob(df_keep.drop(c, axis = 1)))
modeling_df.drop(['Dayofyear'], axis = 1, inplace = True)
df_trn, y_trn, nas = proc_df(modeling_df, 'Attended', max_n_cat = 7)
X_train, X_valid = split_vals(df_trn, n_trn)
m = RandomForestClassifier(min_samples_leaf = 5,
                           max_features='log2', n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print("Training Acc:", round(m.score(X_train, y_train),5)),
print("Validation Acc:", round(m.score(X_valid, y_valid), 5))
print("Out-of-Bag Acc:", round(m.oob_score_, 5))
probs = m.predict_proba(X_valid)
probs = [p[1] for p in probs]
fpr, tpr, thresholds = metrics.roc_curve(y_valid, probs)
roc_auc = metrics.roc_auc_score(y_valid, probs)
plt.plot(fpr, tpr, color = 'darkorange',
        label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.legend(loc="lower right")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
df, y, nas = proc_df(modeling_df, "Attended")
m = RandomForestClassifier(min_samples_leaf = 5,
                           max_features='log2', n_jobs=-1, oob_score=True)
m.fit(df, y)
print("Training Acc:", round(m.score(df, y),5)),
print("Out-of-Bag Acc:", round(m.oob_score_, 5))