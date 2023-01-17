from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, log_loss
from IPython.display import display

from sklearn import metrics
%load_ext autoreload
%autoreload 2

%matplotlib inline
#!kaggle competitions download -c kobe-bryant-shot-selection
df_raw = pd.read_csv('../input/data.csv', low_memory=False, 
                     parse_dates=["game_date"])
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
display_all(df_raw.tail(100).T)
display_all(df_raw.describe(include='all').T)
#Add new fields for date
add_datepart(df_raw, 'game_date')
def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km
#One hot encoding!!!
df_raw=pd.concat([df_raw,pd.get_dummies(df_raw['combined_shot_type'],prefix='combined_shot_type')],axis=1).drop(['combined_shot_type'],axis=1)
def data_cleaning(data):
    
    #Feature engineering
    #Calculate distance from home stadum and add to data
    conditions = [
    ((abs(data['lat'] - 34.0443) >= 0.01) & (abs(data['lat'] - -118.27) >= 0.01)),
    ((abs(data['lat'] - 34.0443) < 0.01) & (abs(data['lat'] - -118.27) < 0.01))
    ]
    choices = [0, 1]
    data['home_game']=np.select(conditions, choices)
    
    data['travel_dist'] = haversine_np(-118.2700,34.0443,data['lon'],data['lat'])
    
    #Round lon and lat to get areas close together
    data.round({'lon': 4, 'lat': 4})
    
    #Split action type into two features - first feature contains Jumping,Running etc.
    data['action_type_first'] = [action_type_string.split(' ')[0] for action_type_string in data['action_type']]
    data['action_type_second'] = [str(action_type_string.split(' ')[1:]) for action_type_string in data['action_type']]
    
    data.drop(['action_type','game_id', 'game_Elapsed','game_event_id', 'matchup', 'team_name', 'game_Month', 'game_Dayofyear', 'lon', 'lat'], axis=1, inplace=True)
    
df_clean = df_raw.copy()
data_cleaning(df_clean)
print("Data cleaned")

display_all(df_clean.tail(100).T)
train_cats(df_clean) #Setup categories
#Sort the labels
df_clean.shot_zone_basic.cat.set_categories(['Restricted Area', 'In The Paint (Non-RA)', 'Mid-Range','Above the Break 3', 'Left Corner 3', 'Right Corner 3', 'Backcourt'], ordered=True, inplace=True)
#df_clean.shot_zone_range.cat.set_categories(['Less Than 8 ft.', '8-16 ft.','16-24 ft.', '24+ ft.', 'Back Court Shot'], ordered=True, inplace=True)
df_clean.shot_zone_area.cat.set_categories(['Back Court(BC)', 'Center(C)', 'Left Side Center(LC)', 'Right Side Center(RC)','Left Side(L)', 'Right Side(R)'], ordered=True, inplace=True)

#df_clean.action_type.cat.categories #Could do some auto sort method for this?
df_test, _, _ = proc_df(df_clean[df_clean['shot_made_flag'].isnull()], 'shot_made_flag')

#tv = Train and Validation data
df_tv, y_tv, nas = proc_df(df_clean[df_clean['shot_made_flag'].notnull()], 'shot_made_flag')

ids=df_test['shot_id']

df_test.drop(['shot_id'], axis=1, inplace=True)
df_tv.drop(['shot_id'], axis=1, inplace=True)

def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 5000  # Could use cross validato
n_trn = len(df_tv)-n_valid
#raw_train, raw_valid = split_vals(df_raw, n_trn)
X_train, X_valid = split_vals(df_tv, n_trn)
y_train, y_valid = split_vals(y_tv, n_trn)

X_train.shape, y_train.shape, X_valid.shape
m = RandomForestRegressor(n_estimators=30, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
preds=m.predict(X_valid)
#print(classification_report(y_valid, preds))
print(m.oob_score_)
print(log_loss(y_valid, preds))
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
fi = rf_feat_importance(m, df_tv);
plot_fi(fi[:25]);
to_keep = fi[fi.imp>0.005].cols; len(to_keep)
df_keep_tv = df_tv[to_keep].copy()
df_keep_test = df_test[to_keep].copy()

X_train, X_valid = split_vals(df_keep_tv, n_trn)
print(X_valid.shape)
m = RandomForestRegressor(n_estimators=100, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
preds=m.predict(X_valid)
#print(classification_report(y_valid, preds))
print(log_loss(y_valid, preds))

fi = rf_feat_importance(m, df_keep_tv);
plot_fi(fi[:25]);
from scipy.cluster import hierarchy as hc
corr = np.round(scipy.stats.spearmanr(df_keep_tv).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df_keep_tv.columns, orientation='left', leaf_font_size=16)
plt.show()
def get_oob(df):
    m = RandomForestRegressor(n_estimators=100, n_jobs=-1, oob_score=True)
    x, _ = split_vals(df, n_trn)
    m.fit(x, y_train)
    score = log_loss(y_valid, preds)
    return score
#This was used to see what features could be removed - was then added to data cleaning function
#for c in ('game_','game_Dayofyear', 'game_Week'):
#    print(c, get_oob(df_keep_tv.drop(c, axis=1)))
#set_rf_samples(10000)
reset_rf_samples()
m = RandomForestRegressor(n_estimators=800, max_features=0.8, n_jobs=-1, oob_score=False)
m.fit(X_train, y_train)

predicted = m.predict(X_valid)
score = log_loss(y_valid, predicted)
print(score)
#This was used to do hyperparameters optimization
m = RandomForestRegressor()
params={'n_estimators': [900,1000,1100], 'max_features': [0.35], 'max_depth' : [8]}
gs = GridSearchCV(m, params, cv=4)
%time gs.fit(X_train, y_train)
preds=gs.predict(X_valid)

score = log_loss(y_valid, preds)
print(score) #0.6071409938301751 {'max_depth': 8, 'max_features': 0.35, 'n_estimators': 900}
print(gs.best_params_) 
predicted_target = gs.predict(df_keep_test)

submit=pd.DataFrame({'shot_id': ids, 'shot_made_flag': predicted_target})
submit.to_csv('submission.csv', index=False)
!kaggle competitions submit -c kobe-bryant-shot-selection -f submission.csv -m "Regressor"