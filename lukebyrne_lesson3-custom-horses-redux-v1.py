%matplotlib inline
%reload_ext autoreload
%autoreload 2
%load_ext Cython
from fastai.structured import *
from fastai.column_data import *
np.set_printoptions(threshold=50, edgeitems=20)
from sklearn.model_selection import train_test_split

PATH='../input/'
df_orig = pd.read_csv(f'{PATH}horses.csv')
df_orig.columns
# This allows us to act two mins out, means we can manually get our quinellas and trifectas down
df_orig['bf_odds'] = df_orig['bf_odds_two_mins_out']
df_orig['vic_tote'] = df_orig['vic_tote_two_mins_out']
df_orig['nsw_tote'] = df_orig['nsw_tote_two_mins_out']
# Rows count BEFORE position_two checked for NaN
n = len(df_orig); n
# Remove all rows where position_two is nan
df_orig = df_orig[np.isfinite(df_orig['position_two'])]
# Rows count AFTER position_two checked for NaN
n = len(df_orig); n
# Fill previous_margin NA with -1
#df_orig = df_orig['previous_margin'].fillna(5).astype(np.int32)
# The rest backfill
df_orig = df_orig.fillna(method='bfill')
# Convert market_name, lets get the middle and last piece from it i.e. Distance and Mdn 3yo etc
foo = lambda x: x.split(' ')[2].lower()
df_orig['race_type'] = df_orig['market_name'].map(foo)
foo = lambda x: x.split(' ')[1].lower().replace('m', '')
df_orig['distance'] = df_orig['market_name'].map(foo)
# Clean the horse name
import re
foo = lambda x: x.lower()
df_orig['name_clean'] = df_orig['name'].map(foo)

bar = lambda x: x.replace('. ', '')
df_orig['name_clean'] = df_orig['name_clean'].map(bar)

doh = lambda x: re.sub('[0-9]', '', x)
df_orig['name_clean'] = df_orig['name_clean'].map(doh)

#df_orig['name_clean'] = df_orig['runner_name_uuid']
# Clean and lowercase the jockey name
foo = lambda x: x.lower().rstrip(' *')
df_orig['jockey_clean'] = df_orig['jockey'].map(foo)
# Clean and lowercase the trainer name
foo = lambda x: x.lower().rstrip(' *')
df_orig['trainer_clean'] = df_orig['trainer'].map(foo)
# Get the sire
foo = lambda x: x.lower().rstrip(' *')
df_orig['sire_clean'] = df_orig['sire'].map(foo)
# Get the dam
foo = lambda x: x.lower().rstrip(' *')
df_orig['dam_clean'] = df_orig['dam'].map(foo)
foo = lambda x: re.sub('[0-9]', '', x)
df_orig['condition'] = df_orig['condition'].map(foo)
# # Lets just look at the major meetings
# venues = [
#     'Canterbury', 
#     'Caulfield',
#     'Eagle Farm', 
#     'Flemington', 
#     'Geelong', 
#     'Gold Coast', 
#     'Ipswich', 
#     'Kembla Grange', 
#     'Moonee Valley', 
#     'Mornington', 
#     'Morphettville', 
#     'Randwick', 
#     'Rosehill', 
#     'Sandown', 
#     'Warrnambool', 
#     'Warwick Farm'
# ]

# df_orig = df_orig[df_orig['venue_name'].isin(venues)]
# n = len(df_orig); n
# # Lets just look at the major races
# race_types = [
#     'mdn', 
#     '2yo', 
#     '3yo', 
#     'hcap', 
#     'cl1', 
#     'cl2', 
#     'cl3', 
#     'cl5', 
#     'wfa', 
#     'listed', 
#     'grp3', 
#     'cup', 
#     'hrd', 
#     'stpl',
#     'qlty', 
#     'cl6', 
#     'cl4', 
#     'grp2', 
#     'qtly', 
#     'grp1', 
#     'hcp', 
#     '4yo'
# ]

# df_orig = df_orig[df_orig['race_type'].isin(race_types)]
# n = len(df_orig); n
add_datepart(df_orig, 'date', drop=False)
# Get the sex of the horse and jockey on either side
# Convert form_comment to a sentiment
# Odds 1-10, mean median slope etc ??
# Condition performance win/place ratio
# Get overall, track, distance, track_distance
# Adding in margin here is essentially leakage, we need the margin from the previous race
# Select the columns we want from df_orig
columns = [    
    'position_two',
    'market_id',
    'runner_id',
    'bf_odds',
#     'nsw_tote',
#     'vic_tote',
    'date',
    'jockey_clean',
    'trainer_clean',
    'venue_name',              
    'race_number',
    'condition',
    'weather',
    'distance',
    'race_type',
    'sire_clean',
    'name_clean', 
    'barrier',
    'blinkers',
    'emergency',
    'dfs_form_rating',
    'tech_form_rating',
    'total_rating_points',
    'handicap_weight',
    'penalty',
    'sex',
    'age',
    'jockey_sex',
    'class_level',
    'field_strength',
    'days_since_last_run',
    'runs_since_spell',
    'previous_margin',
    'Day',
    'Month',
    'Dayofweek', 
    'Elapsed'
]
df_all = df_orig[columns]
df_all.head(5)
cat_vars = [
    'date',
    'jockey_clean',
    'trainer_clean',
    'venue_name',              
    'race_number', 
    'condition',
    'weather',
    'distance',
    'race_type',
    'sire_clean',
    'name_clean',
    'barrier',
    'blinkers',
    'emergency',
    'dfs_form_rating',
    'tech_form_rating',
    'total_rating_points',
    'handicap_weight',
    'penalty',
    'sex',
    'age',
    'jockey_sex',
    'class_level',
    'field_strength',
    'days_since_last_run',
    'runs_since_spell',
    'previous_margin',
    'Day',
    'Month',
    'Dayofweek', 
    'Elapsed'
]

contin_vars = ['bf_odds']
for cat in cat_vars: df_all[cat] = df_all[cat].astype('category').cat.as_ordered()
for contin in contin_vars: df_all[contin] = df_all[contin].astype('float32')
df_all['position_two'] = df_all['position_two'].astype('float32')

# Reset the index
df_all = df_all.reset_index(drop=True)
df_all.shape
# Use a date split?
date_split = datetime.datetime(2018,5,30)
date_split
# Split out our training set
#training_df = df_all[df_all.index <= split]
training_df = df_all[df_all.date <= date_split]
training_df.reset_index(inplace=True)
training_df.shape
# Split out our test set
#test_df = df_all[df_all.index > split].copy()
test_df = df_all[df_all.date > date_split].copy()
test_df.reset_index(inplace=True)
test_df_orig = test_df
test_df.shape
# Setup our df_train
df_train, y, nas, mapper = proc_df(training_df, 'position_two', do_scale=True)
yl = np.log(y+0.01)
# Setup our df_test
df_test, _, nas, mapper = proc_df(test_df, 'position_two', do_scale=True,
                                  mapper=mapper, na_dict=nas)
training_validaton_split = 0.999
# Get our validation ids
split = math.ceil(len(df_train) * training_validaton_split)
val_idx = np.flatnonzero((df_train.index > split))
len(val_idx)
# Get our embeddings
cat_sz = [(c, len(training_df[c].cat.categories)+1) for c in cat_vars]
emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]
# Create a loss function
def inv_y(a): return np.exp(a)

def exp_rmspe(y_pred, targ):
    targ = inv_y(targ)
    pct_var = (targ - inv_y(y_pred))/targ
    return math.sqrt((pct_var**2).mean())

max_log_y = np.max(yl)
y_range = (0, max_log_y*1.2)
# Batch size
bs = 128
# Hidden layers
#hl = [1000]
hl = [1000, 500]
#hl = [2000, 1000, 500, 250, 100]
# Hidden layers learning rate
#hl_lr = [0.001]
hl_lr = [0.001, 0.01]
#hl_lr = [0.001, 0.01, 0.01, 0.01]

# Setup a model
md = ColumnarModelData.from_data_frame('/tmp/', val_idx, df_train, yl.astype(np.float32), cat_flds=cat_vars, bs=bs,
                                       test_df=df_test)
# Pass in our embeddings
m = md.get_learner(emb_szs, len(df_all.columns)-len(cat_vars),
                    0.04, 1, hl, hl_lr, y_range=y_range)
#m.summary
# Fit
lr = 1e-4
#m.fit(lr, 1, metrics=[exp_rmspe])
m.fit(lr, 3, metrics=[exp_rmspe], cycle_len=1)
# Check out test set
x,y = m.predict_with_targs()
exp_rmspe(x,y)
# Get our predictons and marry them up with other test data
pred_test = m.predict(True)
pred_test = np.exp(pred_test)
predictions = pred_test.flatten().tolist()
predictions= [round(prediction, 2) for prediction in predictions ]
market_ids = [round(id, 0) for id in test_df_orig.market_id ]
runner_ids = [round(id, 0) for id in test_df_orig.runner_id ]
bf_odds = [round(id, 2) for id in test_df_orig.bf_odds ]
# Place all our prediction data in a df predictions_vs_actuals
# p_vs_a == predictions_vs_actuals
p_vs_a = list(zip(market_ids, runner_ids, predictions, test_df_orig.position_two, bf_odds))
p_vs_a_df = pd.DataFrame(p_vs_a)
p_vs_a_df.columns = ['market_id', 'runner_id', 'predicted', 'actual', 'bf_odds']
p_vs_a_df = p_vs_a_df.sort_values(['market_id','predicted'], ascending=True)
len(p_vs_a_df)
#p_vs_a_df.head(5)
# Determine which runners would be in the top 25 of predictions
quantile = lambda x: x.quantile(.25)
p_vs_a_df['quantile'] = p_vs_a_df.groupby('market_id')['predicted'].transform(quantile)
# Create a new dataframe where only have the top quantile of runners
top_df = p_vs_a_df.loc[p_vs_a_df['predicted'] <= p_vs_a_df['quantile']]
top_df = top_df.reset_index(drop=True)
#top_df.head(5)
BF_COMMISSION = 0.07
ODDS_PREMIUM = 0.90
WIN_STAKE = 5
#win = lambda x: -WIN_STAKE if x.actual > 1 else (((ODDS_PREMIUM * x.bf_odds) * WIN_STAKE)) * (1 - BF_COMMISSION)
win = lambda x: -WIN_STAKE if x.actual > 1 else (((ODDS_PREMIUM * x.bf_odds) * WIN_STAKE) - WIN_STAKE) * (1 - BF_COMMISSION)
top_df['back'] = top_df.apply(win, axis=1)
numbers_bets = len(top_df)
numbers_bets
win = top_df['back'].sum()
win
#ROI
roi = win / (numbers_bets * WIN_STAKE)
roi
top_df['back'].cumsum().plot(x='index', y='win', figsize=(20,10))
lay = lambda x: -((ODDS_PREMIUM * x.bf_odds) * WIN_STAKE) if x.actual == 1 else (WIN_STAKE * (1 - BF_COMMISSION))
top_df['lay'] = top_df.apply(lay, axis=1)
win = top_df['lay'].sum()
win
#ROI
roi = win / (numbers_bets * WIN_STAKE)
roi
top_df['lay'].cumsum().plot(x='date', y='win', figsize=(20,10))
grouped_df = top_df.groupby('market_id')['actual'].apply(list)
grouped_df = pd.DataFrame(grouped_df.to_frame().to_records())
grouped_df.columns = ['market_id', 'positions']
#grouped_df.head(10) 
first_numbers = {1.0}
first = lambda x: True if first_numbers.issubset(x.positions) else False
grouped_df['first'] = grouped_df.apply(first, axis=1)
exacta_numbers = {1.0, 2.0}
exacta = lambda x: True if exacta_numbers.issubset(x.positions) else False
grouped_df['exacta'] = grouped_df.apply(exacta, axis=1)
trifecta_numbers = {1.0, 2.0, 3.0}
trifecta = lambda x: True if trifecta_numbers.issubset(x.positions) else False
grouped_df['trifecta'] = grouped_df.apply(trifecta, axis=1)
first_four_numbers = {1.0, 2.0, 3.0, 4.0}
first_four = lambda x: True if first_four_numbers.issubset(x.positions) else False
grouped_df['first_four'] = grouped_df.apply(first_four, axis=1)
grouped_df.count()
grouped_df[grouped_df['first'] == True].count()
grouped_df[grouped_df['exacta'] == True].count()
grouped_df[grouped_df['trifecta'] == True].count()
grouped_df[grouped_df['first_four'] == True].count()
odds_df = pd.read_csv(f'{PATH}odds_exotics.csv')
columns = ['market_id', 'quinella', 'exacta', 'trifecta', 'first_four']
odds_df = odds_df[columns]
odds_df = odds_df.set_index(['market_id'])
def exotic_odds(market_id, column):
    try:
        return odds_df.loc[market_id][column]
    except Exception as e:
        #print(e)
        return 0
# Get the odds for each exotic
grouped_df['quinella_odds'] = grouped_df['market_id'].apply(lambda x: exotic_odds(x, 'quinella'))
grouped_df['exacta_odds'] = grouped_df['market_id'].apply(lambda x: exotic_odds(x, 'exacta'))
grouped_df['trifecta_odds'] = grouped_df['market_id'].apply(lambda x: exotic_odds(x, 'trifecta'))
grouped_df['first_four_odds'] = grouped_df['market_id'].apply(lambda x: exotic_odds(x, 'first_four'))
grouped_df.head(5)
exacta_df = grouped_df[grouped_df['positions'].map(len) >= 2]
columns = ['market_id', 'positions', 'exacta', 'exacta_odds']
exacta_df = exacta_df[columns]
def exacta_cost(positions):
    length = len(positions)    
    if length <= 2:
        return 2
    elif length <= 3:
        return 6
    elif length <= 4:
        return 12
    elif length <= 5:
        return 20
    else:
        return 2
    
exacta_df['cost'] = exacta_df['positions'].apply(lambda x: exacta_cost(x))
exacta_df['cost'].fillna(2, inplace=True)
exacta_winnings = lambda x: x.exacta_odds - x.cost if x.exacta == True  else -x.cost
exacta_df['winnings'] = exacta_df.apply(exacta_winnings, axis=1)
exacta_df['winnings'].sum()
exacta_df.head(5)
trifecta_df = grouped_df[grouped_df['positions'].map(len) >= 3]
columns = ['market_id', 'positions', 'trifecta', 'trifecta_odds']
trifecta_df = trifecta_df[columns]
def trifecta_cost(positions):
    length = len(positions)    
    if length <= 3:
        return 6
    elif length <= 4:
        return 24
    elif length <= 5:
        return 60
    else:
        return 6
trifecta_df['cost'] = trifecta_df['positions'].apply(lambda x: trifecta_cost(x))
trifecta_df['cost'].fillna(6, inplace=True)
trifecta_winnings = lambda x: x.trifecta_odds - x.cost if x.trifecta == True  else -x.cost
trifecta_df['winnings'] = trifecta_df.apply(trifecta_winnings, axis=1)
trifecta_df['winnings'].sum()
trifecta_df.head(5)
firt_four_df = grouped_df[grouped_df['positions'].map(len) >= 4]
columns = ['market_id', 'positions', 'first_four', 'first_four_odds']
firt_four_df = firt_four_df[columns]
def first_four_cost(positions):
    length = len(positions)    
    if length <= 4:
        return 24
    elif length <= 5:
        return 120
    elif length <= 6:
        return 360
    else:
        return 24
firt_four_df['cost'] = firt_four_df['positions'].apply(lambda x: first_four_cost(x))
firt_four_df['cost'].fillna(6, inplace=True)
first_four_winnings = lambda x: x.first_four_odds - x.cost if x.first_four == True  else -x.cost
firt_four_df['winnings'] = firt_four_df.apply(first_four_winnings, axis=1)
firt_four_df['winnings'].sum()
firt_four_df.head(5)
top_df_fn=f'/tmp/top_df.csv'
top_df.to_csv(top_df_fn, index=False)
FileLink(top_df_fn)
grouped_df_fn=f'/tmp/grouped_df.csv'
grouped_df.to_csv(grouped_df_fn, index=False)
FileLink(grouped_df_fn)