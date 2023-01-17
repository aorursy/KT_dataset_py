import m5_helpers as h

import pandas as pd 

import numpy as np
help(h)
MAIN_PATH = '/kaggle/input/m5-forecasting-uncertainty/'



train_df = pd.read_csv(F'{MAIN_PATH}sales_train_evaluation.csv')

prices_df = pd.read_csv(F'{MAIN_PATH}sell_prices.csv')

cal_df = pd.read_csv(F'{MAIN_PATH}calendar.csv')
################ pinball loss example ###################

actuals = train_df.head().iloc[0, -28:]



# Add some random noise to simulate predictions 

preds = np.clip(actuals + np.random.normal(scale=.2, size=28), 0, None)



# Lets say we are predicting for the .25 quantile 

u = .25



# So then the SPL for this series and quantile would be

pl = np.where(actuals >= preds, (actuals - preds) * u, (preds - actuals) * (1 - u)).mean()

pl
################## simple pl example function #####################

# This is just for example. We will not use this for our score 

# calculations. 

def pl(actuals, preds, u): 

    return np.where(series >=actuals, (actuals - preds) * u, (preds - actuals) * (1 - u)).mean()
############# weighted scaled pinball loss ################

# To calculate the wspl we will need both the weights and 

# the scaling factor for each series. To that end, we will 

# have to aggregate the data in train_df into all the series.

# We will need a rollup matrix and index to aggregate all 

# series. 

rollup_matrix_csr, rollup_index = h.get_rollup(train_df)



# A weight_dataframe with an index for all the series with the 

# weights and scales would be nice to have. The weights 

# are the same for accuracy and uncertainty, and I have 

# made a function, get_w_df that gets those weights, but

# as of now, get_w_df only gives scaling factors for the wrmsse. 

# We will need different scaling factors the wspl.

w_df = h.get_w_df(

    train_df,

    cal_df,

    prices_df,

    rollup_index,

    rollup_matrix_csr,

    start_test=1914,

)

w_df.head()
###################### spl scaling factor #######################

# We calculate scales for days preceding 

# the start of the testing/scoring period. 

start_test = 1914

df = train_df.loc[:, 'd_1':f'd_{start_test-1}']



# We will need to aggregate all series 

agg_series = rollup_matrix_csr * df.values



# Make sure leading zeros are not included in calculations

agg_series = h.nan_leading_zeros(agg_series)



# Now we can compute the scale and add 

# it as a column to our w_df

scale = np.nanmean(np.abs(np.diff(agg_series)), axis = 1)

scale.shape

w_df['spl_scale'] = scale



# It may also come in handy to have a scaled_weight 

# on hand. 

w_df['spl_scaled_weight'] = w_df.weight / w_df.spl_scale

display(w_df.head())
############## spl scaling factor function ###############

def add_spl_scale(w_df, train_df, rollup_matrix_csr): 

    # We calculate scales for days preceding 

    # the start of the testing/scoring period. 

    start_test = 1914

    df = train_df.loc[:, 'd_1':f'd_{start_test-1}']



    # We will need to aggregate all series 

    agg_series = rollup_matrix_csr * df.values



    # Make sure leading zeros are not included in calculations

    agg_series = h.nan_leading_zeros(agg_series)



    # Now we can compute the scale and add 

    # it as a column to our w_df

    scale = np.nanmean(np.abs(np.diff(agg_series)), axis = 1)

    scale.shape

    w_df['spl_scale'] = scale



    # It may also come in handy to have a scaled_weight 

    # on hand.  

    w_df['spl_scaled_weight'] = w_df.weight / w_df.spl_scale

    

    return w_df
############## spl for all series with quantile u = .25 ##################

start_test = 1914

u = .25

level_12_acutals = train_df.loc[:, f'd_{start_test}': f'd_{start_test + 27}']



# But we need actuals for all series

actuals = rollup_matrix_csr * level_12_acutals.values



# We will just use zero predictions for example

preds = np.zeros(actuals.shape)



# Lets calculate the pinball loss

pl = np.where(actuals >= preds, (actuals - preds) * u, (preds - actuals) * (1 - u)).mean(axis=1)



# Now calculate the scaled pinball loss.  

all_series_spl = pl / w_df.spl_scale

all_series_spl
########## Function for all level pinball loss for quantile u ############

def spl_u(actuals, preds, u):

    """Returns the scaled pinball loss for each series"""

    pl = np.where(actuals >= preds, (actuals - preds) * u, (preds - actuals) * (1 - u)).mean(axis=1)



    # Now calculate the scaled pinball loss.  

    all_series_spl = pl / w_df.spl_scale

    return all_series_spl
########## wspl for all quantiles ############

def wspl(actuals, preds): 

    """

    :acutals:, 9 vertical copies of the ground truth for all series. 

    :preds: predictions for all series and all quantiles. Same 

    shape as actuals"""

    quantiles = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]

    scores = []

    

    # In this case, preds has every series for every  

    # quantile T, so it has 42840 * 9 rows. We first 

    # break it up into 9 parts to get the wspl_T for each.

    # We also do the same for actuals. 

    preds_list = np.split(preds, 9)

    actuals_list = np.split(actuals, 9)

    

    for i in range(9):

        scores.append(spl_u(actuals_list[i], preds_list[i], quantiles[i]))

        

    #  We divide score by 9 

    # to get the average wspl of each quantile. 

    spl = sum(scores) / 9

    wspl = (w_df.weight * spl).sum() / 12

    return wspl
################ Test on zero predictions #################

start_test = 1914

level_12_acutals = train_df.loc[:, f'd_{start_test}': f'd_{start_test + 27}']

actuals = rollup_matrix_csr * level_12_acutals.values

actuals = np.tile(actuals.T, 9).T

preds = np.zeros((42840 * 9, 28))

wspl(actuals, preds)
# Load in a submission we know the score of. 

# This one scored .17921

# We only want the _validaiton portion to test our wspl

sub = pd.read_csv('/kaggle/input/from-point-to-uncertainty-prediction/submission.csv').iloc[:42840 * 9, 1:]

print(wspl(actuals, sub.values), 'is supposed to be close to .17921')

print('This is not good. The problem is our depends on the predictions to be in a specific order.')
# Lets get our index the way we need it 

# to work with our rollup matrix. This is the 

# oder our predictions will naturally be in. 

# If we want to test other sample submissions 

# with our wspl function, they will need to 

# be reindexed to match our index. Therefore, we 

# will need to build a column that matched the 

# formatting of the sample submission exactly.

rollup_matrix_csr, rollup_index = h.get_rollup(train_df)

w_df = h.get_w_df(

    train_df,

    cal_df,

    prices_df,

    rollup_index,

    rollup_matrix_csr,

    start_test=1914,

)



# I think the submission index was created 

# by combining the Agg_Level_1 and Agg_Level_2

# columns. 

w = pd.read_csv('/kaggle/input/m5methods/validation/weights_validation.csv')

    

# We will also need the sample submission file. 

ss = pd.read_csv('/kaggle/input/m5-forecasting-uncertainty/sample_submission.csv')



# Lets just take the first 42840 rows of 

# submissino file for now. 

ss = ss.iloc[:42840]



# Give it w_df index so we can easily 

# look at different levels without 

# having to count rows. 

ss.index = w_df.index



# We will need to see both indexes 

# at different levels, along with the 

# sample submission. 

def p(i, head_length=2): 

    display(w[w.Level_id == f'Level{i}'].head(head_length))

    display(w_df.loc[i].head(head_length))
p(1)
# Lets add a sub_id col to w_df that 

# we will build to match the submission 

# file. 

w_df['sub_id'] = w_df.index.get_level_values(1)



###### level 1-5, 10 change ########

w_df.loc[1:5, 'sub_id'] = w_df.sub_id + '_X'

w_df.loc[10, 'sub_id'] = w_df.sub_id + '_X'



######## level 11 change ##########

splits = w_df.loc[11, 'sub_id'].str.split('_')

w_df.loc[11, 'sub_id'] = (splits.str[3] + '_' + \

                          splits.str[0] + '_' + \

                          splits.str[1] + '_' + \

                          splits.str[2]).values
# Lets observe our work

for i in list(range(1, 6)) + [10]: 

    p(i)

print('Looks good')
# Add '_0.005_validation' to sub_id column

# We will need to do this for every quantile. 

# For now, we are just doing it for 0.005

w_df['sub_id'] = w_df.sub_id + '_0.005_validation'
w_df.head()
# Verify we have all the right labels 

# in our index. This should be the null set.

set(ss.id).difference(set(w_df.sub_id))
# Are they in the right order?

(w_df.index == ss.id).all()
# Lets reindex w_df and see if they match. 

w_df = w_df.set_index('sub_id')

w_df = w_df.reindex(ss.id)

(w_df.index == ss.id).all()
############## sub_id function ################

def add_sub_id(w_df):

    """ adds a column 'sub_id' which will match the 

    labels in the sample_submission 'id' column. Next 

    step will be adding '_{quantile}_validation/evaluation'

    onto the sub_id column. This will be done in another 

    function. 

    

    :w_df: dataframe with the multi-index that is 

    genereated by get_rollup()

    

    Returns w_df with added 'sub_id' column"""

    # Lets add a sub_id col to w_df that 

    # we will build to match the submission file. 

    w_df['sub_id'] = w_df.index.get_level_values(1)



    ###### level 1-5, 10 change ########

    w_df.loc[1:5, 'sub_id'] = w_df.sub_id + '_X'

    w_df.loc[10, 'sub_id'] = w_df.sub_id + '_X'



    ######## level 11 change ##########

    splits = w_df.loc[11, 'sub_id'].str.split('_')

    w_df.loc[11, 'sub_id'] = (splits.str[3] + '_' + \

                              splits.str[0] + '_' + \

                              splits.str[1] + '_' + \

                              splits.str[2]).values

    

    return w_df





############## Adding '_{quantile}_validation' ##############

# Start with a fresh w_df. 

w_df = h.get_w_df(

    train_df,

    cal_df,

    prices_df,

    rollup_index,

    rollup_matrix_csr,

    start_test=1914,

)



# Use tao = 0.005

T = 0.005



# Add 'sub_id' column 

w_df = add_sub_id(w_df)



# Add tao onto sub_id

w_df['sub_id'] = w_df.sub_id + f"_{T}_validation"

w_df.head()
################## add quantile function ################

def add_quantile_to_sub_id(w_df, u): 

    """Used to format 'sub_id' column in w_df. w_df must 

    already have a 'sub_id' column. This used to match 

    the 'id' column of the submission file."""

    # Make sure not to affect global variable if we 

    # don't want to. 

    w_df = w_df.copy()

    w_df['sub_id'] = w_df.sub_id + f"_{u:.3f}_validation"

    return w_df
# Load in a submission we know the score of. 

# This one scored .17921

# We only want the _validaiton portion to test our wspl

sub = pd.read_csv('/kaggle/input/from-point-to-uncertainty-prediction/submission.csv').iloc[:42840 * 9]

sub.head()
# The id column is not in the right order for 

# our wspl function. Lets fix that. 

# First we start with a fresh w_df. Then we add 

# on the 'sub_id' column. Then we extend the dataframe 

# to be 9 times as long with the appropriate 

# "_{quantile}_validation" lable added to 'sub_id'

w_df = h.get_w_df(

    train_df,

    cal_df,

    prices_df,

    rollup_index,

    rollup_matrix_csr,

    start_test=1914,

)



# We need the spl scale

w_df = add_spl_scale(w_df, train_df, rollup_matrix_csr)

w_df = add_sub_id(w_df)
quantiles = [0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995]

copies = [add_quantile_to_sub_id(w_df, quantiles[i]) for i in range(9)]

w_df_9 = pd.concat(copies, axis = 0)
# We need the submission file to be in the 

# the same order as our w_df index to use 

# our function. 

sorted_sub = sub.set_index('id').reindex(w_df_9.sub_id)
# We need the true values

start_test = 1914

level_12_acutals = train_df.loc[:, f'd_{start_test}': f'd_{start_test + 27}']

actuals = rollup_matrix_csr * level_12_acutals.values

actuals = np.tile(actuals.T, 9).T
print(wspl(actuals, sorted_sub.values))

print('BOOM!')
from m5_helpers import *
####################################################################################

############################ WSPL and helpers ######################################



############################ WSPLEvaluator Object ##################################

class WSPLEvaluator(): 

    """ Will generate w_df and ability to score prediction for any start_test period """

    def __init__(self, train_df, cal_df, prices_df, start_test=1914):

        self.rollup_matrix_csr, self.rollup_index = get_rollup(train_df)

                        

        self.w_df = get_w_df(

                        train_df,

                        cal_df,

                        prices_df,

                        self.rollup_index,

                        self.rollup_matrix_csr,

                        start_test=start_test,

                    )

        

        self.quantiles = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]

        level_12_actuals = train_df.loc[:, f'd_{start_test}': f'd_{start_test + 27}']

        self.actuals = self.rollup_matrix_csr * level_12_actuals.values

        self.actuals_tiled = np.tile(self.actuals.T, 9).T

        

        

    def score_all(self, preds): 

        scores_df, total = wspl(self.actuals_tiled, preds, self.w_df)

        self.scores_df = scores_df

        self.total_score = total

        print(f"Total score is {total}")





############## spl scaling factor function ###############

def add_spl_scale(w_df, train_df, rollup_matrix_csr): 

    # We calculate scales for days preceding 

    # the start of the testing/scoring period. 

    start_test = 1914

    df = train_df.loc[:, 'd_1':f'd_{start_test-1}']



    # We will need to aggregate all series 

    agg_series = rollup_matrix_csr * df.values



    # Make sure leading zeros are not included in calculations

    agg_series = h.nan_leading_zeros(agg_series)



    # Now we can compute the scale and add 

    # it as a column to our w_df

    scale = np.nanmean(np.abs(np.diff(agg_series)), axis = 1)

    scale.shape

    w_df['spl_scale'] = scale



    # It may also come in handy to have a scaled_weight 

    # on hand.  

    w_df['spl_scaled_weight'] = w_df.weight / w_df.spl_scale

    

    return w_df



########## Function for all level pinball loss for quantile u ############

def spl_u(actuals, preds, u, w_df):

    """Returns the scaled pinball loss for each series"""

    pl = np.where(actuals >= preds, (actuals - preds) * u, (preds - actuals) * (1 - u)).mean(axis=1)



    # Now calculate the scaled pinball loss.  

    all_series_spl = pl / w_df.spl_scale

    return all_series_spl



########## wspl for all quantiles ############

def wspl(actuals, preds, w_df): 

    """

    :acutals:, 9 vertical copies of the ground truth for all series. 

    :preds: predictions for all series and all quantiles. Same 

    shape as actuals"""

    quantiles = [0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995]

    scores = []

    

    # In this case, preds has every series for every  

    # quantile T, so it has 42840 * 9 rows. We first 

    # break it up into 9 parts to get the wspl_T for each.

    # We also do the same for actuals. 

    preds_list = np.split(preds, 9)

    actuals_list = np.split(actuals, 9)

    

    for i in range(9):

        scores.append(spl_u(actuals_list[i], preds_list[i], quantiles[i], w_df))

        

    # Store all our results in a dataframe

    scores_df = pd.DataFrame(dict(zip(quantiles, [w_df.weight * score for score in scores])))

    

    #  We divide score by 9 

    # to get the average wspl of each quantile. 

    spl = sum(scores) / 9

    wspl_by_series = (w_df.weight * spl)

    total = wspl_by_series.sum() / 12

    

    return scores_df, total



####################################################################################

############################ formatting for submission #############################



############## sub_id function ################

def add_sub_id(w_df):

    """ adds a column 'sub_id' which will match the 

    labels in the sample_submission 'id' column. Next 

    step will be adding '_{quantile}_validation/evaluation'

    onto the sub_id column. This will be done in another 

    function. 

    

    :w_df: dataframe with the multi-index that is 

    genereated by get_rollup()

    

    Returns w_df with added 'sub_id' column"""

    # Lets add a sub_id col to w_df that 

    # we will build to match the submission file. 

    w_df['sub_id'] = w_df.index.get_level_values(1)



    ###### level 1-5, 10 change ########

    w_df.loc[1:5, 'sub_id'] = w_df.sub_id + '_X'

    w_df.loc[10, 'sub_id'] = w_df.sub_id + '_X'



    ######## level 11 change ##########

    splits = w_df.loc[11, 'sub_id'].str.split('_')

    w_df.loc[11, 'sub_id'] = (splits.str[3] + '_' + \

                              splits.str[0] + '_' + \

                              splits.str[1] + '_' + \

                              splits.str[2]).values

    

    return w_df







################## add quantile function ################

def add_quantile_to_sub_id(w_df, u): 

    """Used to format 'sub_id' column in w_df. w_df must 

    already have a 'sub_id' column. This used to match 

    the 'id' column of the submission file."""

    # Make sure not to affect global variable if we 

    # don't want to. 

    w_df = w_df.copy()

    w_df['sub_id'] = w_df.sub_id + f"_{u:.3f}_validation"

    return w_df
# Must add m5_helpers utility script

from m5_helpers import WSPLEvaluator
%%time

e = WSPLEvaluator(train_df, cal_df, prices_df, start_test=1914)
e.score_all(sorted_sub.values)
# Now we can get the total score 

e.total_score
# We can also see scores for all series 

# and all quantiles

# with a multi-index for organization. 

scores_df = e.scores_df

scores_df
# We can look at each level and quantile 

# combination. 

scores_df.groupby(level=0).sum()
# We can also see scores by levels

scores_df.groupby(level=0).sum().mean(axis=1)
# Or scores by quantile

scores_df.groupby(level=0).sum().mean(axis=0)
scores_df.loc[3]


scores_df.loc[3].plot(title='scores by store for each quantile')
scores_df.loc[3].T.plot(title='scores for quantile for each store', figsize=(14,8))
scores_df.loc[1].T.plot()