# Some key parameters/variables to control the notebook operation

LOCATION_KAGGLE = True

SHOW_EDA = True



# used to label any output files

version_str = "v32"



# Can write up to 5GB to the current directory (/kaggle/working/)

out_dir = "."
# Usual suspects to use

import numpy as np # linear algebra

from numpy import random

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



import matplotlib.pyplot as plt



# t-SNE

from sklearn.manifold import TSNE



from time import time



import os
# The seed is set once here at beginning of notebook.

RANDOM_SEED = 360

# Uncomment this to get a time-based random value, 0 to 1023

##RANDOM_SEED = int(time()) % 2**10

# in either case initialize the seed

np.random.seed(RANDOM_SEED)
# Use the y_yhat_plots() routine to shown how the prediction is doing.

# This routine is taken from the file chirp_roc_lib.py in the github repo at: 

#   https://github.com/dan3dewey/chirp-to-ROC

# Some small modifications have been made here.

import roc_plots

from roc_plots import *
# In case changes are made and we want to reload it while the kernel is running:

import importlib

importlib.reload(roc_plots)

from roc_plots import *
# Show available data files (Kaggle provided code):

for dirname, _, filenames in os.walk('../input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Where are the data files

# Data dir

if LOCATION_KAGGLE:

    dat_dir ='../input/lish-moa/'

    # CSV file names - features and targets are separate

    train_feats = "train_features.csv"

    train_targs = "train_targets_scored.csv"

    test_feats = "test_features.csv"

    test_targs = "sample_submission.csv"

else:

    dat_dir ="../input/"

    # CSV file names - features and targets are separate

    train_feats = "train_features.csv.zip"

    train_targs = "train_targets_scored.csv.zip"

    test_feats = "test_features.csv.zip"

    test_targs = "sample_submission.csv.zip"
# Read the files and do All the basic feature processing

# time it

t_preproc = time()



# Read in the train and test data





# = = = = =

# Train

df_train_feats = pd.read_csv(dat_dir+train_feats)

df_train_targs = pd.read_csv(dat_dir+train_targs)





# = = = = =

# Test

df_test_feats = pd.read_csv(dat_dir+test_feats)

df_test_targs = pd.read_csv(dat_dir+test_targs)



print("{:.2f} seconds -- read in data files\n".format(time() - t_preproc))

 



# Separate out the Controls

#

# cp_type feature has only 2 values: 'trt_cp' 'ctl_vehicle'

#   trt_cp      means a treatment is applied

#   ctl_vehicle means it was a control

# There are 1866 controls in Train, and 358 in Test.

train_ctls = df_train_feats.cp_type == 'ctl_vehicle'

test_ctls = df_test_feats.cp_type == 'ctl_vehicle'

#

# Get dfs of just the control features:

df_train_ctls = df_train_feats[train_ctls].copy()

df_test_ctls = df_test_feats[test_ctls].copy()

#

# Check the sum of all the train-control Targets - should be 0.

print("Train control targets sum:",

      sum(df_train_targs[train_ctls].drop(columns=['sig_id']).sum()))

#

# Create Treatment-only dfs of feats and targs:

df_treat_feats = df_train_feats[~train_ctls].copy()

df_treat_targs = df_train_targs[~train_ctls].copy()

#

# For now, don't need to make treatment-only df for Test:

# We have the test controls, df_test_ctls, to compare with train values.

# Probably do not need separate Test-treament df,

# since we'll just make predictions on all Test ids and

# then set the Test-control targets to 0.





# Check the sum of all the treatment targets - should be 16844.

print("Treatment targets sum:",

      sum(df_treat_targs.drop(columns=['sig_id']).sum()))





# Create some other columns, etc.

# Normally additional features would be added to the df train (or df treat, here.)

# To keep the df_treat_feats 'clean' for easy stat.s,

# instead make a df_aug_feats from treat and add other columns to it:

df_aug_feats = df_treat_feats.copy()





# All Targets --> numMoA

# --- Add a numMoA column (number of MoA set in each sig_id row) ---

# Of course this leaks target information, so don't use for prediction ;-)

# Instead, make a binary target from numMoA, e.g., y=1 when numMoA > 0

#

# Use this==0 as the target for "MoA or not-MoA" ML:

df_aug_feats['numMoA'] = df_treat_targs.drop(['sig_id'],axis=1).sum(axis=1)





# Subset of Targets --> numSub

# --- Add a numSub column (similar to numMoA but for a SUBSET of the targets.)

#

# These 9 targets were identified as "islands" in the t-SNE output, 

# so expect they can be well 'learned':

##targ_subset = ['proteasome_inhibitor', 'nfkb_inhibitor', 'glucocorticoid_receptor_agonist',

##               'raf_inhibitor', 'cdk_inhibitor', 'hmgcr_inhibitor', 

##               'egfr_inhibitor', 'hsp_inhibitor', 'tubulin_inhibitor']

##targ_subset_name = "9: t-SNE islands"

##df_aug_feats['numSub'] = df_treat_targs[targ_subset].sum(axis=1)

#

# These 2 targets are detectable and mostly occur together,

# numSub>0 is the OR of them, sumSub>1 is the AND.

##targ_subset = ['proteasome_inhibitor', 'nfkb_inhibitor']

##targ_subset_name = "2: proteasome & nfkb"

##df_aug_feats['numSub'] = df_treat_targs[targ_subset].sum(axis=1)

#

# The tSNE-9 without the "big 2":

targ_subset = ['glucocorticoid_receptor_agonist',

               'raf_inhibitor', 'cdk_inhibitor', 'hmgcr_inhibitor', 

               'egfr_inhibitor', 'hsp_inhibitor', 'tubulin_inhibitor']

targ_subset_name = "7: t-SNE-9 w/o big 2"

df_aug_feats['numSub'] = df_treat_targs[targ_subset].sum(axis=1)

#

# Try these 4 together:

##targ_subset = ['glucocorticoid_receptor_agonist',

##               'raf_inhibitor', 'cdk_inhibitor', 'hmgcr_inhibitor']

##targ_subset_name = "4: low t-SNE rms"

##df_aug_feats['numSub'] = df_treat_targs[targ_subset].sum(axis=1)

#

# All the targets with a > 0.01

#    *except* for 'proteasome_inhibitor', 'nfkb_inhibitor' because they are unique -

#             they are most common and appear together usually.

##targ_subset = ['acetylcholine_receptor_antagonist', 'adrenergic_receptor_agonist',

##               'adrenergic_receptor_antagonist', 'calcium_channel_blocker',

##               'cdk_inhibitor', 'cyclooxygenase_inhibitor',

##               'dna_inhibitor', 'dopamine_receptor_antagonist',

##               'egfr_inhibitor', 'flt3_inhibitor',

##               'glucocorticoid_receptor_agonist', 'glutamate_receptor_antagonist',

##               'histamine_receptor_antagonist', 'hmgcr_inhibitor',

##               'kit_inhibitor',

##               'pdgfr_inhibitor', 'phosphodiesterase_inhibitor',

##               'raf_inhibitor',

##               'serotonin_receptor_agonist', 'serotonin_receptor_antagonist',

##               'sodium_channel_inhibitor', 'tubulin_inhibitor']

##targ_subset_name = "22: a above 0.01"

##df_aug_feats['numSub'] = df_treat_targs[targ_subset].sum(axis=1)

#

# All above 0.010 except for the tSNE-9:     * Not very detectable *

##targ_subset = ['acetylcholine_receptor_antagonist', 'adrenergic_receptor_agonist',

##               'adrenergic_receptor_antagonist', 'calcium_channel_blocker', 'cyclooxygenase_inhibitor',

##               'dna_inhibitor', 'dopamine_receptor_antagonist', 'flt3_inhibitor',

##               'glutamate_receptor_antagonist', 'histamine_receptor_antagonist', 'kit_inhibitor',

##               'pdgfr_inhibitor', 'phosphodiesterase_inhibitor', 'serotonin_receptor_agonist',

##               'serotonin_receptor_antagonist', 'sodium_channel_inhibitor']

##targ_subset_name = "16: >0.010, w/o tSNE-9"

##df_aug_feats['numSub'] = df_treat_targs[targ_subset].sum(axis=1)

#

# These are the targets with 'highly detectable' g-vectors (unless listed in tSNE-9 already)

#    'topoisomerase_inhibitor', 'hdac_inhibitor', 'mtor_inhibitor', 'mek_inhibitor',

#    'pi3k_inhibitor', 'protein_synthesis_inhibitor', 'atpase_inhibitor'

##targ_subset = ['topoisomerase_inhibitor', 'hdac_inhibitor', 'mtor_inhibitor', 'mek_inhibitor',

##        'pi3k_inhibitor', 'protein_synthesis_inhibitor', 'atpase_inhibitor']

##targ_subset_name = "7: detectable"

##df_aug_feats['numSub'] = df_treat_targs[targ_subset].sum(axis=1)

#

# OR

#

# A single target ;-)     These lines are repeated below in Machine Learning to

#                         easily switch between single targets.

##targ_subset = ['histamine_receptor_antagonist']

##targ_subset_name = "1: hista"

##df_aug_feats['numSub'] = df_treat_targs[targ_subset].sum(axis=1)

#



# --- The average and std of the c-0 to c-99 features ---

print("\n{:.2f} s -- Adding Train c-ave, c-std...".format(time() - t_preproc))



# TRAIN: Add the mean and std over the c values, for each id:

ccols = list(range(776, 776+100))

n_treats = len(df_treat_feats)

c_aves = np.zeros(n_treats)

c_stds = np.zeros(n_treats)

# 5th and 95th percentile values:

c_5pc = np.zeros(n_treats)

c_95pc = np.zeros(n_treats)

n_cs = len(ccols); i5pc = int(0.05*n_cs) ; i95pc = int(0.95*n_cs)

for irow in range(n_treats):

    these_cs = df_treat_feats.iloc[irow, ccols].values

    these_cs.sort()

    

    c_aves[irow] = these_cs.mean()

    c_stds[irow] = these_cs.std()

    c_5pc[irow] = these_cs[i5pc]

    c_95pc[irow] = these_cs[i95pc]

# and put them in the augmented df:

df_aug_feats['c-ave'] = c_aves

df_aug_feats['c-std'] = c_stds

df_aug_feats['c-5%'] = c_5pc

df_aug_feats['c-95%'] = c_95pc



print("\n{:.2f} s -- Adding Test c-ave, c-std...".format(time() - t_preproc))



# TEST: generate these and add them to the df_test_feats:

ccols = list(range(776, 776+100))

n_treats = len(df_test_feats)

c_aves = np.zeros(n_treats)

c_stds = np.zeros(n_treats)

# 5th and 95th percentile values:

c_5pc = np.zeros(n_treats)

c_95pc = np.zeros(n_treats)

n_cs = len(ccols); i5pc = int(0.05*n_cs) ; i95pc = int(0.95*n_cs)

for irow in range(n_treats):

    these_cs = df_test_feats.iloc[irow, ccols].values

    these_cs.sort()

    

    c_aves[irow] = these_cs.mean()

    c_stds[irow] = these_cs.std()

    c_5pc[irow] = these_cs[i5pc]

    c_95pc[irow] = these_cs[i95pc]

# and put them in the test df:

df_test_feats['c-ave'] = c_aves

df_test_feats['c-std'] = c_stds

df_test_feats['c-5%'] = c_5pc

df_test_feats['c-95%'] = c_95pc





# --- The average and std of the g-0 to g-771 features ---

print("\n{:.2f} s -- Adding Train g-ave, g-std...".format(time() - t_preproc))



# TRAIN: Add the mean and std over the *** g values ***, for each id:

gcols = list(range(4, 4+772))

n_treats = len(df_treat_feats)

g_aves = np.zeros(n_treats)

g_stds = np.zeros(n_treats)

# fractions above 2 and below -2:

g_hif = np.zeros(n_treats)

g_lof = np.zeros(n_treats)

# 5th and 95th percentile values:

g_5pc = np.zeros(n_treats)

g_95pc = np.zeros(n_treats)

n_gs = len(gcols); i5pc = int(0.05*n_gs) ; i95pc = int(0.95*n_gs)

for irow in range(n_treats):

    these_gs = df_treat_feats.iloc[irow, gcols].values

    these_gs.sort()



    g_aves[irow] = these_gs.mean()

    g_stds[irow] = these_gs.std()

    g_hif[irow] = sum(1.0*(these_gs > 2.0))/n_gs

    g_lof[irow] = sum(1.0*(these_gs < -2.0))/n_gs

    g_5pc[irow] = these_gs[i5pc]

    g_95pc[irow] = these_gs[i95pc]

# and put them in the augmented df:

df_aug_feats['g-ave'] = g_aves

df_aug_feats['g-std'] = g_stds

df_aug_feats['g-hif'] = g_hif

df_aug_feats['g-lof'] = g_lof

df_aug_feats['g-hilof'] = g_hif/(g_lof + 1.3e-3)  # use hilopc instead

df_aug_feats['g-5%'] = g_5pc

df_aug_feats['g-95%'] = g_95pc

df_aug_feats['g-hilopc'] = g_95pc/(g_5pc + 1.3e-3)



print("\n{:.2f} s -- Adding Test g-ave, g-std...".format(time() - t_preproc))



# TEST: Add the mean and std over the *** g values ***, for each id:

gcols = list(range(4, 4+772))

n_treats = len(df_test_feats)   # repurposing n_treats ;-)

g_aves = np.zeros(n_treats)

g_stds = np.zeros(n_treats)

# fractions above 2 and below -2:

g_hif = np.zeros(n_treats)

g_lof = np.zeros(n_treats)

# 5th and 95th percentile values:

g_5pc = np.zeros(n_treats)

g_95pc = np.zeros(n_treats)

n_gs = len(gcols); i5pc = int(0.05*n_gs) ; i95pc = int(0.95*n_gs)

for irow in range(n_treats):

    these_gs = df_test_feats.iloc[irow, gcols].values

    these_gs.sort()

    g_aves[irow] = these_gs.mean()

    g_stds[irow] = these_gs.std()

    g_hif[irow] = sum(1.0*(these_gs > 2.0))/n_gs

    g_lof[irow] = sum(1.0*(these_gs < -2.0))/n_gs

    g_5pc[irow] = these_gs[i5pc]

    g_95pc[irow] = these_gs[i95pc]

# and put them in the test df:

df_test_feats['g-ave'] = g_aves

df_test_feats['g-std'] = g_stds

df_test_feats['g-hif'] = g_hif

df_test_feats['g-lof'] = g_lof

df_test_feats['g-hilof'] = g_hif/(g_lof + 1.3e-3)  # use hilopc instead

df_test_feats['g-5%'] = g_5pc

df_test_feats['g-95%'] = g_95pc

df_test_feats['g-hilopc'] = g_95pc/(g_5pc + 1.3e-3)



print("\n{:.2f} seconds -- added basic new feature columns".format(time() - t_preproc))
# Check for NaN's in the data

# Go through the columns one at a time



# Do it for feats and targs:

# feats: (train, test)

#   All OK - no NaNs found.

# targs: (train only)

#   All OK - no NaNs found.



if False:

    n_train = len(df_train_feats)

    n_test = len(df_test_feats)

    print("\nChecking for NaNs:\n")

    all_ok = True

    #

    nanpc_train = 0; nanpc_test = 0

    #  do it for feats  and  targs

    for col in df_train_targs.columns:

        nona_train = len(df_train_targs[col].dropna(axis=0))

        nanpc_train = 100.0*(n_train-nona_train)/n_train

        ##nona_test = len(df_test_feats[col].dropna(axis=0))

        ##nanpc_test = 100.0*(n_test-nona_test)/n_test

        # Only show it if there are NaNs:

        if (nanpc_train + nanpc_test > 0.0):

            print("{:.3f}%  {} OK out of {}".format(nanpc_train, nona_train, n_train), "  "+col)

            ##print("{:.3f}%  {} OK out of {}".format(nanpc_test, nona_test, n_test), "  "+col)

            all_ok = False

    if all_ok:

        print("   All OK - no NaNs found.\n")

# Use just the treatment rows:

df_treat_targs
# Can (re-)do the following analyses with a downselected set of ids,

# with the down selection based on feature/target criteria.



# This will select all  21948 treatment sig_ids

select = df_treat_feats.cp_time > 0

select_str = "All treatment ids"





# No obvious difference when selecting on cp_time or cp_dose.



# Select only the longest time  7180 rows

##select = df_treat_feats.cp_time == 72

##select_str = "All treatment ids w/ t=72"



# Select only the shortest time   7166 rows

##select = df_treat_feats.cp_time == 24

##select_str = "All treatment ids w/ t=24"



# Select only the LOW dose, D2   10752 rows

##select = df_treat_feats.cp_dose == "D2"

##select_str = "All treatment ids w/ D2"



# Select only the HIGH dose, D1  11196  rows

##select = df_treat_feats.cp_dose == "D1"

##select_str = "All treatment ids w/ D1"





# Can select based on the numMoA value using df_aug_feats - not sure what it means ;-)

##select = df_aug_feats.numMoA <= 1

##select_str = "Treatment ids with numMoA = 0 or 1"



##select = df_aug_feats.numMoA > 1

##select_str = "Treatment ids with numMoA >= 2"
# What fraction of ids are active for each target?

n_rows = len(df_treat_targs[select])



# Calculate the sum in each column:

targ_col_sums = df_treat_targs[select].drop(['sig_id'],axis=1).sum(axis=0)

n_targs = len(targ_col_sums)

print("Number of targets:",n_targs,"   Sum of all:",sum(targ_col_sums))



plt.hist(np.array(targ_col_sums),bins=100)

plt.title("Histogram of the target (col) sums")

plt.xlabel("Sum of the column = number of active ids for target")

plt.ylabel("Number of targets")

plt.show()
# Average active value of each target

aves_targs = targ_col_sums/n_rows



# List the highest ones

print(aves_targs.sort_values(ascending=False)[0:30])

print("\nStarting from the lowest:\n")

print(aves_targs.sort_values(ascending=True)[0:10])
# All targets with a > 0.01:

a_above_010 = list(aves_targs[aves_targs > 0.010].index)



print(a_above_010)
# Plot all 206 average active values, sorted

plt.plot(np.array(aves_targs.sort_values(ascending=True)))

plt.title(select_str)

plt.ylabel("$a$ value for target")

plt.xlabel("Targets (sorted)")

plt.show()
# How many of the 206 Targets are set, aka active, in each id ?



# Calculate the sum in each row:

targ_row_sums = df_treat_targs[select].drop(['sig_id'],axis=1).sum(axis=1)

print("Number of rows:",n_rows,"   Sum of all:",sum(targ_row_sums))



# Note: this same calculation was used for the numMoA augmented feature;

# so, this line would make the same plot:

#  plt.hist(np.array(df_aug_feats.numMoA.values),bins=50)



plt.hist(np.array(targ_row_sums),bins=50)

plt.title("Histogram of the id (row) sums")

plt.xlabel("Sum of the row = number of active targets")

plt.ylabel("Number of ids")

plt.show()
# Get the numbers of ids that have 0, 1, 2, 3+ active targets:

num_ids_vs_active = np.histogram(np.array(targ_row_sums),

                                 bins=[-0.5,0.5,1.5,2.5,9.5])[0]

# The fraction of rows with 0, 1, 2, 3+ active targets in them:

frac_vs_active = num_ids_vs_active / n_rows

for inum, counts in enumerate(num_ids_vs_active):

    print(inum, counts, frac_vs_active[inum])
# Are the target active fractions roughly independent?

# Assuming independence we can calculate the expected fraction vs active number.



# Probabilty of getting 0 of the targets active:

prob0 = 1.0

for this_ave in aves_targs:

    prob0 = prob0 * (1 - this_ave)

print("0 active:   Actual frac = ",frac_vs_active[0],

        "   Expected frac = ",prob0)

# Probability of exactly 1 active:

prob1 = 0.0

# add prob of each one being active and all the others not:

for this_ave in aves_targs:

    prob1 = prob1 + this_ave * prob0/(1-this_ave)

print("1 active:   Actual frac = ",frac_vs_active[1],

        "   Expected frac = ",prob1)

# Probability of 2 active (slight approximation?)

prob2 = 0.0

for this_ave in aves_targs:

    prob2 = prob2 + this_ave * (prob1 - this_ave * prob0/(1-this_ave))

# divide by 2 since double counted:

prob2 = prob2 / 2

print("2 active:   Actual frac = ",frac_vs_active[2],

        "   Expected frac = ",prob2)



prob3etc = 1.0 - prob0 - prob1 - prob2

print("3+ active:   Actual frac = ",frac_vs_active[3],

        "   Expected frac = ",prob3etc)
# Get all the rows that have 2 MoAs set

df_just_targs = df_treat_targs.drop(['sig_id'],axis=1)

df_two_moas = df_just_targs[df_aug_feats['numMoA'] == 2]



# array of values: 1,2,3,4,...,206

one_to_206 = 1 + np.array(list(range(206)))



all_pairids = []

for irow in range(len(df_two_moas)):

    two_moas = np.sort(df_two_moas.iloc[irow].values * one_to_206)[[-2,-1]]

    i_combined = 1000*two_moas[0] + two_moas[1]

    all_pairids.append(i_combined)

ser_pairids = pd.Series(all_pairids)
pair_counts = ser_pairids.value_counts()

print("Unique number of pairs:",len(pair_counts))



pair_counts[0:15]
# Each of the target numbers in the combined pair id are from 1 to 206.

# To get their names subtract 1:

print(df_just_targs.columns[137-1], ",", df_just_targs.columns[164-1])
# Calculate the individual scores_targs from the a values

scores_targs = 0.0*aves_targs

for itarg in range(len(aves_targs)):

    this_ave = max(aves_targs[itarg], 1e-5)

    scores_targs[itarg]= -1.0*(this_ave*np.log(this_ave) +

                               (1-this_ave)*np.log(1-this_ave))



# Plot all 206 average score values, sorted

plt.plot(np.array(scores_targs.sort_values(ascending=True)))

plt.title(select_str)

plt.ylabel("ave score for target")

plt.xlabel("Targets (sorted)")

plt.show()



# Calculate the average of those over all the aves_targs (= a) values:

print("Expect a score around", sum(scores_targs)/len(aves_targs))

# If the control ids are included with target=0 then the score decreases a bit:

expect_score = (len(df_treat_targs)/len(df_train_targs))*sum(scores_targs)/len(aves_targs)

print("Expected score, corrected for controls, is",expect_score)
scores_targs.sort_values(ascending=False)[0:15]
# It's tempting to think that the more present MoAs (the higher a = aves_targs values)

# might be easier to 'learn' and make accurate predictions for.

# What score would we get if

# i) we just "guessed", e.g., used aves_targs values for the lower k targets, and

# ii) we had 100% accuracy for all targets above k.



# We can calculate and plot that vs the cutoff k value:

sorted_scores = np.array(scores_targs.sort_values(ascending=True))

cumave_scores = 0.0*sorted_scores

score_sum = 0.0

for isc, this_sc in enumerate(sorted_scores):

    score_sum += this_sc

    cumave_scores[isc] = score_sum/len(cumave_scores)



# Plot expected score vs the a cutoff value

plt.plot(cumave_scores)

plt.title(select_str)

plt.ylabel("Score if all higher targets are known")

plt.xlabel("k, number of lower targets guessed")

plt.show()



# The plot and the array cumave_scores[k-1] has:

# score of 0 when k=0: that's when we know all the targets values.

# score of ~ 0.22 when k=206: that's when we are guessing all target values.

#                             (before applying the treatment vs total correction.)

# score of 0.0100 when k=167: that's when we are guessing targets 1 through 167

#                             and know exactly the 39 targets from 168 to 206

#                             (these target numbers are as sorted in increasing a order.)
df_test_feats[~test_ctls]
## Look at the df_treat_feats with the additional features added

df_aug_feats
# The features in df_treat_feats.columns are:

#iloc

#   0  sig_id  - anonymized values (I believe)

#   1  cp_type - determines controls

#   2  cp_time - 3 values: 24, 48, 72

#   3  cp_dose - 2 values D1, D2

#

#   4  g-0  to -  ~ Gaussian-with-outliers, 772 of them in all

# 775  g-771

#

# 776  c-0  to - ?, 100 of them in all

# 875  c-99



# Look at the values using iloc[ irow, icol ]



# Pick a single feature (COLUMN), show the feature value for a bunch of the ids' values

icol = 800+10

col_str = df_treat_feats.columns[icol]

# Show some controls and treatment ones

plt.plot(df_treat_feats.iloc[1000:1500, icol].values,'r.')

plt.plot(df_train_ctls.iloc[0:500, icol].values,'y.',alpha=0.5)



plt.title(col_str+" : Treated (red) and  Control (yellow)")

plt.ylabel("Value of the feature")

plt.xlabel("A bunch of sig_ids")

plt.show()





# The features in df_treat_feats.columns are:

#iloc

#   0  sig_id  - anonymized values (I believe)

#   1  cp_type - determines controls

#   2  cp_time - 3 values: 24, 48, 72

#   3  cp_dose - 2 values D1, D2

#

#   4  g-0  to -  ~ Gaussian-with-outliers, 772 of them in all

# 775  g-771

#

# 776  c-0  to - ?, 100 of them in all

# 875  c-99



# Look at the values using iloc[ irow, icol ]



# Single sig_id (ROW), show a bunch of the target values



# Looking at the c values for a single id

# Wow: check out irow = 1199 !

#      less dramatic but cool: 1197

# --> The c values do all seem to move together...



# Pick a single sig_id

irow = 1197



# Look at all 100 of the c features for that sig_id

icol = 776   # c starts here

col_str = df_treat_feats.columns[icol]

icols = list(range(icol, icol+100))

# Show some controls and treatment ones

plt.plot(df_treat_feats.iloc[irow, icols].values,'r.')

plt.plot(df_train_ctls.iloc[irow, icols].values,'y.',alpha=0.5)



plt.title(df_treat_feats.iloc[irow,0]+" : Treated (red) and  Control (yellow)")

plt.ylabel("Value of the feature")

plt.xlabel(col_str+" and following target values")

plt.show()
# The features c-ave and c-std were added to the augmented df

# Plot c-std vs c-ave.

# Choose how to color the points:

#  by numMoA -- looks useful to help determine the MoA

colors = 1.0*np.array(df_aug_feats.numMoA > 0) + 1.0*np.array(df_aug_feats.numMoA > 1)

#  by cp_dose -- no obvious pattern, just intermixed.

##colors = 1.0*np.array(df_aug_feats.cp_dose == 'D1')

#  by cp_time -- clear pattern: higher times are further along the swoosh...

##colors = 1.0*np.array(df_aug_feats.cp_time)

#

df_aug_feats.plot(x='c-ave',y='c-std',kind='scatter', figsize=(12,7),

                 c=colors, colormap='jet', alpha=0.25, marker='o',s=50,

                 title='C-std vs C-ave for all non-control sig_ids'+

                  '   Colored by number of MoAs (0, 1, 2+)')

plt.savefig("C-std_vs_C-ave_MoA-color_"+version_str+".png")

plt.show()



# This is very cool! Looks like c-ave and c-std have some ability to

# help decide if numMoA is 0, 1, 2+.



#  * * *  Note that there are many ids with MoA=1

#         that overlap with the "control blob" at (0,0.5).  * * *







#  by cp_time -- clear pattern: higher times are further along the swoosh...

colors = 1.0*np.array(df_aug_feats.cp_time)

#

df_aug_feats.plot(x='c-ave',y='c-std',kind='scatter', figsize=(12,7),

                 c=colors, colormap='jet', alpha=0.25, marker='o',s=50,

                 title='C-std vs C-ave for all non-control sig_ids'+

                  '   Colored by cp_time (24, 48, 72)')

plt.savefig("C-std_vs_C-ave_time-color_"+version_str+".png")

plt.show()

#  - - C CONTROLS - -

# Take a look at the mean and std over the c values of the Train controls -

# The controls look very different: 

# They are concentrated at 0.0,0.5 blob, don't have the parabola swoosh band,

# and do have a hint of the straight not-MoA line.

if True:

    ccols = list(range(776, 776+100))

    n_treats = len(df_train_ctls)

    c_aves = np.zeros(n_treats)

    c_stds = np.zeros(n_treats)

    for irow in range(n_treats):

        c_aves[irow] = df_train_ctls.iloc[irow, ccols].values.mean()

        c_stds[irow] = df_train_ctls.iloc[irow, ccols].values.std()

    

    ##lt.plot(c_aves,'b.')

    ##plt.show()



    ##plt.plot(c_stds,'b.')

    ##plt.show()



    plt.plot(c_aves,c_stds,'b.')

    plt.title("Very different from the Treatment ids")

    plt.ylabel("c-std for Controls")

    plt.ylabel("c-ave for Controls")

    plt.show()
# TEST: Plot c-std vs c-ave -- no MoA colors, but can show controls.



# 0 - control, 1 - not control

colors = 1.4 * 1.0*np.array(df_test_feats.cp_type != 'ctl_vehicle')

colors[0]=2.0



df_test_feats.plot(x='c-ave',y='c-std',kind='scatter', figsize=(12,7),

                 c=colors, colormap='jet', alpha=0.15, marker='o',s=20,

                 title='C-std vs C-ave for all TEST sig_ids'+

                  '   Colored by Control (blue) or not-control (orange)')

plt.show()





# The test values look similarly distributed...
# The features c-5% and c-95% were added to the augmented df

# Plot c-5% vs c-95%.

# Choose how to color the points:

#  by numMoA -- looks useful to help determine the MoA

colors = 1.0*np.array(df_aug_feats.numMoA > 0) + 1.0*np.array(df_aug_feats.numMoA > 1)

#  by cp_dose -- no obvious pattern, just intermixed.

##colors = 1.0*np.array(df_aug_feats.cp_dose == 'D1')

#  by cp_time -- clear pattern: higher times are further along the swoosh...

##colors = 1.0*np.array(df_aug_feats.cp_time)

#

df_aug_feats.plot(x='c-95%',y='c-5%',kind='scatter', figsize=(12,7),

                 c=colors, colormap='jet', alpha=0.25, marker='o',s=50,

                 title='C-5% vs C-95% for all non-control sig_ids'+

                  '   Colored by number of MoAs (0, 1, 2+)')

plt.show()



#  by cp_time -- clear pattern: higher times are further along the swoosh...

colors = 1.0*np.array(df_aug_feats.cp_time)

#

df_aug_feats.plot(x='c-95%',y='c-5%',kind='scatter', figsize=(12,7),

                 c=colors, colormap='jet', alpha=0.25, marker='o',s=50,

                 title='C-5% vs C-95% for all non-control sig_ids'+

                  '   Colored by number of MoAs (0, 1, 2+)')

plt.show()
# The features g-ave and g-std were added to the augmented df

# Plot g-std vs c-ave.  * used C-ave <-- g-ave not very useful, mostlys ~ 0.

# Choose how to color the points:

#  by numMoA -- looks useful to help determine the MoA

colors = 1.0*np.array(df_aug_feats.numMoA > 0) + 1.0*np.array(df_aug_feats.numMoA > 1)

#  by cp_dose -- no obvious pattern, just intermixed.

##colors = 1.0*np.array(df_aug_feats.cp_dose == 'D1')

#  by cp_time -- clear pattern when C-ave is used on x axis.

##colors = 1.0*np.array(df_aug_feats.cp_time)

#

df_aug_feats.plot(x='c-ave',y='g-std',kind='scatter', figsize=(12,7),

                 c=colors, colormap='jet', alpha=0.25, marker='o',s=50,

                 title='g-std vs C-ave for all non-control sig_ids'+

                  '   Colored by number of MoAs (0, 1, 2+)')

plt.show()



# As kind of expected: the g-ave tends to remain around 0,

# so,use C-ave instead of g-ave for the plot.

# The g-std is larger especially for numMoA = 2+ ids





#  by cp_time -- clear pattern when C-ave is used on x axis.

colors = 1.0*np.array(df_aug_feats.cp_time)

#

df_aug_feats.plot(x='c-ave',y='g-std',kind='scatter', figsize=(12,7),

                 c=colors, colormap='jet', alpha=0.25, marker='o',s=50,

                 title='g-std vs C-ave for all non-control sig_ids'+

                  '   Colored by cp_time (24, 48, 72)')

plt.show()
# TEST: Plot g-std vs C-ave -- but no MoA colors.



# 0 - control, 1 - not control

colors = 1.4 * 1.0*np.array(df_test_feats.cp_type != 'ctl_vehicle')

colors[0]=2.0



df_test_feats.plot(x='c-ave',y='g-std',kind='scatter', figsize=(12,7),

                 c=colors, colormap='jet', alpha=0.25, marker='o',s=20,

                 title='g-std vs C-ave for all TEST sig_ids'+

                  '   Colored by Control (blue) or not-control (orange)')



plt.show()



# The test values look similarly distributed...
# Look at C-std vs g-std

#

# Choose how to color the points:

#  by numMoA -- looks useful to help determine the MoA

colors = 1.0*np.array(df_aug_feats.numMoA > 0) + 1.0*np.array(df_aug_feats.numMoA > 1)

#  by cp_dose -- no obvious pattern, just intermixed.

##colors = 1.0*np.array(df_aug_feats.cp_dose == 'D1')

#  by cp_time -- clear patterns in plot

##colors = 1.0*np.array(df_aug_feats.cp_time)

#

df_aug_feats.plot(x='g-std',y='c-std',kind='scatter', figsize=(12,7),

                 c=colors, colormap='jet', alpha=0.25, marker='o',s=50,

                 title='C-std vs g-std for all non-control sig_ids'+

                  '   Colored by number of MoAs (0, 1, 2+)')

plt.show()







#  by cp_time -- clear patterns in plot

colors = 1.0*np.array(df_aug_feats.cp_time)

#

df_aug_feats.plot(x='g-std',y='c-std',kind='scatter', figsize=(12,7),

                 c=colors, colormap='jet', alpha=0.25, marker='o',s=50,

                 title='C-std vs g-std for all non-control sig_ids'+

                  '   Colored by cp_time (24, 48, 72)')

plt.show()
# Look at g-lof vs g-hif

#

# Choose how to color the points:

#  by numMoA -- looks useful to help determine the MoA

colors = 1.0*np.array(df_aug_feats.numMoA > 0) + 1.0*np.array(df_aug_feats.numMoA > 1)

#  by cp_dose -- no obvious pattern, just intermixed.

##colors = 1.0*np.array(df_aug_feats.cp_dose == 'D1')

#  by cp_time -- clear patterns in plot

##colors = 1.0*np.array(df_aug_feats.cp_time)

#

df_aug_feats.plot(x='g-hif',y='g-lof',kind='scatter', figsize=(12,7),

                 c=colors, colormap='jet', alpha=0.25, marker='o',s=50,

                 title='g-lof vs g-hif for all non-control sig_ids'+

                  '   Colored by number of MoAs (0, 1, 2+)')

plt.show()



# These basically track each other, mostly.
# Look at g-5% vs g-95%

#

# Choose how to color the points:

#  by numMoA -- looks useful to help determine the MoA

colors = 1.0*np.array(df_aug_feats.numMoA > 0) + 1.0*np.array(df_aug_feats.numMoA > 1)

#  by cp_dose -- no obvious pattern, just intermixed.

##colors = 1.0*np.array(df_aug_feats.cp_dose == 'D1')

#  by cp_time -- clear patterns in plot

##colors = 1.0*np.array(df_aug_feats.cp_time)

#

df_aug_feats.plot(x='g-95%',y='g-5%',kind='scatter', figsize=(12,7),

                 c=colors, colormap='jet', alpha=0.25, marker='o',s=50,

                 title='g-5% vs g-95% for all non-control sig_ids'+

                  '   Colored by number of MoAs (0, 1, 2+)')

plt.show()
# Look at g-95% vs g-hif

#

# Choose how to color the points:

#  by numMoA -- looks useful to help determine the MoA

colors = 1.0*np.array(df_aug_feats.numMoA > 0) + 1.0*np.array(df_aug_feats.numMoA > 1)

#  by cp_dose -- no obvious pattern, just intermixed.

##colors = 1.0*np.array(df_aug_feats.cp_dose == 'D1')

#  by cp_time -- clear patterns in plot

##colors = 1.0*np.array(df_aug_feats.cp_time)

#

df_aug_feats.plot(x='g-hif',y='g-95%',kind='scatter', figsize=(12,7),

                 c=colors, colormap='jet', alpha=0.25, marker='o',s=50,

                 title='g-95% vs g-hif for all non-control sig_ids'+

                  '   Colored by number of MoAs (0, 1, 2+)')

plt.show()
# Look at g-5% vs g-lof

#

# Choose how to color the points:

#  by numMoA -- looks useful to help determine the MoA

colors = 1.0*np.array(df_aug_feats.numMoA > 0) + 1.0*np.array(df_aug_feats.numMoA > 1)

#  by cp_dose -- no obvious pattern, just intermixed.

##colors = 1.0*np.array(df_aug_feats.cp_dose == 'D1')

#  by cp_time -- clear patterns in plot

##colors = 1.0*np.array(df_aug_feats.cp_time)

#

df_aug_feats.plot(x='g-lof',y='g-5%',kind='scatter', figsize=(12,7),

                 c=colors, colormap='jet', alpha=0.25, marker='o',s=50,

                 title='g-5% vs g-lof for all non-control sig_ids'+

                  '   Colored by number of MoAs (0, 1, 2+)')

plt.show()
# Look at g-hif vs C-ave

#

# Choose how to color the points:

#  by numMoA -- looks useful to help determine the MoA

colors = 1.0*np.array(df_aug_feats.numMoA > 0) + 1.0*np.array(df_aug_feats.numMoA > 1)

#  by cp_dose -- no obvious pattern, just intermixed.

##colors = 1.0*np.array(df_aug_feats.cp_dose == 'D1')

#  by cp_time -- clear patterns in plot

##colors = 1.0*np.array(df_aug_feats.cp_time)

#

df_aug_feats.plot(x='c-ave',y='g-hif',kind='scatter', figsize=(12,7),

                 c=colors, colormap='jet', alpha=0.25, marker='o',s=50,

                 title='g-hif vs C-ave for all non-control sig_ids'+

                  '   Colored by number of MoAs (0, 1, 2+)')

plt.show()







#  by cp_time -- clear patterns in plot

colors = 1.0*np.array(df_aug_feats.cp_time)

#

df_aug_feats.plot(x='c-ave',y='g-hif',kind='scatter', figsize=(12,7),

                 c=colors, colormap='jet', alpha=0.25, marker='o',s=50,

                 title='g-hif vs C-ave for all non-control sig_ids'+

                  '   Colored by cp_time (24, 48, 72)')



plt.show()
# Look at g-95% vs C-ave

#

# Choose how to color the points:

#  by numMoA -- looks useful to help determine the MoA

colors = 1.0*np.array(df_aug_feats.numMoA > 0) + 1.0*np.array(df_aug_feats.numMoA > 1)

#  by cp_dose -- no obvious pattern, just intermixed.

##colors = 1.0*np.array(df_aug_feats.cp_dose == 'D1')

#  by cp_time -- clear patterns in plot

##colors = 1.0*np.array(df_aug_feats.cp_time)

#

df_aug_feats.plot(x='c-ave',y='g-95%',kind='scatter', figsize=(12,7),

                 c=colors, colormap='jet', alpha=0.25, marker='o',s=50,

                 title='g-95% vs C-ave for all non-control sig_ids'+

                  '   Colored by number of MoAs (0, 1, 2+)')

plt.show()





colors = 1.0*np.array(df_aug_feats.cp_time)

#

df_aug_feats.plot(x='c-ave',y='g-95%',kind='scatter', figsize=(12,7),

                 c=colors, colormap='jet', alpha=0.25, marker='o',s=50,

                 title='g-95% vs C-ave for all non-control sig_ids'+

                  '   Colored by cp_time (24, 48, 72)')

plt.show()
# One more strange 'feature': g-hilopc = g-95%/g-5%

#                     and/or  g-hilof = g-hif/g-lof



# Of the two, hilof may be 'better', information-wise,

# though both are unconvincing in the scatter plots.

# Leave them out of features.



# Choose how to color the points:

#  by numMoA -- looks useful to help determine the MoA

colors = 1.0*np.array(df_aug_feats.numMoA > 0) + 1.0*np.array(df_aug_feats.numMoA > 1)

#  by cp_dose -- no obvious pattern, just intermixed.

##colors = 1.0*np.array(df_aug_feats.cp_dose == 'D1')

#  by cp_time -- clear patterns in plot

##colors = 1.0*np.array(df_aug_feats.cp_time)

#

df_aug_feats.plot(x='c-ave',y='g-hilof',kind='scatter', figsize=(12,7),

                 c=colors, colormap='jet', alpha=0.25, marker='o',s=20,

                 title='g-hilof vs C-ave for all non-control sig_ids'+

                  '   Colored by number of MoAs (0, 1, 2+)', ylim=(0,2))





#  by cp_time -- clear patterns in plot

colors = 1.0*np.array(df_aug_feats.cp_time)

#

df_aug_feats.plot(x='c-ave',y='g-hilof',kind='scatter', figsize=(12,7),

                 c=colors, colormap='jet', alpha=0.25, marker='o',s=20,

                 title='g-hilof vs C-ave for all non-control sig_ids'+

                  '   Colored by cp_time (24, 48, 72)', ylim=(0,2))

plt.show()

# The features in df_treat_feats.columns are:

#iloc

#   0  sig_id  - anonymized values (I believe)

#   1  cp_type - determines controls

#   2  cp_time - 3 values: 24, 48, 72

#   3  cp_dose - 2 values D1, D2

#

#   4  g-0  to -  ~ Gaussian-with-outliers, 772 of them in all

# 775  g-771

#

# 776  c-0  to - ?, 100 of them in all

# 875  c-99



# Look at the values using iloc[ irow, icol ]



# Pick a single feature (COLUMN), and plot the feature values for a bunch of the ids.

icol = 4+600

col_str = df_treat_feats.columns[icol]

# Show some controls and treatment ones

plt.plot(df_treat_feats.iloc[1000:1500, icol].values,'r.')

plt.plot(df_train_ctls.iloc[0:500, icol].values,'y.',alpha=0.5)



plt.title(col_str+" : Treated (red) and  Control (yellow)")

plt.ylabel("Value of the feature")

plt.xlabel("A bunch of sig_ids")

plt.show()
# The features in df_treat_feats.columns are:

#iloc

#   0  sig_id  - anonymized values (I believe)

#   1  cp_type - determines controls

#   2  cp_time - 3 values: 24, 48, 72

#   3  cp_dose - 2 values D1, D2

#

#   4  g-0  to -  ~ Gaussian-with-outliers, 772 of them in all

# 775  g-771

#

# 776  c-0  to - ?, 100 of them in all

# 875  c-99



# Look at the values using iloc[ irow, icol ]



# Single sig_id (ROW), show a bunch of the target values



# Looking at the g values for a single id

irow = 201



icol = 4  # 4 is start of the g features

col_str = df_treat_feats.columns[icol]

icols = list(range(icol, icol+772))

# Show some controls and treatment ones

plt.plot(df_treat_feats.iloc[irow, icols].values,'r.')

plt.plot(df_train_ctls.iloc[irow, icols].values,'y.',alpha=0.5)



plt.title(df_treat_feats.iloc[irow,0]+" : Treated (red) and  Control (yellow)")

plt.ylabel("Value of the feature")

plt.xlabel(col_str+" and following target values")

plt.show()
# The g-0 to g-771 column names are:

gcol_strs = df_train_feats.columns[4:4+772]
# Form and look at the average g-vectors for the CONTROLS

sel_dose = 'All'    # No selection on dose

sel_time = 244872   # 24, 48, 72, or 244872 for all

targ_str = 'Controls'

if sel_time > 100:

    cp_select = ((df_train_feats.cp_type == 'ctl_vehicle') &

             (df_train_feats.cp_time < sel_time))

else:

        cp_select = ((df_train_feats.cp_type == 'ctl_vehicle') &

             (df_train_feats.cp_time == sel_time))

        

# Get the statistics on the g columns

df_g_stats = df_train_feats.loc[cp_select,gcol_strs].describe()

# transpose it so that the gs are in rows and the stats in columns

df_g_stats = df_g_stats.T

# add a z-score column

sqrt_count = np.sqrt(df_g_stats.loc['g-0','count'])

df_g_stats['z-score'] = df_g_stats['mean'] / (df_g_stats['std'] / sqrt_count)



print("\nThere are",df_g_stats.loc['g-0','count']," ids with dose="+sel_dose+

          ", time="+str(sel_time)+", and MoA is",targ_str)





# Save this g-vector of mean control values,

# it will be subtracted from target g-vectors to reduce the control pattern.

control_means = df_g_stats['mean']
# Show the mean control vector values and their stds 

df_g_stats[['mean']].plot(kind='line',style='.b')

plt.title("g-vector values: {:.0f} ids,".format(df_g_stats.loc['g-0','count']) +

          " dose="+sel_dose+", time="+str(sel_time)+", MoA="+targ_str)

plt.show()



df_g_stats[['std']].plot(kind='line',style='.b')

plt.title("stds of the g-vector")

plt.show()

# Look at a selected MoA=1 target:

if True:

    # Pick the target:

    ##targ_str = 'serotonin_receptor_agonist'

    ##targ_str = 'serotonin_receptor_antagonist'

    ##targ_str = 'calcium_channel_blocker'

    ##targ_str = 'vegfr_inhibitor'

    ##targ_str = 'cdk_inhibitor'

    targ_str = 'nfkb_inhibitor'

    

    # dose and time to use:

    sel_dose = 'All'  # No selection on dose

    sel_time = 244872



    if sel_time > 100:

        cp_select = (df_train_targs[targ_str] > 0) & (df_train_feats.cp_time < sel_time)

    else:

        cp_select = (df_train_targs[targ_str] > 0) & (df_train_feats.cp_time == sel_time)

else:

    # For comparison can also look at (some of) the controls

    sel_dose = 'All'  # No selection on dose

    sel_time = 48

    targ_str = 'Controls'

    cp_select = ((df_train_feats.cp_type == 'ctl_vehicle') &

             (df_train_feats.cp_time == sel_time))





df_g_stats = df_train_feats.loc[cp_select,gcol_strs].describe()

# transpose it so that the gs are in rows and the stats in columns

df_g_stats = df_g_stats.T

# add a z-score column

sqrt_count = np.sqrt(df_g_stats.loc['g-0','count'])

# Subtract off the control means

df_g_stats['mean-ctl'] = df_g_stats['mean'] - control_means

df_g_stats['z-score'] = (df_g_stats['mean-ctl'] / (df_g_stats['std'] / sqrt_count))

# and an abs(z-score):

df_g_stats['z-abs'] = df_g_stats['z-score'].abs()



print("\nThere are",df_g_stats.loc['g-0','count']," ids with dose="+sel_dose+

          ", time="+str(sel_time)+", and MoA is",targ_str)

##df_g_stats[['mean']].plot(kind='line',style='.b')

##plt.show()



df_g_stats[['mean-ctl']].plot(kind='line',style='.g')

plt.title("g-vector values: {:.0f} ids,".format(df_g_stats.loc['g-0','count']) +

          " dose="+sel_dose+", time="+str(sel_time)+", MoA="+targ_str)

plt.show()



df_g_stats[['std']].plot(kind='line',style='.b')

plt.title("stds of the g-vector")

plt.show()



if LOCATION_KAGGLE:

    # Define the columns of the g-vectors dataframe

    all_cols = ['targ_str','count','median_std']

    # add on the g columns

    all_cols = all_cols + list(gcol_strs)



    # Setup a dataframe

    df_g_vectors = pd.DataFrame(columns=all_cols)



    print(df_g_vectors)





    # Go through the MoA targets



    all_targ_strs = list(df_treat_targs.columns[1:])



    # dose and time to use:

    sel_dose = 'All'  # No selection on dose

    sel_time = 244872

    

    for targ_str in all_targ_strs:



        if sel_time > 100:

            cp_select = (df_train_targs[targ_str] > 0) & (df_train_feats.cp_time < sel_time)

        else:

            cp_select = (df_train_targs[targ_str] > 0) & (df_train_feats.cp_time == sel_time)



        df_g_stats = df_train_feats.loc[cp_select,gcol_strs].describe()

        # transpose it so that the gs are in rows and the stats in columns

        df_g_stats = df_g_stats.T

        # add a z-score column

        sqrt_count = np.sqrt(df_g_stats.loc['g-0','count'])

        # Subtract off the control means

        df_g_stats['mean-ctl'] = df_g_stats['mean'] - control_means



        ##print("Averaging",df_g_stats.loc['g-0','count']," ids with MoA = ",targ_str)



        # Add this g-vector to the dataframe

        row_values = [targ_str, df_g_stats.loc['g-0','count'], df_g_stats['std'].median()]

        for gval in list(df_g_stats['mean-ctl']):

            row_values.append(gval)

        df_g_vectors = df_g_vectors.append(pd.DataFrame([row_values],

                            columns=df_g_vectors.columns),ignore_index=True)



    # save the dataframe

    df_g_vectors.to_csv("g_vectors_"+sel_dose+str(sel_time)+"_"+

                    version_str+".csv",index=False,float_format='%.3f')

    

else:

    # Read in a previously saved file:

    df_g_vectors = pd.read_csv("./g_vectors_All244872_v10_SAVE.csv")
# Each row has the average g-values for one of the MoAs.

df_g_vectors
# Show that it has the same values for the example shown before:

targ_str = 'nfkb_inhibitor'

# select the desired row

irow = sum(df_g_vectors.index * (df_g_vectors.targ_str == targ_str))

df_g_vectors.iloc[irow,3:3+772].plot(kind='line',style='.g')

plt.title("g-vector for MoA = "+targ_str)

plt.show()
# Get statistics for each target MoA row 

df_target_stats = df_g_vectors.drop(columns=

        ['targ_str','count','median_std']).T.describe(percentiles=[0.05,0.5,0.95]).T

df_target_stats
# Show the ~ max, min g-values for the MoA targets

df_target_stats.plot('95%','5%',kind='scatter')

plt.show()



# This linear shape comes from having similar magnitudes

# for the most positive and for the most negative of the g-values.
# The max - min :

targ_max_min_sort = (df_target_stats['95%'] - 

                   df_target_stats['5%']).sort_values(ascending=False)

plt.plot(targ_max_min_sort.values,'.b')

plt.show()
# Select the ones that are above 2.4(24) or 3.0(13)

# as the ones that are most 'predictable':

n_predict = 24

targs_to_predict = targ_max_min_sort[0:n_predict]

targs_to_predict = list(targs_to_predict.index)



# show them

targ_max_min_sort[0:n_predict]

##targs_to_predict
# Show their names with the number of sig_ids they label

df_targ_to_predict = df_g_vectors.loc[targs_to_predict,['count','targ_str']]



# Show the ones with 70 or more counts:

list(df_targ_to_predict[df_targ_to_predict['count'] > 69].targ_str)

# Use results above for "guessing" estimate of score, and repeat the calculation, but

# subtract off the errors for the targs_to_predict ones, assuming we 'know' them:

print("If we can correctly predict all {} targets above,\n".format(len(targs_to_predict)),

      "then the expected score (incl controls) is:\n",

      ((len(df_treat_targs)/len(df_train_targs)) * 

         (sum(scores_targs) - sum(scores_targs[targs_to_predict])) / len(aves_targs)))
# Get statistics for each single g component (column) over the 206 MoAs

df_vector_stats = df_g_vectors.describe(percentiles=

                    [0.05,0.5,0.95]).drop(columns=['count','median_std']).T

# KLUDGE: Same stats but over all samples (instead of one per MoA)

##df_vector_stats = df_aug_feats[gcol_strs].describe(percentiles=

##                    [0.05,0.5,0.95]).T



df_vector_stats
df_vector_stats.hist('95%',bins=50)

plt.show()



df_vector_stats.hist('5%',bins=50)

plt.show()

df_vector_stats.plot('95%','5%',kind='scatter')

plt.show()



# This plot shows that an individual g features tends to

# either go positive or go negative, from the control range around (0.5, -0.5).

# There are 3 that seem to be more symmetric, two near (-1.7,1) and one (1.5,-1.2).
gs_max_min_sort = (df_vector_stats['95%'] - 

                   df_vector_stats['5%']).sort_values(ascending=False)

gs_max_min_sort.plot(style='.')

plt.show()
# Most 'sensitive' are the ones above 2.4 :

# Put their names in a list:

gs_to_use_22 = gs_max_min_sort[0:22]

gs_to_use_22 = list(gs_to_use_22.index)



# show them

gs_max_min_sort[0:22]
# Dataframe of the "gs to use" in columns and targets (206) as rows.

df_g_vectors[gs_to_use_22]
# Get the correlations between these selected gs

# use abs to better show highly correlated from little correlation

gs_corr = np.abs(df_g_vectors[gs_to_use_22].corr())
# plot the heatmap

sns.heatmap(gs_corr,

        xticklabels=gs_corr.columns,

        yticklabels=gs_corr.columns)

plt.show()
# Manually pick an uncorrelated subset:

#   g-231, g-175, g-178 are very uncorrelated with others !?

#   g-75 and g-65 standout, and g-332 too, though less so.

#   The first 3: g-392, g-100, g-158 and g-91

#     also have low-ish correlation with most others.

#   Finally, g-50 has high correlation with most of the rest.



if True:

    # So use this subset:

    gs_to_use = ['g-392', 'g-100', 'g-158', 'g-91',

             'g-231', 'g-175', 'g-178',

             'g-75', 'g-65', 'g-332',  'g-50']



    gs_corr = np.abs(df_g_vectors[gs_to_use].corr())

    sns.heatmap(gs_corr,

        xticklabels=gs_corr.columns,

        yticklabels=gs_corr.columns)

    plt.show()
# Show the histograms of each of the selected g feature over the 206 taargets

# (Want to use a log scale for the y-axis.)

df_g_vectors[gs_to_use].hist(figsize=(10,8),sharex=True,sharey=True,layout=(3,4),bins=20)

plt.show()





if True:

    # Show the heatmap for all the g- features that will be used:

    gs_corr = np.abs(df_aug_feats[['g-hif','g-95%'] + gs_to_use].corr())

    sns.heatmap(gs_corr,

        xticklabels=gs_corr.columns,

        yticklabels=gs_corr.columns)

    plt.show()
if True:

    # Scatter plot of a specific g feature vs g-hif  over all sig_ids



    colors = 1.0*np.array(df_aug_feats.numMoA > 0) + 1.0*np.array(df_aug_feats.numMoA > 1)

    #  by cp_time -- clear patterns in plot

    ##colors = 1.0*np.array(df_aug_feats.cp_time)



    df_aug_feats.plot('g-hif','g-175',kind='scatter',c=colors,figsize=(9,6),

                 colormap='jet', alpha=0.25, marker='o',s=20)

    plt.show()
# 23 Targets we want to detect better:



# All above 0.010 except for the tSNE-9:     * Not very detectable *

targ_do_better = ['acetylcholine_receptor_antagonist', 'adrenergic_receptor_agonist',

               'adrenergic_receptor_antagonist', 'calcium_channel_blocker', 'cyclooxygenase_inhibitor',

               'dna_inhibitor', 'dopamine_receptor_antagonist', 'flt3_inhibitor',

               'glutamate_receptor_antagonist', 'histamine_receptor_antagonist', 'kit_inhibitor',

               'pdgfr_inhibitor', 'phosphodiesterase_inhibitor', 'serotonin_receptor_agonist',

               'serotonin_receptor_antagonist', 'sodium_channel_inhibitor']

##targ_subset_name = "16: >0.010, w/o tSNE-9"



# These are the targets with 'highly detectable' g-vectors (unless listed in tSNE-9 already)

targ_do_better = (targ_do_better + 

        ['topoisomerase_inhibitor', 'hdac_inhibitor', 'mtor_inhibitor', 'mek_inhibitor',

        'pi3k_inhibitor', 'protein_synthesis_inhibitor', 'atpase_inhibitor'])

##targ_subset_name = "7: detectable"



len(targ_do_better)
# Get statistics for each single g component (column) over the 23 DO BETTER MoAs

df_dobetter_vects = df_g_vectors.set_index('targ_str').copy()

df_dobetter_vects = df_dobetter_vects.loc[targ_do_better]

df_dobetter_stats = df_dobetter_vects.describe(percentiles=

                    [0.15,0.5,0.85]).drop(columns=['count','median_std']).T



df_dobetter_stats
df_dobetter_stats.hist('85%',bins=50)

plt.show()



df_dobetter_stats.hist('15%',bins=50)

plt.show()



df_dobetter_stats.hist('std',bins=50)

plt.show()
df_dobetter_stats.plot('85%','15%',kind='scatter')

plt.show()
dbgs_max_min = (df_dobetter_stats['85%'] - df_dobetter_stats['15%'])

df_dobetter_stats['85%-15%'] = dbgs_max_min



df_dobetter_stats.plot('std','85%-15%',kind='scatter')

plt.show()



dbgs_max_min_sort = dbgs_max_min.sort_values(ascending=False)

dbgs_max_min_sort.plot(style='.')

plt.show()
# Most 'sensitive' are the ones above 1.75:

# Put their names in a list:

dbgs_num = 28

dbgs_to_use = dbgs_max_min_sort[0:dbgs_num]

dbgs_to_use_28 = list(dbgs_to_use.index)



# show them

dbgs_max_min_sort[0:dbgs_num]
# Drop any gs that are already in the gs_to_use_22 list:

dbgs_to_add = dbgs_to_use_28.copy()

for this_g in gs_to_use_22:

    try:

        dbgs_to_add.remove(this_g)

    except:

        pass



print("\nFound",len(dbgs_to_add),"g-vectors to add to the features:\n")

print(dbgs_to_add)



# Using 15%, 85%   shares: 146, 201, 228, 72

# ['g-146', 'g-201', 'g-406', 'g-208', 'g-215', 'g-386', 'g-228', 'g-529', 'g-298', 'g-72']
# Look at the correlations between these

if True:

    dbgs_corr = np.abs(df_g_vectors[dbgs_to_add].corr())

    sns.heatmap(dbgs_corr,

        xticklabels=dbgs_corr.columns,

        yticklabels=dbgs_corr.columns)

    plt.show()
# Similar to the above for g features BUT did not make MoA average c vectors - use all the data.

# C columns:

# 776  c-0  to - ?, 100 of them in all

# 875  c-99

ccol_strs = df_aug_feats.columns[776:875+1]

##ccol_strs
# KLUDGE: Same stats but over all samples (instead of one per MoA)

df_ccol_stats = df_aug_feats[ccol_strs].describe(percentiles=

                    [0.05,0.5,0.95]).T



##df_ccol_stats
df_ccol_stats.plot('95%','5%',kind='scatter')

plt.show()



# There is very little variation in the 95% values (survival)

# it's the 5% (lowest) values that vary across the cs.
cs_max_min_sort = (df_ccol_stats['95%'] - 

                   df_ccol_stats['5%']).sort_values(ascending=False)

cs_max_min_sort.plot(style='.')

plt.show()
# Most 'sensitive' are the ones above 6.8 :

# Put their names in a list:

cs_to_use = cs_max_min_sort[0:18]

cs_to_use = list(cs_to_use.index)



# show them

cs_max_min_sort[0:18]
# Get the correlations between these selected cs

# use abs to better show highly correlated from little correlation

cs_corr = np.abs(df_aug_feats[cs_to_use].corr())
# plot the heatmap

sns.heatmap(cs_corr,

        xticklabels=cs_corr.columns,

        yticklabels=cs_corr.columns)



# As noted the c- features are very correlated...
# Manually pick a few for a least-correlated subset:

if True:

    # So use this subset (including c-38 that seems correlated with most):

    cs_to_use = ['c-38','c-65','c-70','c-48']



    # Show the correlations and include the other c- features too:

    cs_corr = np.abs(df_aug_feats[['c-ave','c-std','c-5%','c-95%'] + 

                                  cs_to_use].corr())

    sns.heatmap(cs_corr,

        xticklabels=cs_corr.columns,

        yticklabels=cs_corr.columns)

# Scatter plot of a specific c feature vs c-ave



colors = 1.0*np.array(df_aug_feats.numMoA > 0) + 1.0*np.array(df_aug_feats.numMoA > 1)

#  by cp_time -- clear patterns in plot

##colors = 1.0*np.array(df_aug_feats.cp_time)



df_aug_feats.plot('c-ave','c-65',kind='scatter',c=colors,figsize=(9,6),

                 colormap='jet', alpha=0.25, marker='o',s=20)

plt.show()
# t-SNE results

#

# Doing t-SNE on the train features (below, using the 11 features in gs_to_use)

# identified some clear "islands" of active MoA sig_ids.

# Each island is dominated by a single target MoA.

# This suggests that it should be possible to predict these particular MoAs accurately.

# These are the island-targets I identified:  (the scores are from, e.g., scores_targs['hsp_inhibitor'])

#                    cts = counts in a specific island region (not all for the target)

#                    * = also in the most-detectable MoA list above.

# cts  score         MoA target

# 718  0.1453   *   proteasome_inhibitor  AND  <-- these two mostly appear together

# 718  0.1612   *      nfkb_inhibitor          <--

# 236  0.0655       glucocorticoid_receptor_agonist

# 185  0.0567   *   raf_inhibitor

# 145  0.0799   *   cdk_inhibitor

# 142  0.0689       hmgcr_inhibitor  81+61

#  92  0.0792       egfr_inhibitor

#  66  0.0274   *   hsp_inhibitor

#  49  0.0754   *   tubulin_inhibitor

#

# These were all put in a subset list, repeated here:

tSNE_9subset = ['proteasome_inhibitor', 'nfkb_inhibitor',  'glucocorticoid_receptor_agonist',

               'raf_inhibitor', 'cdk_inhibitor', 'hmgcr_inhibitor', 

               'egfr_inhibitor', 'hsp_inhibitor', 'tubulin_inhibitor']

# Use results above for "guessing" estimate of score, and repeat the calculation, but

# subtract off the errors for the *** targ_subset *** ones, assuming we 'know' them:

print("If we can correctly predict all {} of the t-SNE targets above,\n".format(len(tSNE_9subset)),

      "then the expected score (incl controls) is:\n",

      ((len(df_treat_targs)/len(df_train_targs)) * 

         (sum(scores_targs) - sum(scores_targs[tSNE_9subset])) / len(aves_targs)))
# Calculating the rms variation of a target's t-SNE points around their average -

# a rough idea of clustering?

#

# The targets with high counts (>200) and lowish rms (<58)

#                             targ_str  count       0-ave      1-ave        rms

# these two:

# 136                   nfkb_inhibitor  832.0  112.883379  -0.535171  54.513346

# 163             proteasome_inhibitor  726.0  129.556319  -0.973011  25.720070

# and these four are also listed above

# 63                     cdk_inhibitor  340.0   85.844863 -14.761353  43.245544

# 96   glucocorticoid_receptor_agonist  266.0   50.430224  74.849734  35.988496

# 109                  hmgcr_inhibitor  283.0   39.967208 -55.079256  46.488723

# 169                    raf_inhibitor  223.0   36.522937 -81.275786  48.158390

tSNE_4subset = ['glucocorticoid_receptor_agonist',

               'raf_inhibitor', 'cdk_inhibitor', 'hmgcr_inhibitor']
# Use results above for "guessing" estimate of score, and repeat the calculation, but

# subtract off the errors for the *** targ_subset *** ones, assuming we 'know' them:

print("If we can correctly predict all {} of the t-SNE targets above,\n".format(len(tSNE_4subset)),

      "then the expected score (incl controls) is:\n",

      ((len(df_treat_targs)/len(df_train_targs)) * 

         (sum(scores_targs) - sum(scores_targs[tSNE_4subset])) / len(aves_targs)))
# Doing the t-SNE

# Can choose to do it or not, always do it when submitting to Kaggle:

DO_TSNE = False
if DO_TSNE or LOCATION_KAGGLE:

    # Select the train data with the features to do t-SNE on  *** Include the cs_to_use too ***

    # All sig_ids:

    train_data_TSNE = df_train_feats[gs_to_use + cs_to_use].copy()

    print(train_data_TSNE.columns)

    

    # Set basic parameters

    this_TSNE = TSNE(perplexity=30.0, learning_rate=200.0, init='pca', 

                 n_iter=2000, n_iter_without_progress=150, min_grad_norm=1e-6,

                 verbose=1, random_state=17, n_jobs=-2)



    # Do the t-SNE on all, or use a subset of the data

    ##n_samples = 5000

    n_samples = len(train_data_TSNE)   # Use All



    t_start = time()

    # Do the fit & transform

    train_TSNE_out = this_TSNE.fit_transform(train_data_TSNE.iloc[0:n_samples])

    train_TSNE_out = pd.DataFrame(train_TSNE_out)

    print(" t-SNE(Train)   Total time:",time()-t_start,"   Iterations:", this_TSNE.n_iter_)

if DO_TSNE or LOCATION_KAGGLE:

    # Make a dataframe for some t-SNE summary info by target

    # Start with the first 2 columns of df_g_vectors

    df_tSNE_summary = df_g_vectors[['targ_str','count']].copy()

    n_targs = len(df_tSNE_summary)

    tsne0_ave = np.zeros(n_targs)

    tsne1_ave = np.zeros(n_targs)

    tsne_rms = np.zeros(n_targs)



    for itarg, this_targ in enumerate(df_tSNE_summary['targ_str']):

        # Find all the sig_ids for this target

        targ_sel = (df_train_targs[this_targ] > 0)

        num_ids = sum(targ_sel)

        tsne0_ave[itarg] = sum(train_TSNE_out[0] * 1.0*targ_sel)/num_ids

        tsne1_ave[itarg] = sum(train_TSNE_out[1] * 1.0*targ_sel)/num_ids

        tsne_rms[itarg] = np.sqrt(sum( (1.0*targ_sel)*((train_TSNE_out[0]-tsne0_ave[itarg])**2 +

                                        (train_TSNE_out[1]-tsne1_ave[itarg])**2))/num_ids)

        # print stuff and cut it short for testing...

        ##print(itarg,this_targ,num_ids,tsne0_ave[itarg],tsne1_ave[itarg],tsne_rms[itarg])

        ##if itarg > 9:

        ##    break

    

    # Put them in the df

    df_tSNE_summary['0-ave'] = tsne0_ave

    df_tSNE_summary['1-ave'] = tsne1_ave

    df_tSNE_summary['rms'] = tsne_rms



    # Show info about them

    df_tSNE_summary.hist('rms',bins=90)

    plt.title("Histogram of the t-SNE rms for the targets")

    plt.show()



    df_tSNE_summary.plot('0-ave','1-ave',c='rms',cmap='jet',kind='scatter',s=10)

    plt.title("Location of the t-SNE averages for the targets")

    plt.show()



    df_tSNE_summary.plot('rms','count',c='rms',cmap='jet',kind='scatter',s=10)

    plt.title("Counts vs t-SNE rms for the targets")

    plt.show()



    # List high-counts, lowish rms ones

    print(df_tSNE_summary[( (df_tSNE_summary['rms'] < 58) & (df_tSNE_summary['count'] > 200) )])
if DO_TSNE or LOCATION_KAGGLE:

    # Show where Controls are

    colors = 1.0*(df_train_feats['cp_type'] != 'ctl_vehicle')

    plt.figure(figsize = [10.4, 8])

    plt.scatter(train_TSNE_out[0], train_TSNE_out[1], 

            cmap='prism', c=colors[0:n_samples], alpha=0.3, s=5)

    plt.title("t-SNE Output with color-coding: Controls (red), non-Controls (green)")

    plt.xlabel("t-SNE[0]"); plt.ylabel("t-SNE[1]")

    plt.savefig('tSNE_controls_coloring.jpg')

    plt.show()

    

    # Show where any active MoA is

    numMoA = df_train_targs.sum(axis=1)

    colors = 1.0*(numMoA > 0)

    ##print("Sum of MoA colors = ",sum(colors), sum(colors > 0))

    plt.figure(figsize = [10.4, 8])

    plt.scatter(train_TSNE_out[0], train_TSNE_out[1], 

            cmap='prism', c=colors[0:n_samples], alpha=0.3, s=5)

    plt.title("t-SNE Output with color-coding: numMoA=0 (red), numMoA >= 1 (green)")

    plt.xlabel("t-SNE[0]"); plt.ylabel("t-SNE[1]")

    plt.savefig('tSNE_MoA_coloring.jpg')

    

    # Show where a selected target's MoAs are set in tSNE space

    #

    # Both nfkb_inhibitor and proteasome_inhibitor are set:

    ##colors = 1.0*((df_train_targs['nfkb_inhibitor'] > 0) &

    ##            (df_train_targs['proteasome_inhibitor'] > 0))

    # Other 'island's:

    ##colors = 1.0*(df_train_targs['cdk_inhibitor'] > 0) 

    ##colors = 1.0*(df_train_targs['raf_inhibitor'] > 0)

    ##colors = 1.0*(df_train_targs['glucocorticoid_receptor_agonist'] > 0)

    # Or any that are in the targ_subset

    colors = 1.0*(df_train_targs[targ_subset].sum(axis=1) > 0)

    #        

    ##print("Sum of MoA colors = ",sum(colors), sum(colors > 0))

    plt.figure(figsize = [10.4, 8])

    plt.scatter(train_TSNE_out[0], train_TSNE_out[1], 

            cmap='prism', c=colors[0:n_samples], alpha=0.3, s=5)

    plt.title("t-SNE Output with color-coding:  not selected(red), Target Subset (green)")

    plt.xlabel("t-SNE[0]"); plt.ylabel("t-SNE[1]")

    plt.savefig('tSNE_targetsubset_coloring.jpg')

    plt.show()
if DO_TSNE or LOCATION_KAGGLE:   

    # Do the test too:  doesn't have as many samples...    *** Include the cs_to_use too ***

    test_data_TSNE = df_test_feats[gs_to_use + cs_to_use].copy()

    print(test_data_TSNE.columns)

    this_TSNE = TSNE(perplexity=30.0, learning_rate=200.0, init='pca', 

                 n_iter=2000, n_iter_without_progress=150, min_grad_norm=1e-6,

                 verbose=1, random_state=17, n_jobs=-2)

    n_samples = len(test_data_TSNE)   # Use All

    t_start = time()

    test_TSNE_out = this_TSNE.fit_transform(test_data_TSNE.iloc[0:n_samples])

    test_TSNE_out = pd.DataFrame(test_TSNE_out)

    print(" t-SNE(Test)   Total time:",time()-t_start,"   Iterations:", this_TSNE.n_iter_)



    # Show where TEST Controls are

    colors = 1.0*(df_test_feats['cp_type'] != 'ctl_vehicle')

    plt.figure(figsize = [10.4, 8])

    plt.scatter(test_TSNE_out[0], test_TSNE_out[1], 

            cmap='prism', c=colors[0:n_samples], alpha=0.3, s=5)

    plt.title("t-SNE -TEST- Output with color-coding: Controls (red), non-Controls (green)")

    plt.xlabel("t-SNE[0]"); plt.ylabel("t-SNE[1]")

    plt.savefig('tSNE_controls-TEST_coloring.jpg')

    plt.show()
# eXtreme Gradient Boost classifier



from xgboost import XGBClassifier



# Other ML things we'll use:

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss

from sklearn.metrics import make_scorer



from sklearn.model_selection import GridSearchCV

# Select and fill the features



# Drop some of the g features based on correlations and scatter plots

# Use all the gs_to_use_22 and add in the dbgs_to_add ones:

##features = (['c-ave', 'c-std', 'c-5%', 'c-95%'] + cs_to_use +

##           ['g-95%', 'g-hif'] + gs_to_use_22 + dbgs_to_add)



# Use the 11 gs_to_use and add in the 10 dbgs_to_add ones:

features = (['c-ave', 'c-std', 'c-5%', 'c-95%'] + cs_to_use +

           ['g-95%', 'g-hif'] + gs_to_use + dbgs_to_add)



print("\nLength of cs_to_use:",len(cs_to_use), cs_to_use)

print("\nLength of gs_to_use:",len(gs_to_use), gs_to_use_22)

print("\nTotal number of features used:",len(features))

print(features)



# Can select/update the desired target here:    OR NOT

# A single target ;-)



# Choose from the "tSNE 9":

#       'proteasome_inhibitor', 'nfkb_inhibitor', 'glucocorticoid_receptor_agonist',

#       'raf_inhibitor', 'cdk_inhibitor', 'hmgcr_inhibitor', 

#       'egfr_inhibitor', 'hsp_inhibitor', 'tubulin_inhibitor'

##targ_subset = ['nfkb_inhibitor'];  targ_subset_name = "1: nfkb"

##targ_subset = ['proteasome_inhibitor'];  targ_subset_name = "1: prot"

##targ_subset = ['glucocorticoid_receptor_agonist'];  targ_subset_name = "1: gluc"

##targ_subset = ['raf_inhibitor'];  targ_subset_name = "1: raf_"

##targ_subset = ['cdk_inhibitor'];  targ_subset_name = "1: cdk_"

##targ_subset = ['hmgcr_inhibitor'];  targ_subset_name = "1: hmgcr"

##targ_subset = ['egfr_inhibitor'];  targ_subset_name = "1: egfr"

##targ_subset = ['hsp_inhibitor'];  targ_subset_name = "1: hsp_"

##targ_subset = ['tubulin_inhibitor'];  targ_subset_name = "1: tubu"



# These are the targets with 'highly detectable' g-vectors (unless listed in tSNE-9 already)

#    'topoisomerase_inhibitor', 'hdac_inhibitor', 'mtor_inhibitor', 'mek_inhibitor',

#    'pi3k_inhibitor', 'protein_synthesis_inhibitor', 'atpase_inhibitor'

##targ_subset = ['topoisomerase_inhibitor'];  targ_subset_name = "1: topoi"

##targ_subset = ['hdac_inhibitor'];  targ_subset_name = "1: hdac_"

##targ_subset = ['mtor_inhibitor'];  targ_subset_name = "1: mtor_"

##targ_subset = ['mek_inhibitor'];  targ_subset_name = "1: mek_i"

##targ_subset = ['pi3k_inhibitor'];  targ_subset_name = "1: pi3k_"

##targ_subset = ['protein_synthesis_inhibitor'];  targ_subset_name = "1: prote"

##targ_subset = ['atpase_inhibitor'];  targ_subset_name = "1: atpas"



# Add other ones that are above 0.012:  (not necessarily 'detectable', though)

#      'acetylcholine_receptor_antagonist', 'adrenergic_receptor_agonist', 'adrenergic_receptor_antagonist',

#      'calcium_channel_blocker', 'cyclooxygenase_inhibitor', 'dna_inhibitor', 'dopamine_receptor_antagonist',

#      'flt3_inhibitor', 'glutamate_receptor_antagonist', 'histamine_receptor_antagonist',

#      'kit_inhibitor', 'pdgfr_inhibitor',

#      'phosphodiesterase_inhibitor', 'serotonin_receptor_antagonist', 'sodium_channel_inhibitor'

##targ_subset = ['acetylcholine_receptor_antagonist'];  targ_subset_name = "1: acety"

##targ_subset = ['adrenergic_receptor_agonist'];  targ_subset_name = "1: adren_ago"

##targ_subset = ['adrenergic_receptor_antagonist'];  targ_subset_name = "1: adren_ant"

##targ_subset = ['calcium_channel_blocker'];  targ_subset_name = "1: calci"

##

##targ_subset = ['histamine_receptor_antagonist'];  targ_subset_name = "1: hista"

## . . .

##targ_subset = ['sodium_channel_inhibitor'];  targ_subset_name = "1: sodiu"







# and fill the numSub from this target subset:

##df_aug_feats['numSub'] = df_treat_targs[targ_subset].sum(axis=1)

# Fill the X,y and Xkag,y_kag



# The Target is set here also:

# - Usually y=1 when numMoA is either =0 or >0

# - Another option is to use numSub instead:

#     this is the sum of MoAs of a subset of targets,

#     in the variable: targ_subset and 

#        given a name: targ_subset_name.

# Defined subsets are (the number is how many targets are in it):

#      "9: t-SNE islands"

#      "2: proteasome & nfkb"



USE_TARG_SUBSET = False



# Make 3 sets of X,y, selecting on cp_time:



# (v19+) The classifier target is   *** numMoA > 0 ***

# So y=1 means -->  numMoA > 0



# Select the three sets of sig_ids based on cp_time



select_train_24 = df_aug_feats['cp_time'] == 24

X24 = df_aug_feats.loc[ select_train_24, features ].copy()



select_train_48 = df_aug_feats['cp_time'] == 48

X48 = df_aug_feats.loc[ select_train_48, features ].copy()



select_train_72 = df_aug_feats['cp_time'] == 72

X72 = df_aug_feats.loc[ select_train_72, features ].copy()



# Their targets

if USE_TARG_SUBSET:

    # Usually use > 0, i.e, if MoA total is 1 or more.

    if ("2: pro" in targ_subset_name):

        # But, for the "2: prot..." subset use > 1 to find where both are active (very common.)

        y24 = 1.0*(df_aug_feats.loc[select_train_24, 'numSub'].values > 1)

        y48 = 1.0*(df_aug_feats.loc[select_train_48, 'numSub'].values > 1)

        y72 = 1.0*(df_aug_feats.loc[select_train_72, 'numSub'].values > 1)

    else:

        y24 = 1.0*(df_aug_feats.loc[select_train_24, 'numSub'].values > 0)

        y48 = 1.0*(df_aug_feats.loc[select_train_48, 'numSub'].values > 0)

        y72 = 1.0*(df_aug_feats.loc[select_train_72, 'numSub'].values > 0)

    

else:

    # Not using a subset, so use numMoA as the target

    y24 = 1.0*(df_aug_feats.loc[select_train_24, 'numMoA'].values > 0)

    y48 = 1.0*(df_aug_feats.loc[select_train_48, 'numMoA'].values > 0)

    y72 = 1.0*(df_aug_feats.loc[select_train_72, 'numMoA'].values > 0)

    

    

if USE_TARG_SUBSET:

    print("\n  *** A subset of targets is used:  "+targ_subset_name,"  ***")

    

print("\nThe X24, y24 have lengths of {} and {}.\n".format(len(X24),len(y24)))

print("The X48, y48 have lengths of {} and {}.\n".format(len(X48),len(y48)))

print("The X72, y72 have lengths of {} and {}.\n".format(len(X72),len(y72)))





# To-be-predicted features and (dummy) target

# select on cp_time:

select_kag_24 = df_test_feats['cp_time'] == 24

Xkag24 = df_test_feats.loc[select_kag_24, features].copy()

y_kag24 = np.zeros(len(Xkag24))

print("\nThe Xkag24, y_kag24 have lengths of {} and {}.\n".format(len(Xkag24),len(y_kag24)))



select_kag_48 = df_test_feats['cp_time'] == 48

Xkag48 = df_test_feats.loc[select_kag_48, features].copy()

y_kag48 = np.zeros(len(Xkag48))

print("The Xkag48, y_kag48 have lengths of {} and {}.\n".format(len(Xkag48),len(y_kag48)))



select_kag_72 = df_test_feats['cp_time'] == 72

Xkag72 = df_test_feats.loc[select_kag_72, features].copy()

y_kag72 = np.zeros(len(Xkag72))

print("The Xkag72, y_kag72 have lengths of {} and {}.\n".format(len(Xkag72),len(y_kag72)))



# The features don't need to be scaled.

# Optionally do this (if using a target subset)

# Disable because there are some many features ;-)

if False and USE_TARG_SUBSET:

    # For some targets and some features,

    # Go through the targets and for each target

    #   find all the sig_ids that have that target active and cp_time=48,

    #   and calculate the statistics of the feature values.

    #   For each of the features,

    #     plot all of that feature's values that are in the sig_ids selected.

    #

    # Use df_treat_targs and df_aug_feats to have access to everything...



    # select the target(s)

    ##targs_to_plot = ['nfkb_inhibitor','cdk_inhibitor']

    # Just show the last one

    targs_to_plot = [targ_subset[-1]]



    # These can be any features (not just gs),

    # some of the generally higher-importance ones

    ##gs_to_plot = ['g-hif','g-75','g-100','g-392','c-ave','c-95%']

    gs_to_plot = features



    # Loop over the targets

    for targ_str in targs_to_plot:

        # Get the sig_ids with this target active and cp_time is 48

        si_select = (df_treat_targs[targ_str] > 0) & (df_aug_feats['cp_time'] == 48)

        this_targ_gs = df_aug_feats.loc[si_select, gs_to_plot]

        print("\n\n"+targ_str+":\n")

        df_g_stats = this_targ_gs.describe()

        df_g_stats = df_g_stats.T.drop(columns=['min','max'])

        print(df_g_stats,"\n")

        # Make the plots for each g

        for this_g in gs_to_plot:

            # get the values

            gvals = this_targ_gs[this_g].values

            # sort them to see common levels (plateaus) - cute but not intuitive to view

            ##gvals.sort()

            plt.plot(gvals,'.b')

            plt.title("Values of "+this_g+" for all sig_ids with "+

                 targ_str+" = 1")

            plt.show()

# Create 'Jumbo' versions of each X and y that are

# 4 times larger and the Xs include added random noise.

  

if True:

    # Create 'Jumbo' versions of each X and y that are

    # 4 times larger and the Xs include added random noise.

    # 

    blurr_cols = cs_to_use + gs_to_use

    blurr_std = 0.5

    

    # Make the Xs

    X24J = X24.copy().append(X24, ignore_index=True).append(X24, 

            ignore_index=True).append(X24, ignore_index=True) 

    lenX = len(X24J)

    for this_col in blurr_cols:

        X24J[this_col] = X24J[this_col] + blurr_std * random.standard_normal(lenX)

        

    X48J = X48.copy().append(X48, ignore_index=True).append(X48, 

            ignore_index=True).append(X48, ignore_index=True)

    lenX = len(X48J)

    for this_col in blurr_cols:

        X48J[this_col] = X48J[this_col] + blurr_std * random.standard_normal(lenX)

        

    X72J = X72.copy().append(X72, ignore_index=True).append(X72, 

            ignore_index=True).append(X72, ignore_index=True)

    lenX = len(X72J)

    for this_col in blurr_cols:

        X72J[this_col] = X72J[this_col] + blurr_std * random.standard_normal(lenX)



        

    # Assemble the ys:

    y24J = np.concatenate([y24,y24,y24,y24])

    y48J = np.concatenate([y48,y48,y48,y48])

    y72J = np.concatenate([y72,y72,y72,y72])
# Thefollowing is from 40% of the way down on the page:

#   https://xgboost.readthedocs.io/en/latest/python/python_api.html



# XGBClassifier(

# max_depth=3, learning_rate=0.1, n_estimators=100,

# verbosity=1, objective='binary:logistic', booster='gbtree',

# tree_method='auto', n_jobs=1, gpu_id=-1,

# gamma=0, min_child_weight=1, max_delta_step=0,

# subsample=1, colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1,

# reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5,

# random_state=0, missing=None)



# get_params output, in alpha order:



#{      'base_score': 0.50,

# 'booster': 'gbtree',

# 'colsample_bylevel': 1,

# 'colsample_bynode': 1,

# 'colsample_bytree': 1,

#                           'gamma': 0,

#                           'learning_rate': 0.1,    # xgb's eta

# 'max_delta_step': 0,

#                           'max_depth': 1,

# 'min_child_weight': 1,

# 'missing': None,

#                           'n_estimators': 100,

# 'n_jobs': 1,

# 'nthread': None,

#       'objective': 'binary:logistic',

# 'random_state': 0,

# 'reg_alpha': 0,    # xgb's alpha

#       'reg_lambda': 1,   # xgb's lambda

# 'scale_pos_weight': 1,

# 'seed': None,

# 'silent': None,

#       'subsample': 1,

#       'verbosity': 1}



#   Used in (v11 with gs_to_use features as well)

#        "max_depth"        : 8,

#        "learning_rate"    : 0.05,

#        "n_estimators"     : 80,

#        "min_child_weight" : 3,

#        "gamma"            : 1.5,

#        "colsample_bytree" : 0.70,

#        "subsample"        : 1.0,

#        "reg_lambda"       : 1.0,



#   Used in (v14 - NO gs_to_use features)

#        "max_depth"        : 6,

#        "learning_rate"    : 0.03,

#        "n_estimators"     : 120,     # oopse: used 100 in v14, should be 120.

#        "min_child_weight" : 1,

#        "gamma"            : 1.5,

#        "colsample_bytree" : 0.90,

#        "subsample"        : 1.0,

#        "reg_lambda"       : 1.0,



xgb_params = {

        "max_depth"        : 8,   #

        "learning_rate"    : 0.06,

        "n_estimators"     : 60,

        "min_child_weight" : 4,    #

        "gamma"            : 1.5,

        "colsample_bytree" : 0.70, #

        "subsample"        : 1.0,

        "reg_lambda"       : 2.0,

    #

        "objective": "binary:logistic",

        "base_score" : 0.50,

        "verbosity" : 1

     }



# Setup hyper-parameter grid for the model:      *** For X48 fitting ***

xgb_param_grid = [

    {

        "max_depth"        : [5,6,7,8,9,10,11,12],                      # 10 is good

        ##"learning_rate"    : [0.04, 0.05, 0.06, 0.07, 0.08],  # 0.06

        ##"n_estimators"       : [40, 50, 60, 70, 80],          # 60 is good

        "min_child_weight"   : [1, 2, 3, 4, 6],                 # use 2

        ##"gamma"            : [0.5, 1.0, 1.5, 2.0, 4.0],       # 1.5 is good

        "colsample_bytree" : [0.5, 0.70, 0.9],                  # 0.80 is good

        ##"subsample"        : [0.8, 1.0],                      # keep value of 1

        ##"reg_lambda"       : [0.2, 0.50, 1.0, 2.0, 5.0],      # 2 is good

     }

]
model_name = 'xgb'

model_base = XGBClassifier(**xgb_params)

param_grid = xgb_param_grid
# Doing this fit here lets us skip over the Hyper-Parameter Search and continue on.



if True:

    # Use the 'Jumbo' versions of each X and y that are

    # 4 times larger and the Xs include added random noise.



    best_fit_mod24 = XGBClassifier(**xgb_params).fit(X24J,y24J)

    # Show these parameters

    print(best_fit_mod24.get_params())

    

    best_fit_mod48 = XGBClassifier(**xgb_params).fit(X48J,y48J)

    # Show these parameters

    print(best_fit_mod48.get_params())

    

    best_fit_mod72 = XGBClassifier(**xgb_params).fit(X72J,y72J)

    # Show these parameters

    print(best_fit_mod72.get_params())



    # Also define cv_folds, gscv_stats incase the following is skipped:

    cv_folds = 4

    gscv_stats = []

# Choose one model to show Feature Importance

best_fit_model = best_fit_mod48

X = X48.copy()

y = y48.copy()



# Show the model parameters:

best_fit_model.get_params()
# Feature importance



if model_name in ['lgr','dtc','rfc','gbc','mlp','xgb']:

    # Plot feature importance

    # Get feature importance

    if model_name == 'mlp':

        # For mlp regressor create a quasi-importance from the weights.

        # "The ith element in the list represents the weight matrix corresponding to layer i."

        # Input layer weights

        ##len(best_regressor.coefs_[0])

        # sum of abs() of input weights for each feature

        feature_importance = np.array([sum(np.abs(wgts)) for wgts in best_fit_model.coefs_[0] ])

    elif model_name == 'lgr':

        # For Logisitic Regression use the coeff.s to approximate an importance

        coeffs = best_fit_model.coef_[0]

        feature_importance = 0.0 * coeffs

        print(" Feature        Import.      coeff.    max from mean")

        for icol, col in enumerate(X.columns):

            col_mean = X[col].mean()

            col_max_from_mean = np.max(np.abs(X[col] - col_mean))

            feature_importance[icol] = abs(coeffs[icol]/col_max_from_mean)

            print("{:10}: {:10.3f}, {:10.3f}, {:10.2f}".format(col, feature_importance[icol], coeffs[icol], col_max_from_mean))

    else:

        # tree models have feature importance directly available:

        feature_importance = best_fit_model.feature_importances_

        

    # make importances relative to max importance

    max_import = feature_importance.max()

    feature_importance = 100.0 * (feature_importance / max_import)

    sorted_idx = np.argsort(feature_importance)

    pos = np.arange(sorted_idx.shape[0]) + 0.5



    plt.figure(figsize=(8, 15))

    ##plt.subplot(1, 2, 2)

    plt.barh(pos, feature_importance[sorted_idx], align='center')

    plt.yticks(pos, X.columns[sorted_idx])

    plt.xlabel(model_name.upper()+' -- Relative Importance')

    plt.title('           '+model_name.upper()+

              ' -- Variable Importance                  max --> {:.3f} '.format(max_import))



    plt.savefig(model_name.upper()+"_importance_"+version_str+".png")

    plt.show()

# Nominal accuracy of each model fit on the original training data



print("")

all_train_score = accuracy_score(y24, best_fit_mod24.predict(X24))

print("24: Nominal (thresh.=0.5) best-fit "+

      "All-Train accuracy: {:.2f} %\n".format(100.0*all_train_score))



print("")

all_train_score = accuracy_score(y48, best_fit_mod48.predict(X48))

print("48: Nominal (thresh.=0.5) best-fit "+

      "All-Train accuracy: {:.2f} %\n".format(100.0*all_train_score))



print("")

all_train_score = accuracy_score(y72, best_fit_model.predict(X72))

print("72: Nominal (thresh.=0.5) best-fit "+

      "All-Train accuracy: {:.2f} %\n".format(100.0*all_train_score))
if USE_TARG_SUBSET:

    print(targ_subset_name)
# Make the model probability predictions on the Training and Test (Kaggle) data



# The 'soft' probabilty values, go from 0 to 1

yh24 = best_fit_mod24.predict_proba(X24)

yh24 = yh24[:,1]

yh48 = best_fit_mod48.predict_proba(X48)

yh48 = yh48[:,1]

yh72 = best_fit_mod72.predict_proba(X72)

yh72 = yh72[:,1]



# Make the Kaggle set predictions too

yh_kag24 = best_fit_mod24.predict_proba(Xkag24)

yh_kag24 = yh_kag24[:,1]

yh_kag48 = best_fit_mod48.predict_proba(Xkag48)

yh_kag48 = yh_kag48[:,1]

yh_kag72 = best_fit_mod72.predict_proba(Xkag72)

yh_kag72 = yh_kag72[:,1]





# yh and yh_kag are the model probability predictions.

# Convert to discrete 0,1 using a threshold:

#

# Select the threshold based on balance between FP and FN,

# e.g. desired Precision, etc.

# Use the same for all cp_time values?

#

if USE_TARG_SUBSET:

    #

    # Threshold to use:  * Iteratively set this to get the desired precision, see below. *

    #

    yh_threshold = 0.236   # set for 95% precision

    #

    #

    # lower it for the "4: low..." subset

    if "4: low" in targ_subset_name:

        yh_threshold = 0.55

    # lower it for the "22: a above 0.01" subset

    if "22: a" in targ_subset_name:

        yh_threshold = 0.60

    #

else:

    # All targets used, for the y=1 is MoA>0 case

    yh_threshold = 0.693   # v32 Set for 99% precision  





# Apply the threshold to get binary predictions

# Training:

yp24 = 1.0*(yh24 > yh_threshold)

yp48 = 1.0*(yh48 > yh_threshold)

yp72 = 1.0*(yh72 > yh_threshold)

# Test (Kaggle):

yp_kag24 = 1.0*(yh_kag24 > yh_threshold)

yp_kag48 = 1.0*(yh_kag48 > yh_threshold)

yp_kag72 = 1.0*(yh_kag72 > yh_threshold)





print("")

ave_train_score = accuracy_score(y24, yp24)

print("24: Using a threshold of {} gives an ".format(yh_threshold)+

      "accuracy: {:.2f} %\n".format(100.0*ave_train_score))

ave_train_score = accuracy_score(y48, yp48)

print("48: Using a threshold of {} gives an ".format(yh_threshold)+

      "accuracy: {:.2f} %\n".format(100.0*ave_train_score))

ave_train_score = accuracy_score(y72, yp72)

print("72: Using a threshold of {} gives an ".format(yh_threshold)+

      "accuracy: {:.2f} %\n".format(100.0*ave_train_score))
# Combine all of these into complete: X, y, yh ?  

# Or look at one by itself...

# Define the 8 variables to use:



if True:

    # Combine all together

    

    # Train, all Xs

    X = df_aug_feats[features].copy()

    

    # ys for Train, start with all -1:

    y = np.zeros(len(X)) - 1

    yh = np.zeros(len(X)) - 1

    yp = np.zeros(len(X)) - 1

    # Use the select_train_24, etc to load the separate ys into y:

    y[select_train_24] = y24

    y[select_train_48] = y48

    y[select_train_72] = y72

    

    yh[select_train_24] = yh24

    yh[select_train_48] = yh48

    yh[select_train_72] = yh72



    yp[select_train_24] = yp24

    yp[select_train_48] = yp48

    yp[select_train_72] = yp72

    



    # Test, all Xs

    Xkag = df_test_feats[features].copy()

    

    # ys for Test, start with all -1:

    y_kag = np.zeros(len(Xkag)) - 1

    yh_kag = np.zeros(len(Xkag)) - 1

    yp_kag = np.zeros(len(Xkag)) - 1

    # Use the select_kag_24, etc to load the separate ys into y:

    y_kag[select_kag_24] = y_kag24

    y_kag[select_kag_48] = y_kag48

    y_kag[select_kag_72] = y_kag72

    

    yh_kag[select_kag_24] = yh_kag24

    yh_kag[select_kag_48] = yh_kag48

    yh_kag[select_kag_72] = yh_kag72



    yp_kag[select_kag_24] = yp_kag24

    yp_kag[select_kag_48] = yp_kag48

    yp_kag[select_kag_72] = yp_kag72

    

else:

    # Use a particular one of the 3 cp_times

    X = X48

    y = y48

    yh = yh48

    yp = yp48

    

    Xkag = Xkag48

    y_kag = y_kag48

    yh_kag = yh_kag48

    yp_kag = yp_kag48

# See how the prediction, yh, compares with the known y values:



roc_area, ysframe = y_yhat_plots(y, yh, title="y and y_score", y_thresh=yh_threshold,

                       plots_prefix=model_name.upper()+"_"+version_str,

                                return_ysframe_too=True)

# Determine the yh_threshold needed for a given precision.

# Can then go to 3rd code cell back, set the threshold to desired value, and repeat from there.

#

prec_thresh = ysframe[ysframe['Precis'] >  0.99 ].Thresh.min()

# Set to 3 decimals and increase by 0.001:

prec_thresh = int(1.0 + 1000.0*prec_thresh)/1000.0

prec_thresh
# Show the log-loss if it is a single-target classification

if (USE_TARG_SUBSET and ("1: " in targ_subset_name)):

    y_ave = sum(y)/len(y)

    log_loss_dumb = log_loss(y, y_ave + np.zeros(len(y)))

    log_loss_y_yh = log_loss(y, yh)

    # These parameters are based on what's used below for a single target:

    adj1 = 0.98

    adj0 = max(0.02*y_ave, (sum(y) - sum(yp))/(len(y) - sum(yp)))

    yh_adj = yp*(adj1-adj0) + adj0

    log_loss_y_yhadj = log_loss(y, yh_adj)

    

    print('\nFor the single target "'+targ_subset_name+'" :')

    print('\n  The dumb-guess log-loss, yh = ave(y), would be',log_loss_dumb)

    print("\n  The ML's yh gives a log-loss of",log_loss_y_yh)

    print('\n  The yh from factors (below) gives a log-loss of',log_loss_y_yhadj,")")

    print('\n  The score improvement, given the '+str(n_targs)+' targets, is:',

             int(1.e5*(log_loss_dumb - log_loss_y_yhadj)/n_targs),"x10^-5   (using the factors value.)")
# Keeping track of some single-target performances            Test - Control

# Precision set to 97% or 95%



#  The tSNE 9:

#      'proteasome_inhibitor', 'nfkb_inhibitor', 'glucocorticoid_receptor_agonist',

#      'raf_inhibitor', 'cdk_inhibitor', 'hmgcr_inhibitor',

#      'egfr_inhibitor', 'hsp_inhibitor', 'tubulin_inhibitor'

#

# For the single target "1: nfkb" :  The score improvement, given the 206 targets, is: 61 x10^-5

# For the single target "1: prot" :  The score improvement, given the 206 targets, is: 67 x10^-5

# For the single target "1: gluc" :  The score improvement, given the 206 targets, is: 26 x10^-5

# For the single target "1: raf_" :  The score improvement, given the 206 targets, is: 22 x10^-5

# For the single target "1: cdk_" :  The score improvement, given the 206 targets, is: 29 x10^-5     31 - 0

# For the single target "1: hmgcr" : The score improvement, given the 206 targets, is: 26 x10^-5

# For the single target "1: egfr" :  The score improvement, given the 206 targets, is: 30 x10^-5     26 - 0

# For the single target "1: hsp_" :  The score improvement, given the 206 targets, is:  8 x10^-5

# For the single target "1: tubu" :  The score improvement, given the 206 targets, is: 24 x10^-5



#         0.02363

# Total - 0.00262

# ~ 0.02110



# [16] Ones y_ave > 0.010, but not in tSNE-9   * not very encouraging *  Need different features?

#     'acetylcholine_receptor_antagonist', 'adrenergic_receptor_agonist', 'adrenergic_receptor_antagonist',

#     'calcium_channel_blocker', 'cyclooxygenase_inhibitor', 'dna_inhibitor', 'dopamine_receptor_antagonist',

#     'flt3_inhibitor', 'glutamate_receptor_antagonist', 'histamine_receptor_antagonist',

#     'kit_inhibitor', 'pdgfr_inhibitor',

#     'phosphodiesterase_inhibitor', 'serotonin_receptor_antagonist', 'sodium_channel_inhibitor'

#

# For the single target "1: acety" :  The score improvement, given the 206 targets, is: 3 x10^-5

# For the single target "1: adren_ago" :  The score improvement, given the 206 targets, is: 4 x10^-5  1 - 1  :(

# For the single target "1: adren_ant" :  The score improvement, given the 206 targets, is: 3 x10^-5

# For the single target "1: calci"  :  The score improvement, given the 206 targets, is: 1 x10^-5

# . . .

# For the single target "1: hista" :  The score improvement, given the 206 targets, is: 13 x10^-5     15 - 3

# . . .

# For the single target "1: sodiu" :  The score improvement, given the 206 targets, is: 0 x10^-5





# [7] Ones that have 'detectable g-vectors' with > 69 counts

# For the single target "1: topoi" :  The score improvement, given the 206 targets, is:  9 x10^-5

# For the single target "1: hdac_" :  The score improvement, given the 206 targets, is:  6 x10^-5

# For the single target "1: mtor_" :  The score improvement, given the 206 targets, is: 13 x10^-5     15 - 0

# For the single target "1: mek_i" :  The score improvement, given the 206 targets, is:  4 x10^-5      0 - 0 

# For the single target "1: pi3k_" :  The score improvement, given the 206 targets, is:  7 x10^-5      0 - 0

# For the single target "1: prote" :  The score improvement, given the 206 targets, is:  2 x10^-5      3 - 0

# For the single target "1: atpas" :  The score improvement, given the 206 targets, is:  2 x10^-5      4 - 0
# Plot c-std vs c-ave and show the ML-selected ids

# Color by the yp=1 (i.e., MoA=0) ones; use color scheme similar to numMoA

colors = (1-yp); colors[0]=2.0



X.plot(x='c-ave',y='c-std',kind='scatter', figsize=(12,7),

                 c=colors, colormap='jet', alpha=0.25, marker='o',s=50,

                 title='Train:  C-std vs C-ave for all non-control sig_ids'+

                  '   Colored by ML MoA>0 (blue, yp=1)')

plt.savefig("C-std_vs_C-ave_ML-color_"+version_str+".png")

plt.show()



# The line-segment ids are the clearest non-MoA ones.
# Show the yh distribution by known notMoA status (similar to confusion dots output)



# Temporarily ... Add 'soft' predictions and ys to the X dataframe:

X['yh_preds'] = yh

X['y_actual'] = y



X.hist('yh_preds', by='y_actual', bins=100, sharex=True, sharey=True, layout=(5,1), figsize=(14,9))

plt.show()



# Remove the added columns:

X = X.drop(['y_actual','yh_preds'],axis=1)
print("There are",len(y_kag), "predictions corresponding to the",len(Xkag), "test ids;  ",

         sum(test_ctls),"of them are controls.")

print("The number of y=1 (i.e., MoA>0) predicted values is  ",int(sum(yp_kag)),

         "  (threshold =",yh_threshold,")")
# Show the yh distribution for Test



# Temporarily ... Add 'soft' predictions to the Test features dataframe:

Xkag['yh_kag'] = yh_kag



# Including the controls:

Xkag.hist('yh_kag', bins=100, sharex=True, sharey=True, layout=(5,1), figsize=(14,9))

plt.show()



# Just the controls:

Xkag[test_ctls].hist('yh_kag', bins=100, sharex=True, sharey=True, layout=(5,1), figsize=(14,9))

plt.show()



# Find the number of Controls that are incorrectly predicted as y=1.

# Count the number above the yh_threshold:

ctls_above = sum(Xkag[test_ctls].yh_kag > yh_threshold)

print("There are", ctls_above, "controls predicted as MoA.\n")



# Remove the added columns:

Xkag = Xkag.drop(['yh_kag'],axis=1)
# Use the submission example as the start of submission

# Read it in fresh just in case...

df_test_targs = pd.read_csv(dat_dir+test_targs)

df_test_targs
# Fill the test_targs with the predicted values.

# Here we start with the average MoAs-per-row value for each target (column):

# these are the aves_targs() values determined from training.

for icol, this_col in enumerate(aves_targs.index):

    df_test_targs[this_col] = aves_targs.values[icol] 

    

# To monitor changes, show the sum over all target values

print("Sum of targets:",sum(df_test_targs.drop(columns='sig_id').sum()))
# Go through the row index values of the test CONTROLS       * Inefficient code *

# and set all of their target values to 0.

for irow in df_test_targs[test_ctls].index:

    df_test_targs.iloc[irow,1:] = np.zeros(206)



print("\nThe number of test Controls is",sum(test_ctls),

      ", out of a total of",len(df_test_targs),"test rows.")



# To monitor changes, show the sum over all target values:

predicted_moas = sum(df_test_targs.drop(columns='sig_id').sum())

print("\n","Sum of targets:",predicted_moas," <-- this is the predicted MoA sum\n")



# With the controls now set to zero, this sum is the predicted total number of MoAs.

# Note that the "actual" number of test MoAs (determined sneakily in v13)

# is about 3125, or about 12.4% higher.

# (We could scale the predictions by 1.124 to get a better score, but that would be 'wrong'.)
# Calculation when all targets are used   (i.e., the case where the subset is all targets)

if USE_TARG_SUBSET == False:

    # Values used to determine the target scaling factors:

    #   M = total number of non-control rows

    #   N = total number of predicted MoA values.

    #   m1 = number of predicted MoA>1 rows <-- we'll subtract the number of controls above threshold.

    # Initially, the sum of the predictions in each row is equal to N/M,

    # which from the train data is: 16844/21948 ~ 0.767.

    # If a y=1 row has one MoA (though could be 2...),

    # then the summed predictions should be increased from N/M to 1.0,

    # a factor of 1 / (N/M) = M/N.

    # The other M-m1 rows have N-m1 MoAs in them, for an average of (N-m1)/(M-m1) MoAs/row.

    # The scaling factor for these rows can also be made as (N-m1)/(M-m1) / (N/M).

    # Summarizing, the scaling factors are:

    #  factor(MoA>0) = M/N

    #  factor(MoA=0) = (M/N)*(N-m1)/(M-m1)

    # Calculating them:

    m_rows = len(df_test_targs) - sum(test_ctls)

    n_moas = predicted_moas

    m1 = sum(yp_kag) - ctls_above

    print("\nUsing M, N, and m1 values of:", m_rows, ",", n_moas, ",", m1,

              "(corrected for",ctls_above,"controls)\n")



    factor_1 = m_rows/n_moas

    factor_0 = (m_rows/n_moas)*(n_moas-m1)/(m_rows-m1)



    print("The factors for Moa>0 and MoA=0 are:", factor_1, ",", factor_0,"\n")



    # Create a (column) vector of the correction factors based on yp_kag:

    scale_factors = yp_kag*factor_1 + (1-yp_kag)*factor_0



    # Go through the df and multiply each column by the scale_factors:

    for this_col in df_test_targs.columns[1:]:

        df_test_targs[this_col] = scale_factors * df_test_targs[this_col]

    

    # To monitor changes, show the sum over all target values:

    print(sum(df_test_targs.drop(columns='sig_id').sum()))
# Calculating factors when a SUBSET of targets is used to select MoA>0 rows

if USE_TARG_SUBSET == True:

    print("\n The [",targ_subset_name,"] subset of",len(targ_subset),

          "Targets was used to classify rows as MoA>0:\n\n", 

          targ_subset,"\n")



    # Values used to determine the target scaling factors:

    print("\nUseful values:\n")

    

    #   M = total number of non-control rows

    m_rows = len(df_test_targs) - sum(test_ctls)

    print("M =", m_rows,"  Number of non-control rows")

    #   m1 = number of detected MoA>0 rows <-- we subtract the number of controls above threshold.

    m1 = sum(yp_kag) - ctls_above

    print("m1 =", m1, "  Number of detected MoA>0 rows (corrected for",ctls_above,"controls)\n")

    

    #   N = prediction for the sum of all MoA values. (from previous calc. above)

    n_moas = predicted_moas

    print("N = ", n_moas, "  Predicted sum of all MoA values")

    #   Nsub = predicted sum of MoA values in the target subset.

    nsub_moas = m_rows * df_test_targs.loc[0,targ_subset].sum()

    print("Nsub = ", nsub_moas,"  Predicted sum of MoAs in the target subset\n")



    print("   Initial dumb-guess values in the target columns:")

    print(df_test_targs.loc[0,targ_subset],"\n\n")

    

    # The current prediction for MoA-per-row in the subset is Nsub/M.

    # The rows identified as MoA>0 will instead have at least 1.0 MoA-per-row in the subset,

    # so the current prediction for these rows should be multiplied by the factor:

    factor_1 = 1.0 / (nsub_moas/m_rows)



    # Because m1 rows have been detected with MoA>0,

    # the remaining M-m1 rows will have a reduced MoA total of Nsub-m1 (or less).

    # So the scale factor for these rows in the target columns is

    factor_0 = (nsub_moas - m1)/(m_rows - m1) / (nsub_moas/m_rows)



    # For the case of "2: prot..." and selecting numSub>1 (i.e. detecting 2 or more MoAs)

    # we expect 2 active MoA in each of the m1 rows; to allow for false positives, etc, instead

    # we increase the factor_1 by 1.7 and in the factor_0 equation 1.7*m1 moas are removed:

    if ("2: pro" in targ_subset_name):

        factor_1 = 1.7 / (nsub_moas/m_rows)

        factor_0 = (nsub_moas - 1.7*m1)/(m_rows - m1) / (nsub_moas/m_rows)

        # guard against negative values

        factor_0 = max(factor_0, 0.05)

    

    # For the case of "1: ..." there is a single target,

    # so expect exactly 1 active MoA in each of the m1 rows - scale it close to 1.0 

    if ("1: " == targ_subset_name[0:3]):

        factor_1 = 0.98 / (nsub_moas/m_rows)

        # Reduce the predictions in the non-m1 rows proportional to the number of m1s found.

        factor_0 = (nsub_moas - m1)/(m_rows - m1) / (nsub_moas/m_rows)

        # Guard against going negative, i.e., when m1 is larger than the total expected,

        # don't go lower than 5% of the dumb-guess rate. (Happens only with proteasome ?)

        factor_0 = max(factor_0, 0.05)

    

    print("In the Subset of columns, the factors for the m1 and not-m1 rows are:\n",

          factor_1, ",", factor_0,"\n")

    

    # Apply these factors to the SUBSET columns

    # Create a (column) vector of the correction factors based on yp_kag:

    scale_factors = yp_kag*factor_1 + (1-yp_kag)*factor_0

    

    # Go through the df and multiply each column IN THE SUBSET by the scale_factors:

    for this_col in targ_subset:

        df_test_targs[this_col] = scale_factors * df_test_targs[this_col]

    

    

    # There are also corrections we can apply to the non-SUBSET columns,

    # both in the m1 rows (reducing their predictions)

    # and in the non-m1 rows (a small increase in the predictions).

    # Leave these out for now, v20, to check the main subset-column effect.

    

    # Non-subset ones in the MoA>0 rows

    # For the MoA>0 rows, it's certain that the number of detected MoAs (1+ per row)

    # is more than the expected number, 0.767 per row. Tempting to set them near 0,

    # instead lets reduce the the non-subset predictions a bunch:

    nonsub_1 = 0.33

    

    # Non-subset ones in the MoA=0 rows (i.e., non-m1)

    # These will increase slightly to match what was lost in the nonsub m1 rows

    nonsub_0 = 1.0+(1.0-nonsub_1)*m1/(m_rows - m1)



    # For the case of "2: prot..." and selecting numSub>1 (i.e. detecting 2 or more MoAs)

    # we decrease nonsub_1 to a much smaller value and nonsub_0 changes appropriately:

    if ("2: pro" in targ_subset_name):

        nonsub_1 = 0.03

        nonsub_0 = 1.0+(1.0-nonsub_1)*m1/(m_rows - m1)  

        

    print("In the Non-subset columns, the factors for the m1 and not-m1 rows are:\n",

          nonsub_1, ",", nonsub_0,"\n")

    

    # Apply these factors to the NOT-in-the-SUBSET columns

    # Create a (column) vector of the correction factors based on yp_kag:

    scale_factors = yp_kag*nonsub_1 + (1-yp_kag)*nonsub_0

    

    # Go through the df and multiply each column NOT-in-the-SUBSET by the scale_factors:

    col_list = list(df_test_targs.columns[1:])

    for sub_col in targ_subset:

        col_list.remove(sub_col)

    #

    for this_col in col_list:

        df_test_targs[this_col] = scale_factors * df_test_targs[this_col]



    

    # To monitor changes, show the sum over all target values:

    print("Sum of targets:",sum(df_test_targs.drop(columns='sig_id').sum()))

    # It stays the same because we've just redistributed the MoA values within the subset.
# Show the df

df_test_targs
# Save the result as the submission

df_test_targs.to_csv("submission.csv",index=False)

# that's all.

##!head -10 submission.csv
##!tail -10 submission.csv
# show/confirm the random seed value

print("Used RANDOM_SEED = {}".format(RANDOM_SEED))