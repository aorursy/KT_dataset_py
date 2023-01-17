LOCATION_KAGGLE = True

verbose_max = 1  # limit verbosity

#

out_dir = "."

version_str = "v34"

SHOW_EDA = True

USE_SPLIT_AVE = True
import numpy as np

import pandas as pd



import matplotlib.pyplot as plt



import os

##print(os.listdir("../input"))



from time import time

from time import strftime
# Read in the training and test data

df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")



# All of the initial Training columns are:

#  ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',

#       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']





# Change/Adjust/Make-new feature columns...





# --- PassengerId: sequential number - ignore.





# --- Survived:  the "target" values

# Add a Survived column to the test df for uniformity:

df_test['Survived'] = -1





# --- Pclass:

# Make one-hot versions

for iclass in [1,2,3]:

    df_train['Pclass_'+str(iclass)] = (df_train['Pclass'] == iclass).astype(int)

    df_test['Pclass_'+str(iclass)] = (df_test['Pclass'] == iclass).astype(int)



    

# --- Name: will be processed in next cells.





# --- Sex: 

# from male/female to 1/0:

##df_train['Sex'] = (df_train['Sex'] == "male").astype(int)

##df_test['Sex'] = (df_test['Sex'] == "male").astype(int)

# From male/female to "one hot" Sex_M, Sex_F:

df_train['Sex_M'] = (df_train['Sex'] == "male").astype(int)

df_train['Sex_F'] = (df_train['Sex'] == "female").astype(int)

#

df_test['Sex_M'] = (df_test['Sex'] == "male").astype(int)

df_test['Sex_F'] = (df_test['Sex'] == "female").astype(int)



    

# --- Age: initially about 20% have NaN for age.

#

# Do a little research:

# Get a df of just the ones with NaN for Age

##df_NaNage = df_train[~(df_train['Age'] == df_train['Age'])]

# Average survival for these:

##df_NaNage['Survived'].mean()

# This gives survival rate of 0.2937 for the age = NaNs,

# compared to 0.38 for all train samples.

# Could use Name information (Mr, Mrs, Miss, Master) to assign an age...

# Or flag the NaN ages by setting age to something like 99:

df_train['Age'].fillna(99.0, inplace=True)

df_test['Age'].fillna(99.0, inplace=True)

# Add a new numeric column flagging that no age was given:

df_train['NoAge'] = 1.0*(df_train['Age'] > 95.0)

df_test['NoAge'] = 1.0*(df_test['Age'] > 95.0)

# but for Logistic Regression better to set to the trains median value(?)

##age_median = df_train['Age'].median()

##df_train['Age'].fillna(age_median, inplace=True)

##df_test['Age'].fillna(age_median, inplace=True)

# Or randomly set the NoAge ones in the range 18 to 40:

for inoage in df_train[df_train['NoAge'] == 1.0].index:

    df_train.loc[inoage,'Age'] = int(18.0+(40.0-18.0)*np.random.rand())

for inoage in df_test[df_test['NoAge'] == 1.0].index:

    df_test.loc[inoage,'Age'] = int(18.0+(40.0-18.0)*np.random.rand())

#

# OK, the Age NaNs are taken care of...

# Make a log(Age):

df_train['Age_log'] = np.log(1.0+df_train['Age'])

df_test['Age_log'] = np.log(1.0+df_test['Age'])

# Make Age_young, Age_old: a ReLU-ish on ends of age

df_train['Age_young'] = (3.0 - df_train['Age_log'])

df_train.loc[df_train['Age_young'] < 0.0, 'Age_young'] = 0.0

df_train['Age_old'] = (df_train['Age_log'] - 3.8)

df_train.loc[df_train['Age_old'] < 0.0, 'Age_old'] = 0.0

# and for test

df_test['Age_young'] = (3.0 - df_test['Age_log'])

df_test.loc[df_test['Age_young'] < 0.0, 'Age_young'] = 0.0

df_test['Age_old'] = (df_test['Age_log'] - 3.8)

df_test.loc[df_test['Age_old'] < 0.0, 'Age_old'] = 0.0





# --- SibSp and Parch :

# Try limiting these to just 0, 1, 2, 3, 4 :

cols = ['SibSp','Parch']

for col in cols:

    df_train.loc[(df_train[col] > 4), col] = 4



    

# --- Ticket: complex string, ignore.





# --- Fare:

#

# One of the Test cases has NaN for the Fare:

# 152	1044	3	Storey, Mr. Thomas	1	60.5	0	0	3701	NaN	ooo	S

# Look at similar ones in test 

##df_test[(df_test['Pclass'] == 3) & (df_test['Embarked'] == 'S')].head(20)

# Set it to 9.2250 based on another 4-digit ticket:

# 5	897	3	Svensson, Mr. Johan Cervin	1	14.0	0	0	7538	9.2250	ooo	S

df_test.loc[152,'Fare'] = 9.2250

#

# Add Fare_zero flag for very low fares:

df_train['Fare_0'] = (df_train['Fare'] < 6.0).astype(int)

df_test['Fare_0'] = (df_test['Fare'] < 6.0).astype(int)

# Set a nominal Fare for the Fare_0 ones, based on the Pclass:

fare_class = np.exp([2.0, 2.4, 3.3])

for pclass in [1,2,3]:

    df_train.loc[(df_train['Pclass'] == pclass) & \

                 (df_train['Fare_0'] == 1), 'Fare'] = fare_class[pclass-1]

    df_test.loc[(df_test['Pclass'] == pclass) & \

                (df_test['Fare_0'] == 1), 'Fare'] = fare_class[pclass-1]

# and make log(Fare):

df_train['Fare_log'] = np.log(df_train['Fare'])

df_test['Fare_log'] = np.log(df_test['Fare'])





# --- Cabin: replace NaNs with "ooo"

#

df_train['Cabin'].fillna("ooo", inplace=True)

df_test['Cabin'].fillna("ooo", inplace=True)

# add a new numeric column flagging no cabin:

df_train['NoCabin'] = 1.0*(df_train['Cabin'] == "ooo")

df_test['NoCabin'] = 1.0*(df_test['Cabin'] == "ooo")





# --- Embarked:

# Two of the train rows have NaN for Embarked - set them to S:

df_train.loc[61,'Embarked'] = 'S'

df_train.loc[829,'Embarked'] = 'S'

#

# Create one-hot versions of Embarked:

df_train['Embark_C'] = (df_train['Embarked'] == "C").astype(int)

df_train['Embark_Q'] = (df_train['Embarked'] == "Q").astype(int)

df_train['Embark_S'] = (df_train['Embarked'] == "S").astype(int)

#

df_test['Embark_C'] = (df_test['Embarked'] == "C").astype(int)

df_test['Embark_Q'] = (df_test['Embarked'] == "Q").astype(int)

df_test['Embark_S'] = (df_test['Embarked'] == "S").astype(int)





# Done with basic features.
# Look at the Name column... all unique names it seems.

##df_train['Name'].value_counts()
# Extract the last name from the Name column

def extract_last(row_in):

    lastname = row_in['Name'].split(',')[0]

    # replace "'", " ", and "-" with nothing (for now)

    ignores = ["'"," ","-"]

    for iggy in ignores:

        lastname = lastname.replace(iggy,"")

    return lastname



# Put the last names in LastName

df_train['LastName'] = df_train.apply(extract_last, axis=1)

df_test['LastName'] = df_test.apply(extract_last, axis=1)
# Look at the LastName column... 9 Andersson the most.

##df_train['LastName'].value_counts()
# Extract Mr, Mrs, Miss, Master from the Name column

def extract_MMMM(row_in):

    # first take from the Comma+Space on

    prefix = row_in['Name'].split(', ')[1]

    # then take before the ".":

    prefix = prefix.split(".")[0]

    # modify ones that are not Mr Mrs Miss Master:

    if not(prefix in ["Mr","Mrs","Miss","Master"]):

        if prefix in ["Rev", "Col", "Major", "Sir", "Jonkheer", "Don", "Capt"]:

            prefix = "Mr"

        elif prefix in ["Ms", "Mme", "the Countess", "Lady", "Dona"]:

            prefix = "Mrs"

        elif prefix in ["Mlle"]:

            prefix = "Miss"

        elif prefix in ["Dr"]:

            # can be a female Dr:

            if row_in['Sex'] == 'female':

                prefix = "Mrs"

            else:

                prefix = "Mr"

    ##print("prefix is -->"+prefix+"<--")

    return prefix



# Put the prefix in a column

df_train['MMMM'] = df_train.apply(extract_MMMM, axis=1)

df_test['MMMM'] = df_test.apply(extract_MMMM, axis=1)
# Look at the MMMM counts

# ( Without any corrections implemented in extract_MMMM,

#   use value_counts to identify prefixs other than Mr Mrs Miss and Master:

#   Alternatives found:

#   Master: (no other versions)

#      Mrs: Ms, Mme, the Countess, Lady, Dona

#     Miss: Mlle

#       Mr: Dr, Rev, Col, Major, Sir, Jonkheer, Don, Capt 

#   incorporate these in extract_MMMM above.)



print("\nTrain:\n")

print(df_train['MMMM'].value_counts())

print("\nTEST:\n")

print(df_test['MMMM'].value_counts())
# Look at the survival of these different prefixes:

df_train[['Survived','MMMM']].groupby('MMMM').mean()
# Create one-hot versions of the prefixes:

for prefix in ["Mr","Master","Mrs","Miss"]:

    df_train['Sex_'+prefix] = (df_train['MMMM'] == prefix).astype(int)

    df_test['Sex_'+prefix] = (df_test['MMMM'] == prefix).astype(int)
# Create some features from the LastName



# Get the length of the LastName

def extract_ln_length(row_in):

    lastname = row_in['LastName']

    return len(lastname)



# Count the vowels in LastName

def extract_ln_vowels(row_in):

    lastname = row_in['LastName']

    num_vs = 0

    vowels_list = ["a","e","i","o","u","y"]

    for char in lastname.lower():

        if char in vowels_list:

            num_vs += 1

    return num_vs



# Put the last name lengths and vowels in LN_Length, LN_Vowels

df_train['LN_Length'] = df_train.apply(extract_ln_length, axis=1)

df_test['LN_Length'] = df_test.apply(extract_ln_length, axis=1)

df_train['LN_Vowels'] = df_train.apply(extract_ln_vowels, axis=1)

df_test['LN_Vowels'] = df_test.apply(extract_ln_vowels, axis=1)

# And the ratio

df_train['LN_Vfrac'] = df_train['LN_Vowels']/df_train['LN_Length']

df_test['LN_Vfrac'] = df_test['LN_Vowels']/df_test['LN_Length']
# Check for NaN's in the data sets

# Go through the columns one at a time

# For Titanic the Age is the most NaN'ed column...



n_train = len(df_train)

n_test = len(df_test)

print("\nChecking for NaNs:\n")

all_ok = True

for col in df_test.columns:

    nona_train = len(df_train[col].dropna(axis=0))

    nanpc_train = 100.0*(n_train-nona_train)/n_train

    nona_test = len(df_test[col].dropna(axis=0))

    nanpc_test = 100.0*(n_test-nona_test)/n_test

    # Only show it if there are NaNs:

    if (nanpc_train + nanpc_test > 0.0):

        print("{:.3f}%  {} OK out of {}".format(nanpc_train, nona_train, n_train), "  "+col)

        print("{:.3f}%  {} OK out of {}".format(nanpc_test, nona_test, n_test), "  "+col)

        all_ok = False

if all_ok:

    print("   All OK - no NaNs found.\n")
# Look at the data now...

df_train.head(6)
# and the Test data

df_test.head(6)
# Compare Train and Test averages

# Using a z-score with standard error based on the number of samples

descr_train = df_train.describe()

descr_test = df_test.describe()

# Number of samples in the test set

n_test = descr_test.loc["count","Age"]

n_train = descr_train.loc["count","Age"]

if SHOW_EDA:

    print("     --column--    z-score      Test Mean     Train Mean")

for col in descr_test.columns:

    ave_test = descr_test.loc["mean",col]

    ave_train = descr_train.loc["mean",col]

    std_train = descr_train.loc["std",col]

    if SHOW_EDA:

        print(col.rjust(15), 

            '{:.4f}'.format((ave_test - ave_train)/

                               (std_train*np.sqrt(1.0/n_test+1.0/n_train))).rjust(10),

            '{:.4f}'.format(ave_test).rjust(14),

            '{:.4f}'.format(ave_train).rjust(14))
df_train.plot.scatter("Age","Fare_log",

                        figsize=(8,5),c='Survived',alpha=0.6,colormap="Set1",colorbar=False)

plt.title("Showing All  (red=Perished)")

plt.show()

prefix = "Mrs"

df_train[df_train['MMMM'] == prefix].plot.scatter("Age","Fare_log",

                        figsize=(8,5),c='Survived',alpha=0.6,colormap="Set1",colorbar=False)

plt.title("Showing only "+prefix+"  (red=Perished)")

plt.show()
df_train.plot.scatter("LN_Length","LN_Vowels",

                        figsize=(8,5),c='Survived',alpha=0.6,colormap="Set1",colorbar=False)

plt.title("Showing All  (red=Perished)")

plt.show()



df_train.plot.scatter("LN_Length","LN_Vfrac",

                        figsize=(8,5),c='Survived',alpha=0.3,colormap="Set1",colorbar=False)

plt.title("Showing All  (red=Perished)")

plt.show()
# Survival vs Age for Females, Males

df_train[df_train['Sex_F'] == 1].plot.scatter('Age_log','Survived')

plt.text(1.8, 0.5,"Females")

plt.show()

df_train[df_train['Sex_M'] == 1].plot.scatter('Age_log','Survived')

plt.text(1.9, 0.5,"Males")

plt.show()
# Try the violin plot - Male, Female categories

# choose a numerical column:

vpcol = "Age" #  or can use "Fare_log"



# Simple histogram as a check

##df_train['Age'].plot.hist(figsize=(8,5), bins=50)



df = df_train

status_str = 'All'



fig, axes = plt.subplots(figsize=(8,5))



axes.violinplot(dataset = [df[df.Survived == 1][vpcol].values,

                           df[df.Survived == 0][vpcol].values],

               positions=[1.0,1.5],

               widths=0.4,

               showmeans=False, showmedians=True, showextrema=False,

               points=1000,

               bw_method=0.1,  # 'scott', 'silverman', or a scalar 

               vert=True)



axes.set_title("Survived           Selection: "+status_str+"            Perished")

axes.yaxis.grid(True)

axes.set_xlabel('')

axes.set_ylabel(vpcol)



plt.show()





df = df_train[df_train['Sex'] == 'male']

status_str = 'Male'



fig, axes = plt.subplots(figsize=(8,5))



axes.violinplot(dataset = [df[df.Survived == 1][vpcol].values,

                           df[df.Survived == 0][vpcol].values],

               positions=[1.0,1.5],

               widths=0.4,

               showmeans=False, showmedians=True, showextrema=False,

               points=1000,

               bw_method=0.1,  # 'scott', 'silverman', or a scalar 

               vert=True)



axes.set_title("Survived           Selection: "+status_str+"            Perished")

axes.yaxis.grid(True)

axes.set_xlabel('')

axes.set_ylabel(vpcol)



plt.show()





df = df_train[df_train['Sex'] == 'female']

status_str = 'Female'



fig, axes = plt.subplots(figsize=(8,5))



axes.violinplot(dataset = [df[df.Survived == 1][vpcol].values,

                           df[df.Survived == 0][vpcol].values],

               positions=[1.0,1.5],

               widths=0.4,

               showmeans=False, showmedians=True, showextrema=False,

               points=1000,

               bw_method=0.1,  # 'scott', 'silverman', or a scalar 

               vert=True)



axes.set_title("Survived           Selection: "+status_str+"            Perished")

axes.yaxis.grid(True)

axes.set_xlabel('')

axes.set_ylabel(vpcol)



plt.show()

# How do Age and Fare depend on Pclass



##df_train.plot.scatter('Age_log','Pclass')

##plt.show()

##df_train.plot.scatter('Fare_log','Pclass')

##plt.show()
# Try the violin plot - Pclass categories

# choose a numerical column:

vpcol = "Age" #"Fare_log" # or can use "Age"





# Select survival status

dfs = df_train[df_train['Survived'] == 1].copy()

dfp = df_train[df_train['Survived'] == 0].copy()



fig, axes = plt.subplots(figsize=(12,5))

axes.violinplot(dataset = [dfs[dfs.Pclass == 1][vpcol].values,

                           dfs[dfs.Pclass == 2][vpcol].values,

                           dfs[dfs.Pclass == 3][vpcol].values,

                        dfp[dfp.Pclass == 1][vpcol].values,

                        dfp[dfp.Pclass == 2][vpcol].values,

                        dfp[dfp.Pclass == 3][vpcol].values],

               positions=[1.5,2.5,3.5,1.0,2.0,3.0],

               widths=0.4,

               showmeans=False, showmedians=True, showextrema=False,

               points=1000,

               bw_method=0.1,  # 'scott', 'silverman', or a scalar 

               vert=True)



axes.set_title("Perish/Survival for Pclasses, vs "+vpcol)

axes.yaxis.grid(False)

persur_str = "Perish                Survive"

axes.set_xlabel(persur_str + 30*" " + persur_str + 30*" " + persur_str)

axes.set_ylabel(vpcol)



plt.show()
# Try the violin plot - Pclass categories

# choose a numerical column:

vpcol = "Fare_log" #"Fare_log" # or can use "Age"





# Select survival status

dfs = df_train[df_train['Survived'] == 1].copy()

dfp = df_train[df_train['Survived'] == 0].copy()



fig, axes = plt.subplots(figsize=(12,5))

axes.violinplot(dataset = [dfs[dfs.Pclass == 1][vpcol].values,

                           dfs[dfs.Pclass == 2][vpcol].values,

                           dfs[dfs.Pclass == 3][vpcol].values,

                        dfp[dfp.Pclass == 1][vpcol].values,

                        dfp[dfp.Pclass == 2][vpcol].values,

                        dfp[dfp.Pclass == 3][vpcol].values],

               positions=[1.5,2.5,3.5,1.0,2.0,3.0],

               widths=0.4,

               showmeans=False, showmedians=True, showextrema=False,

               points=1000,

               bw_method=0.1,  # 'scott', 'silverman', or a scalar 

               vert=True)



axes.set_title("Perish/Survival for Pclasses, vs "+vpcol)

axes.yaxis.grid(False)

persur_str = "Perish                Survive"

axes.set_xlabel(persur_str + 30*" " + persur_str + 30*" " + persur_str)

axes.set_ylabel(vpcol)



plt.show()
# These two Age-based features 'signal' low and high ages,

# probably most useful for simple Logisitic Rgression.

df_train.plot.scatter('Age_log','Age_young')

plt.show()

df_train.plot.scatter('Age_log','Age_old')

plt.show()

# Look at the correlation between the numerical values

corr_df = df_train.corr()

# In particular the correlations with Survived:

print(corr_df.Survived)



# There are a bunch with abs(corr) at/above 0.2:

# [Pclass], [Fare], Pclass_1, Pclass_3, Fare_log, Sex_M,F, NoCabin

# (ones in [ ]s are not used as features since others duplicate them.)
# All Features... the training numeric columns:

all_features = descr_train.columns



# Remove the 'answer' column:

all_features = all_features.drop('Survived')

# and the PassengerId:

all_features = all_features.drop('PassengerId')



# List all of these potential features

print(len(all_features),"All features:")

print(all_features)
# Can look at the value counts of a feature column ...

featnum=0

df_train[all_features[featnum]].value_counts()
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier



from sklearn.metrics import accuracy_score

from sklearn.metrics import make_scorer



from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV



from sklearn.svm import SVC



from sklearn.neural_network import MLPClassifier
# Use this routine to shown how the prediction is doing.

# This routine is taken from the file chirp_roc_lib.py in the github repo at: 

#   https://github.com/dan3dewey/chirp-to-ROC

# Some small modifications have been made here.



def y_yhat_plots(y, yh, title="y and y_score", y_thresh=0.5,

                     ROC=True, plots_prefix=None):

    """Output plots showing how y and y_hat are related:

    the "confusion dots" plot is analogous to the confusion table,

    and the standard ROC plot with its AOC value.

    The yp=1 threshold can be changed with the y_thresh parameter.

    y and yh are numpy arrays (not series or dataframe.)

    """

    # The predicted y value with threshold = y_thresh

    y_pred = 1.0 * (yh > y_thresh)



    # Show table of actual and predicted counts

    crosstab = pd.crosstab(y, y_pred, rownames=[

                           'Actual'], colnames=['  Predicted'])

    print("\nConfusion matrix (y_thresh={:.3f}):\n\n".format(y_thresh),

        crosstab)



    # Calculate the various metrics and rates

    tn = crosstab[0][0]

    fp = crosstab[1][0]

    fn = crosstab[0][1]

    tp = crosstab[1][1]



    ##print(" tn =",tn)

    ##print(" fp =",fp)

    ##print(" fn =",fn)

    ##print(" tp =",tp)



    this_fpr = fp / (fp + tn)

    this_fnr = fn / (fn + tp)



    this_recall = tp / (tp + fn)

    this_precision = tp / (tp + fp)

    this_accur = (tp + tn) / (tp + fn + fp + tn)



    this_posfrac = (tp + fn) / (tp + fn + fp + tn)



    print("\nResults:\n")

    print(" False Pos = ", 100.0 * this_fpr, "%")

    print(" False Neg = ", 100.0 * this_fnr, "%")

    print("    Recall = ", 100.0 * this_recall, "%")

    print(" Precision = ", 100.0 * this_precision, "%")

    print("\n    Accuracy = ", 100.0 * this_accur, "%")

    print(" Pos. fract. = ", 100.0 * this_posfrac, "%")



    # Put them in a dataframe for plots and ROC

    # Reduce the number if very large:

    if len(y) > 100000:

        reduce_by = int(0.5+len(y)/60000)

        print("\nUsing 1/{} of the points for dots and ROC plots.".format(reduce_by))

        ysframe = pd.DataFrame([y[0: :reduce_by], yh[0: :reduce_by], 

                                y_pred[0: :reduce_by]], index=[

                           'y', 'y-hat', 'y-pred']).transpose()

        plot_alpha = 0.3

    else:

        ysframe = pd.DataFrame([y, yh, y_pred], index=[

                           'y', 'y-hat', 'y-pred']).transpose()

        plot_alpha = 0.7



    # If the yh is discrete (0 and 1s only) then blur it a bit

    # for a better visual dots plot

    if min(abs(yh - 0.5)) > 0.49:

        ysframe["y-hat"] = (0.51 * ysframe["y-hat"]

                            + 0.49 * np.random.rand(len(yh)))



    # Make a "confusion dots" plot

    # Add a blurred y column

    ysframe['y (blurred)'] = ysframe['y'] + 0.1 * np.random.randn(len(ysframe))



    # Plot the real y (blurred) vs the predicted probability

    # Note the flipped ylim values.

    ysframe.plot.scatter('y-hat', 'y (blurred)', figsize=(12, 5),

                         s=2, xlim=(0.0, 1.0), ylim=(1.8, -0.8), alpha=plot_alpha)

    # show the "correct" locations on the plot

    plt.plot([0.0, y_thresh], [0.0, 0.0], '-',

        color='green', linewidth=5)

    plt.plot([y_thresh, y_thresh], [0.0, 1.0], '-',

        color='gray', linewidth=2)

    plt.plot([y_thresh, 1.0], [1.0, 1.0], '-',

        color='green', linewidth=5)

    plt.title("Confusion-dots Plot: " + title, fontsize=16)

    # some labels

    ythr2 = y_thresh/2.0

    plt.text(ythr2 - 0.03, 1.52, "FN", fontsize=16, color='red')

    plt.text(ythr2 + 0.5 - 0.03, 1.52, "TP", fontsize=16, color='green')

    plt.text(ythr2 - 0.03, -0.50, "TN", fontsize=16, color='green')

    plt.text(ythr2 + 0.5 - 0.03, -0.50, "FP", fontsize=16, color='red')



    if plots_prefix != None:

        plt.savefig(plots_prefix+"_dots.png")

    plt.show()



    # Go on to calculate and plot the ROC?

    if ROC == False:

        return 0





    # Make the ROC curve

    #

    # Set the y-hat as the index and sort on it

    ysframe = ysframe.set_index('y-hat').sort_index()

    # Put y-hat back as a column (but the sorting remains)

    ysframe = ysframe.reset_index()



    # Initialize the counts for threshold = 0

    p_thresh = 0

    FN = 0

    TN = 0

    TP = sum(ysframe['y'])

    FP = len(ysframe) - TP



    # Assemble the fpr and recall values

    recall = []

    fpr = []

    # Go through each sample in y-hat order,

    # advancing the threshold and adjusting the counts

    for iprob in range(len(ysframe['y-hat'])):

        p_thresh = ysframe.iloc[iprob]['y-hat']

        if ysframe.iloc[iprob]['y'] == 0:

            FP -= 1

            TN += 1

        else:

            TP -= 1

            FN += 1

        # Recall and FPR:

        recall.append(TP / (TP + FN))

        fpr.append(FP / (FP + TN))



    # Put recall and fpr in the dataframe

    ysframe['Recall'] = recall

    ysframe['FPR'] = fpr



    # - - - ROC - - - could be separate routine

    zoom_in = False



    # Calculate the area under the ROC

    roc_area = 0.0

    for ifpr in range(1, len(fpr)):

        # add on the bit of area (note sign change, going from high fpr to low)

        roc_area += 0.5 * (recall[ifpr] + recall[ifpr - 1]

                           ) * (fpr[ifpr - 1] - fpr[ifpr])



    plt.figure(figsize=(8, 8))

    plt.title("ROC: " + title, size=16)

    plt.plot(fpr, recall, '-b')

    # Set the scales

    if zoom_in:

        plt.xlim(0.0, 0.10)

        plt.ylim(0.0, 0.50)

    else:

        # full range:

        plt.xlim(0.0, 1.0)

        plt.ylim(0.0, 1.0)



    # The reference line

    plt.plot([0., 1.], [0., 1.], '--', color='orange')



    # The point at the y_hat = y_tresh threshold

    if True:

        plt.plot([this_fpr], [this_recall], 'o', c='blue', markersize=15)

        plt.xlabel('False Postive Rate', size=16)

        plt.ylabel('Recall', size=16)

        plt.annotate('y_hat = {:.2f}'.format(y_thresh),

                            xy=(this_fpr+0.01 + 0.015,

                            this_recall), size=14, color='blue')

        plt.annotate(' Pos.Fraction = ' +

                        '  {:.0f}%'.format(100 * this_posfrac),

                        xy=(this_fpr + 0.03, this_recall - 0.045),

                        size=14, color='blue')



    # Show the ROC area (shows on zoomed-out plot)

    plt.annotate('ROC Area = ' + str(roc_area)

                 [:5], xy=(0.4, 0.1), size=16, color='blue')



    # Show the plot

    if plots_prefix != None:

        plt.savefig(plots_prefix+"_ROC.png")

    plt.show()



    return roc_area  # or ysframe
# Get X,y from dataframe

def get_Xy_values(df_in, features):

    # Extract and return the features, X dataframe, and target values, y (np.array).



    X = df_in[features].copy()

    y = df_in.Survived.values



    print("\nThe y target has {} values.\n".format(len(y)))

    return X, y
##all_features
# Select which ones to use from all available:

# ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Pclass_1', 'Pclass_2',

#       'Pclass_3', 'Sex_M', 'Sex_F', 'NoAge', 'Age_log', 'Age_young',

#       'Age_old', 'Fare_0', 'Fare_log', 'NoCabin', 'Embark_C', 'Embark_Q',

#       'Embark_S', 'Sex_Mr', 'Sex_Master', 'Sex_Mrs', 'Sex_Miss',

#       'LN_Length', 'LN_Vowels', 'LN_Vfrac']



# - Exclude Pclass, Fare and Age (but include other versions of them).

# - Use Sex_Mr,etc instead of Sex_M,F.

# - Add LN_Length,Vfrac   (Leave out LN_Vowels)

# - Include Fare_log ? or not ?

features = [        'SibSp', 'Parch',          \

            'Pclass_1', 'Pclass_2', 'Pclass_3','NoCabin', \

            ##'NoAge', 'Age_log', 'Age_young', 'Age_old', 'Fare_0', \

            'NoAge', 'Age_log', 'Age_young', 'Age_old', 'Fare_0', 'Fare_log', \

            'Embark_C', 'Embark_Q', 'Embark_S', #'Sex_M', 'Sex_F']

            'Sex_Mr', 'Sex_Master', 'Sex_Mrs', 'Sex_Miss',

            'LN_Length', 'LN_Vfrac']

# List the selected features

print(len(features),"Selected features:")

print(features)
print("Training:")

X, y = get_Xy_values(df_train, features)



# Get the Kaggle test set (Note: y_kag is not valid)

print("Kaggle Test:")

Xkag, y_kag = get_Xy_values(df_test, features)
# Offset, Scale all features so that Train features have mean 0.0 and standard deviation 1.0:

for col in X.columns:

    col_mean = X[col].mean()

    col_std = X[col].std()

    ##print(col_mean, col_std)

    # X

    X[col] = (X[col] - col_mean)/col_std

    # Xkag

    Xkag[col] = (Xkag[col] - col_mean)/col_std
# Select model to use:  lgr, dtc, rfc, gbc, svc, mlp, xgb



model_name = "xgb"

# LogisticRegression(

#  penalty=’l2’, dual=False, tol=0.0001, C=1.0,

# fit_intercept=True, intercept_scaling=1, class_weight=None,

# random_state=None, solver=’warn’, max_iter=100, multi_class=’warn’,

# verbose=0, warm_start=False, n_jobs=None)



lgr_params = {'tol': 0.00001,

              'C': 0.04,

              'solver': 'sag',

              'max_iter': 10000,

              'multi_class': 'ovr',

              'verbose': 1}

# Do each value once since the fitting is not very random

lgr_param_grid = {'tol': [0.00001],

                  'C': [0.005, 0.008, 0.01, 0.012, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, \

                        0.045, 0.05, 0.06, 0.07, 0.10]}

# Decision Tree Classifier



# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html



# DecisionTreeClassifier(

# criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1,

# min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,

# min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)



dtc_params = {'max_depth': 3,

                'min_samples_leaf': 2,

                'min_impurity_decrease': 0.001

             }



##dtc_param_grid = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9],

##                'min_samples_leaf': [2, 3, 4, 5, 7, 9],

##                'min_impurity_decrease': [0.001]

##             }



dtc_param_grid = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],

                'min_samples_leaf': [2, 3, 4, 5, 6, 7, 8, 9, 10],

                'min_impurity_decrease': [0.001]

             }
# Random Forest Classifier



# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html



# RandomForestClassifier(

# n_estimators=’warn’, criterion=’gini’, max_depth=None, min_samples_split=2, min_samples_leaf=1,

# min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0,

# min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0,

# warm_start=False, class_weight=None)





rfc_params = {'n_estimators': 200,

              'max_depth': 5,

              'min_samples_leaf': 2,

              'min_impurity_decrease': 0.001

             }



##rfc_param_grid = {'max_depth': [4, 5, 6, 7, 8, 9, 10, 11, 12],

##                  'min_samples_leaf': [2, 3, 4, 5, 6, 7, 8],

##                  'min_impurity_decrease': [0.001]

##                 }



# Use max_depth = 5 and min_samples_leaf = 2;

# Run these same parameters many times to see spread

rfc_param_grid = [

    {'max_depth': [5, 5],

                  'min_samples_leaf': [2, 2],

                  'min_impurity_decrease': [0.001]

                 },

    {'max_depth': [5, 5],

                  'min_samples_leaf': [2, 2],

                  'min_impurity_decrease': [0.001]

                 },

    {'max_depth': [5, 5],

                  'min_samples_leaf': [2, 2],

                  'min_impurity_decrease': [0.001]

                 },

    {'max_depth': [5, 5],

                  'min_samples_leaf': [2, 2],

                  'min_impurity_decrease': [0.001]

                 }

]

# Gradient Boosting Classifier



# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html



# GradientBoostingClassifier(

#      loss=’deviance’,

#      learning_rate=0.1,

#      n_estimators=100,

#      max_depth=3,

#

#      max_features=None,

#      min_impurity_decrease=0.0,

#      min_samples_leaf=1,

#      min_samples_split=2,

#      subsample=1.0,

#

#      n_iter_no_change=None,

#      validation_fraction=0.1, tol=0.0001,

#      verbose=0,

#

#      criterion=’friedman_mse’, 

#      min_weight_fraction_leaf=0.0,

#      min_impurity_split=None, init=None, random_state=None,  

#      max_leaf_nodes=None, warm_start=False, presort=’auto’, 

#      )



# Parameters for this model

#

# possible loss functions:

# ‘deviance’ refers to deviance (= logistic regression) for classification with probabilistic outputs. 

#

# Other parameters for this model.   #***** are ones to focus on tuning...

# Values here are updated to be the current 'best' ones.

gbc_params = {

          'learning_rate': 0.013, # Smaller better but n_estimators grows

          'n_estimators': 400,   # Early stopping will limit this, so just set a large value.

          #

          'max_depth': 4,      #***** Keep small to reduce overfitting

          #

          'max_features': None,      #***** <1.0 reduces variance and increases bias

          'min_impurity_decrease': 0.003,   #*****

          'min_samples_leaf': 20,         # *****

          'min_samples_split': 85,      #*****

          'subsample': 0.80,            #***** less than 1.0 to reduce variance, increase bias

          # early stopping:      

          # allows not tuning the n_estimators parameter

          'n_iter_no_change': 30,

          'tol': 0.00003,

          'validation_fraction': 0.15, 'tol': 0.0005,

          'verbose': 0

          }



# Setup hyper-parameter grid for the model:

gbc_param_grid = [

    {

              'min_impurity_decrease': [0.003],

              'min_samples_leaf': [20, 20],

              'min_samples_split': [85, 85],

              'subsample': [0.80]},

    {

              'min_impurity_decrease': [0.003],

              'min_samples_leaf': [20, 20],

              'min_samples_split': [85, 85],

              'subsample': [0.80]}

]
# Support Vector Classification - SVC



# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html



# SVC(

# C=1.0, kernel=’rbf’, degree=3, gamma=’auto_deprecated’, coef0=0.0,

# shrinking=True, probability=False, tol=0.001, cache_size=200,

# class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)

    

svc_params = {'kernel': 'rbf',  # one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ 

              'degree': 3,       # for 'poly' only

              'coef0': 0.0,      # poly, sigmoid

              'tol': 0.0003,      # stopping critereon

              #

              'C': 2.5,           # Penalty parameter C of the error term

              'gamma': 0.060,     # auto gives 1/n_features

              #

              'shrinking': True,  # use the shrinking heuristic

              'probability': True, # allows proba values

              'cache_size': 200,   # kernel cache (in MB)

              'class_weight': 'balanced',  # the values of y to automatically adjust weights 

              'verbose': True,

              'max_iter': 100000,

              'decision_function_shape': 'ovr'  # same as other classifiers

             }



# Scan some values... 

# poly degree 3: C=2.5, gamma=0.030

svc_param_grid = {'C': [2.4, 2.5, 2.6],

    'gamma': [0.029, 0.030, 0.031]

    }

# rbf            C=3.5, gamma=0.025

##svc_param_grid = {'C': [3.4, 3.5, 3.6],

##    'gamma': [0.024, 0.025, 0.026]

##    }
# Neural Network Classifier



# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html



# MLPClassifier(

# hidden_layer_sizes=(100, ), activation=’relu’, solver=’adam’, alpha=0.0001,

# batch_size=’auto’, learning_rate=’constant’, learning_rate_init=0.001, power_t=0.5,

# max_iter=200, shuffle=True, random_state=None, tol=0.0001,

# verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,

# early_stopping=False, validation_fraction=0.1,

# beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10

              

valid_fraction = 0.15

    

mlp_params = {'hidden_layer_sizes': (14,10,8),

              'alpha': 0.60,   # L2 regularization param

              'learning_rate_init': 0.09,

              # less often changed parameters:

              'batch_size': 35,  # bit better than auto=200

              'momentum': 0.50,  # for sgd; 0.95 more erratic

              #

              'activation': 'relu',

              'solver': 'sgd',

              'learning_rate': 'constant',

              'max_iter': 500, # number of epochs, uses < 100 for (12,4)

              'tol': 0.00003,

              'n_iter_no_change': 30,  # number of epochs

              'early_stopping': True,

              'validation_fraction': valid_fraction,

              #

              'verbose': False

             }



# Scan some values:

# ~ 0.6, 0.09, 0.5 best for (14,10,8)

mlp_param_grid = [

    {'alpha': [0.60, 0.61],  # L2 regularization param

     'learning_rate_init': [0.090, 0.091],

     'momentum': [0.50, 0.51],

    }

]
# eXtreme Gradient Boost classifier



from xgboost import XGBClassifier



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

    

xgb_params = {

        "max_depth"        : 4,

        "learning_rate"    : 0.08,

        "n_estimators"     : 75,

        "min_child_weight" : 4,

        "gamma"            : 1.5,

        "colsample_bytree" : 0.40,

        "subsample"        : 1.0,

        "reg_lambda"       : 1.0,

    #

        "objective": "binary:logistic",

        "base_score" : 0.50,

        "verbosity" : 1

     }



# Setup hyper-parameter grid for the model:

xgb_param_grid = [

    {

        ##"max_depth"        : [5, 7, 9],  # fix at 8 or 4

        "learning_rate"    : [0.01, 0.03, 0.08],   # <-- try different rates; v34 used 0.08

        "n_estimators"     : np.array(range(5,300,10))  # <-- scan values; v34 used 75

        ##"min_child_weight" : [2, 4, 6],   #

        ##"gamma"            : [0.5, 1.5, 3.0, 4.5],  #

        ##"colsample_bytree" : [0.14, 0.20, 0.28, 0.40] #

        ##"subsample"        : [0.8, 1.0],      # keep value of 1

        ##"reg_lambda"       : [0.1, 1.0, 10.0] # keep at 1

     }

]
# Choose the selected model



if model_name == 'lgr':

    model_base = LogisticRegression(**lgr_params)

    param_grid = lgr_param_grid



if model_name == 'dtc':

    model_base = DecisionTreeClassifier(**dtc_params)

    param_grid = dtc_param_grid

    

if model_name == 'rfc':

    model_base = RandomForestClassifier(**rfc_params)

    param_grid = rfc_param_grid

    

if model_name == 'gbc':

    model_base = GradientBoostingClassifier(**gbc_params)

    param_grid = gbc_param_grid

    

if model_name == 'svc':

    model_base = SVC(**svc_params)

    param_grid = svc_param_grid

    

if model_name == 'mlp':

    model_base = MLPClassifier(**mlp_params)

    param_grid = mlp_param_grid

    

if model_name == 'xgb':

    model_base = XGBClassifier(**xgb_params)

    param_grid = xgb_param_grid
# Doing this fit here lets us skip over the Hyper-Parameter Search and continue

if True:

    best_fit_model = model_base.fit(X,y)

    # Show these parameters

    print(best_fit_model.get_params())

    # Also define cv_folds, gscv_stats incase the following is skipped:

    cv_folds = 10

    gscv_stats = []
# Define scoring function(s)



# Demo of make_scorer with simple example:

acc_scorer = make_scorer(accuracy_score, greater_is_better=True)



gscv_scorer = acc_scorer

scorer_name = 'ACC'
# Set the CV parameter or method:

#

# Use KFold with chosen number of folds: 

cv_folds = 10

cv_param = cv_folds
# Do the Grid Search



print("\nDoing Grid Search on model: "+model_name+".\n")



# Select number of CPUs for GSCV depending if on Kaggle or not:

if LOCATION_KAGGLE:

    number_cpus = -1

else:

    number_cpus = -2



#GridSearchCV(estimator, param_grid, scoring=None,

#             fit_params=None,

#             n_jobs=None, iid=’warn’, refit=True, cv=’warn’,

#             verbose=0,

#             pre_dispatch=‘2*n_jobs’, error_score=’raise-deprecating’,

#             return_train_score=’warn’)



# param_grid is:

# Dictionary with parameters names (string) as keys and lists of parameter settings to try as values,

# or a list of such dictionaries.

    

gscv = GridSearchCV(model_base, param_grid,

            scoring=gscv_scorer,      #

            n_jobs=number_cpus, #  -2: all CPUs but one are used

            iid=False,  # independent identical distrib., "agrees with standard defn of CV"

            cv=cv_param,

            refit=True,

            verbose=min(verbose_max,2),

            return_train_score=True)



t0 = time()



_dummy = gscv.fit(X, y)

# Get the best model

best_fit_model = gscv.best_estimator_



# Fit it to all the training data - This was already done by the refit=True

##best_fit_model.fit(X, y)



if (model_name == 'gbc'):

    print("\nGSCV Fitting took {:.1f} minutes. (Final fit took {} iterations.)\n".format(

                                    (time()-t0)/60.0, len(best_fit_model.train_score_)))

elif (model_name == 'mlp'):

    print("\nGSCV Fitting took {:.1f} minutes. (Final fit took {} iterations.)\n".format(

                                    (time()-t0)/60.0, best_fit_model.n_iter_))

else:

    print("\nGSCV Fitting took {:.1f} minutes.\n".format((time()-t0)/60.0))

# The GSCV results are given by the python dictionary:

##gscv.cv_results_



# Here the TEST refers to the out-of-fold/validation data in the CV process.

# Put the grid search results in a pandas dataframe

# Sort by a value, e.g, the test score which is -1*MSE

df_gscv = pd.DataFrame.from_dict(gscv.cv_results_).sort_values(by='mean_test_score',ascending=False)



# Put the original order in an "index" column:

df_gscv = df_gscv.reset_index()



# form the statistics of the columns

gscv_stats = df_gscv.describe()
# Look into the std values given by the GSCV output...

if False:

    # The average std of the test scores is given as:

    print("\nAverage std_test_score = {:.4f}".format(gscv_stats.loc['mean','std_test_score']))

    print("This is much larger than the range of the mean_test_score s (further below.)")



    # The error bars are dominated by the differences in score from split-to-split:

    print("\nThere is large (but consistent) variation between the splits:")

    for itest in range(3):   # len(df_gscv)):

        for isplit in range(cv_folds):

            print("split {}: score = {:.4f}".format(isplit, df_gscv.loc[itest,'split'+str(isplit)+'_test_score']))

        print("mean test score = {:.4f}".format(df_gscv.loc[itest,'mean_test_score']))

        print(" - - -")

    print(" etc")



    # This is much larger than the variation of the same split due to fitting/parameter changes:

    print("\nIn contrast the variation within a single split is smaller:")

    for itest in range(6):   # len(df_gscv)):

        for isplit in [0]:

            print("split {}: score = {:.4f}".format(isplit, df_gscv.loc[itest,'split'+str(isplit)+'_test_score']))

    print(" etc\n")

    for itest in range(6):   # len(df_gscv)):

        for isplit in [4]:

            print("split {}: score = {:.4f}".format(isplit, df_gscv.loc[itest,'split'+str(isplit)+'_test_score']))

    print(" etc\n")



# Calculate the std of the each-split's scores,

# and the average std of a split score.

##print("\nScore random variation is estimated from the std of each split:")

std_split = 0.0

for isplit in range(cv_folds):

    ##print("std split{} = {:.4f}".format(isplit,gscv_stats.loc['std','split'+str(isplit)+'_test_score']))

    std_split += (gscv_stats.loc['std','split'+str(isplit)+'_test_score'])**2

std_split = np.sqrt(std_split/cv_folds)



print("The average standard deviation of test-split scores is {:.4f}".format(std_split))

# The sterr expected from the split std is then:

sterr_splits = std_split/np.sqrt(cv_folds)

print("Hence, the expected sterr of the test scores is {:.4f}".format(sterr_splits))





# Add a sterr_test_score column

# not using this:

##df_gscv['sterr_test_score'] = df_gscv['std_test_score']/np.sqrt(cv_folds)

# but using the sterr_splits instead:

df_gscv['sterr_test_score'] = sterr_splits
# Save the dataframe to a file (in out_dir) - or not:

##timestr = strftime("%m-%d-%y_%H-%M")

# include the model name too

##df_gscv.to_csv(out_dir+"/GSCV_" + model_name +

##               "_{}_{}.csv".format(timestr, version_str), header=True, index=True)



# Show all the rows

##df_gscv

# Show the first some number of rows

df_gscv[0:min([6,len(df_gscv)])]
# Update the gscv_stats variable with the new sterr column

gscv_stats = df_gscv.describe()



print("\nMean test score Min/50%/75%/Max: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(

                gscv_stats.loc['min','mean_test_score'],

                gscv_stats.loc['50%','mean_test_score'], gscv_stats.loc['75%','mean_test_score'],

                gscv_stats.loc['max','mean_test_score']))

print("\nRange of mean_test_score = {:.4f}, sterr_test_score = {:.4f}".format(

                gscv_stats.loc['max','mean_test_score'] - gscv_stats.loc['min','mean_test_score'],

                gscv_stats.loc['50%','sterr_test_score']))

##gscv_stats
# Get a list of all the parameters that were scanned, the keys from the dicts(s):

if type(param_grid) ==  type({1:2}):

    # it's a dictionary

    param_keys = list(param_grid.keys())

else:

    # it's a list of dictionaries

    param_keys = []

    for pdict in param_grid:

        for key in pdict.keys():

            param_keys.append(key)

# make it a sorted, unique list

param_keys = list(set(param_keys))

param_keys.sort()

print(param_keys)
# Show test score with error bars vs the GS original order (index)

# Can color-code points by one of the parameters (or not.)

# Pic a param

this_param = param_keys[0]

df_gscv.plot.scatter('index', 'mean_test_score', yerr='sterr_test_score', figsize=(15,4),

                        title='Test Score ('+scorer_name+') vs Grid Search index',

                        c="param_"+this_param, colormap='plasma')

plt.show()
# Make plots of the test values vs param values for the params

num_param_keys = len(param_keys)

fig, axes = plt.subplots(1,num_param_keys,sharey=True,figsize=(15,5))

for iparam, this_param in enumerate(param_keys):

    # Why is this needed to get scatter to work?!?

    df_gscv["param_"+this_param] = df_gscv["param_"+this_param].astype(float)

    if num_param_keys > 1:

        ax = axes[iparam]

    else:

        ax = axes

    # without or with the error bars

    ##df_gscv.plot.scatter("param_"+this_param,'mean_test_score',ax=ax)

    df_gscv.plot.scatter("param_"+this_param,'mean_test_score',ax=ax, yerr='sterr_test_score',

                         # include this to get better scaling for small values...

                         xlim=(0.000,1.1*max(df_gscv["param_"+this_param])))

    

plt.show()
# Show the test score vs train score with color-code by parameter values (if useful)

if True:

    for this_param in param_keys:

        df_gscv.plot.scatter('mean_train_score','mean_test_score',

                         c="param_"+this_param, colormap='plasma',

                         sharex=True, figsize=(15,4), yerr='sterr_test_score')



    plt.show()
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

    
best_fit_model.get_params()
# Make the model probability predictions on the Training and Test (Kaggle) data



# Just use the final best-fit model fit on all training data

best_fit_model.fit(X,y)



# Fit the model on all the Training data

print("")

all_train_score = accuracy_score(y, best_fit_model.predict(X))

print("Nominal best-fit All-Train accuracy: {:.2f} %\n".format(100.0*all_train_score))



# The probabilty values, 0 to 1

yh = best_fit_model.predict_proba(X)

yh = yh[:,1]



# Make the Kaggle set predictions too

yh_kag = best_fit_model.predict_proba(Xkag)

yh_kag = yh_kag[:,1]



# Or ...

if USE_SPLIT_AVE:

    # The CV fitting above was done on (k-1)/k of the data,

    # do that to evaluate the final model as well.

    # Use the average of the split-predicted proba values.

    # (Or use median? any/much difference?)

    print("Doing (k-1)/k fits:")

    yh = 0.0 * yh

    yh_kag = 0.0 * yh_kag

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=False)

    for train_index, test_index in skf.split(X, y):

        X_split, X_dummy = X.loc[train_index], X.loc[test_index]

        y_split, y_dummy = y[train_index], y[test_index]

        # fit the model on the (k-1)/k of the data

        best_fit_model.fit(X_split, y_split)

        # FYI, accuracy of this model applied to whole dataset

        print("   split --> {}".format(accuracy_score(y, best_fit_model.predict(X))))

        # accumulate and average the probabilty values:

        yh_split = best_fit_model.predict_proba(X)

        yh += (yh_split[:,1])/cv_folds

        yh_kag_split = best_fit_model.predict_proba(Xkag)

        yh_kag += (yh_kag_split[:,1])/cv_folds





# yh and yh_kag are the model probability predictions.

# Convert to discrete 0,1 using a threshold:

#

yh_threshold = 0.45

#

# Training:

yp = 1.0*(yh > yh_threshold)

# Test (Kaggle):

yp_kag = 1.0*(yh_kag > yh_threshold)





print("")

ave_train_score = accuracy_score(y, yp)

print("Split-Train accuracy: {:.2f} %\n".format(100.0*ave_train_score))
# Show the yh distribution by known Survival



# Temporarily ... Put the model speed and ys in the X dataframe:

X['preds'] = yh

X['Survived'] = y



X.hist('preds', by='Survived', bins=100, sharex=True, sharey=True, layout=(5,1), figsize=(14,9))

plt.show()



# Remove the added columns:

X = X.drop(['Survived','preds'],axis=1)
# See how the prediction, yh, compares with the known y values:

roc_area = y_yhat_plots(y, yh, title="y and y_score", y_thresh=yh_threshold,

                       plots_prefix=model_name.upper()+"_"+version_str)
# Look at the errors made



# FP: Predicted Survival but did not survive:

FP_indices = (yh > yh_threshold) & (y < 0.5)

df_FPs = df_train[FP_indices]



# FN: Predicted not to Survive but did survive:

FN_indices = (yh < yh_threshold) & (y > 0.5)

df_FNs = df_train[FN_indices]



print("\n{}: {} FPs and {} FNs\n".format(model_name.upper(),len(df_FPs),len(df_FNs)))
# List FPs: predicted to Survive, but Perished:    mostly females

#

orig_cols = df_FPs.columns

df_FPs[orig_cols[0:12]]
# List FNs: predicted to Perish, but Survived:    mostly males

#

orig_cols = df_FNs.columns

df_FNs[orig_cols[0:12]]
# Histogram of the model's predicted values



fixed_bins = 0.0 + np.array(range(50+1))/(5*10.0)



plt.figure(figsize=(15, 6))

plt.subplot(2, 1, 1)

plt.hist([yh]+[-1.0]+[5.0], bins=fixed_bins, histtype='stepfilled')

plt.title("Train: Prediction Values")

plt.show()
# Summarize the fitting results

if len(gscv_stats) > 0:

    print("\n"+model_name.upper()+

              ":  Train = {:.3f}, {} FPs, {} FNs, AUC={:.3f}   GSCV: Test = {:.3f}, Train = {:.3f}\n".format(

            accuracy_score(y, yp), len(df_FPs), len(df_FNs), roc_area,

            gscv_stats.loc['max','mean_test_score'],

            gscv_stats.loc['max','mean_train_score']))

else:

    print("\n"+model_name.upper()+

              ":  Train = {:.3f}, {} FPs, {} FNs, AUC={:.3f}\n".format(

            accuracy_score(y, yp), len(df_FPs), len(df_FNs), roc_area))





# with Sex_M,F:

#   LGR: Train = 0.827, 54 FPs, 100 FNs, AUC=0.866   GSCV:Test = 0.818, Train = 0.825

#   MLP: Train = 0.842, 44 FPs,  97 FNs, AUC=0.885   GSCV:Test = 0.818, Train = 0.837

#   GBC: Train = 0.862, 38 FPs,  85 FNs, AUC=0.908   GSCV:Test = 0.834, Train = 0.860

# with Sex_Mr,etc:

#   LGR: Train = 0.837, 59 FPs, 86 FNs, AUC=0.879   GSCV:Test = 0.828, Train = 0.838

#   SVC: Train = 0.843, 77 FPs, 63 FNs, AUC=0.903   GSCV:Test = 0.819, Train = 0.839

#   MLP: Train = 0.848, 37 FPs, 98 FNs, AUC=0.896   GSCV:Test = 0.830, Train = 0.849

#        Train = 0.846, 38 FPs, 99 FNs, AUC=0.899   GSCV:Test = 0.828, Train = 0.847

#   GBC: Train = 0.862, 42 FPs, 81 FNs, AUC=0.917   GSCV:Test = 0.835, Train = 0.862

#        Train = 0.865, 42 FPs, 78 FNs, AUC=0.918   GSCV:Test = 0.841, Train = 0.865

#        Train = 0.865, 40 FPs, 80 FNs, AUC=0.919   GSCV:Test = 0.844, Train = 0.865

#

# with LN_Length,LN_Vfrac (not LN_Vowels):

#   LGR:  Train = 0.835, 67 FPs, 80 FNs, AUC=0.881   GSCV:Test = 0.832, Train = 0.841

#   SVC:  Train = 0.851, 72 FPs, 61 FNs, AUC=0.920   GSCV:Test = 0.819, Train = 0.851

#   MLP:  Train = 0.860, 37 FPs, 88 FNs, AUC=0.901   GSCV:Test = 0.837, Train = 0.855

#   MLP:  Train = 0.852, 33 FPs, 99 FNs, AUC=0.904   GSCV:Test = 0.836, Train = 0.855

#   GBC:  Train = 0.870, 39 FPs, 77 FNs, AUC=0.926   GSCV:Test = 0.834, Train = 0.864

#   GBC:  Train = 0.870, 39 FPs, 77 FNs, AUC=0.926   GSCV:Test = 0.836, Train = 0.866

#   GBC:  Train = 0.860, 43 FPs, 82 FNs, AUC=0.923   GSCV:Test = 0.840, Train = 0.867



# but no Fare_log:

#   GBC:  Train = 0.861, 35 FPs, 89 FNs, AUC=0.915   GSCV:Test = 0.827, Train = 0.860



# Using XGB

# (v32)  XGB:  Train = 0.909, 34 FPs, 47 FNs, AUC=0.958   GSCV: Test = 0.844, Train = 0.966

# (v34)  XGB:  Train = 0.866, 46 FPs, 73 FNs, AUC=0.921   GSCV: Test = 0.828, Train = 0.872  
# yh_kag, yp_kag were calculated above when yh,yp were evaluated.
# Kaggle prediction probs

plt.figure(figsize=(15, 6))

plt.subplot(2, 1, 1)

plt.hist([yh_kag]+[-1.0]+[5.0], bins=fixed_bins, histtype='stepfilled')

plt.title("Kaggle: Prediction Values")

plt.show()
# Put the 0,1 predictions into the original df_test which is the Kaggle test data

df_test['Survived'] = yp_kag.astype(int)
# Any -1 s remaining for answers?

all_answered = (df_test.Survived.min() >= 0)

print("All predictions made?  {}".format(all_answered))
# Save the result as the submission

df_test[['PassengerId','Survived']].to_csv("submission.csv",index=False)
# that's all.

!head -10 submission.csv
!tail -10 submission.csv