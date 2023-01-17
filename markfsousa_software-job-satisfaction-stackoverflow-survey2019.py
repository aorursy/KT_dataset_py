import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import r2_score, mean_squared_error, fbeta_score, accuracy_score

import re

import seaborn as sns

import sys

from time import time

%matplotlib inline



def reload_questions(verbose=False):

    """ Method used to load and modify the schema.

    

    The file scheme contains the question used in the survey.

    The original question were edited so they can fit properly in the graphics."""

    

    ds_scheme = pd.read_csv('/kaggle/input/stack-overflow-developer-survey-results-2019/survey_results_schema.csv')

    

    question_replacements = {

        r'MainBranch': r'Which of the following options best describes you today?',

        r'EduOther': r'Which type of non-degree education have you used or participated in?',

        r'DevType': r'Which of the following describe you?',

        r'WorkChallenge': r'Of these options, what are your greatest challenges to productivity as a developer?',

        r'LanguageWorkedWith': r'Which of the following programming, scripting, and markup languages have you worked in the last year?',

        r'PlatformWorkedWith': r'Which platform have you worked with in the past year?',

        r'PlatformDesireNextYear': r'Which platform you want to work in the next year?',

        r'MiscTechWorkedWith': r'Which frameworks, libraries, and tools have you worked to work with in the past year?',

        r'MiscTechDesireNextYear': r'Which frameworks, libraries, and tools you want to work next in the year?',

        r'Age': r'What is your age (in years)?',

        r'Ethnicity': r'Which of the following do you identify as?'

    }

    for column_name, after in question_replacements.items():

        if verbose:

            print("\nBEFORE: ", ds_scheme.loc[ds_scheme['Column'] == column_name, 'QuestionText'].values[0])

        

        ds_scheme.loc[ds_scheme['Column'] == column_name, 'QuestionText'] = after

        

        if verbose:

            print("AFTER: ", ds_scheme.loc[ds_scheme['Column'] == column_name, 'QuestionText'].values[0])

    

    return ds_scheme
df_questions_text = reload_questions(True)
df_first = pd.read_csv('/kaggle/input/stack-overflow-developer-survey-results-2019/survey_results_public.csv')



df_first.drop(columns='Respondent', inplace=True)

#Drop Respondent because it is a index and Pandas already create another index.



df_first.head()
df_first.columns
df_first.EdLevel.value_counts()
df_first.shape
df_first.nunique().sort_values(ascending=True).head(20)
df_first.nunique().sort_values(ascending=False).head(20)
df_first['LanguageDesireNextYear'].value_counts()
for col in df_first.sort_index(axis=1).columns:

    # Look at any string

    if df_first[col].dtype == 'object':

        

        counts = df_first[col].value_counts()



        value_samples = []

        counter = 0

        for index, value in zip(counts.index, counts.values):

            if ';' in index: # Check if is separeted by semicolon

                value_samples.append(index)

                counter = counter + 1

                

                # Print only 5 values, some colums have thousands of uniques with ;

                if counter > 5: break

        if len(value_samples) > 0:

            print(col)

            print(value_samples, '\n')
df_first.WorkPlan.value_counts()
#Columns to be expanded

multiple_choice = ['Containers',

                  'DatabaseDesireNextYear',

                  'DatabaseWorkedWith',

                  'DevEnviron',

                  'DevType',

                  'EduOther',

                  'Ethnicity',

                  'Gender',

                  'JobFactors',

                  'LanguageDesireNextYear',

                  'LanguageWorkedWith',

                  'LastInt',

                  'MiscTechDesireNextYear',

                  'MiscTechWorkedWith',

                  'PlatformDesireNextYear',

                  'PlatformWorkedWith',

                  'SONewContent',

                  'SOVisitTo',

                  'Sexuality',

                  'WebFrameDesireNextYear',

                  'WebFrameWorkedWith',

                  'WorkChallenge']
def expand_categories(df, column, sep=";", template_name="{0}_{1}", replace=False):

    """ Expand the aggregated categories joined by semicolun into different columns.

    

    Parameters:

    df

        The dataframe containing the aggregated columns.

    column

        The column with aggregated values.

    template_name

        The template the name the new columns for expanded values.

    replace

        A boolean indicating if the new columns must replace the old aggregated columns.

        

    Returns

    A dataframe with the new columns

    A list of the new columns

    """

    feat = df[column]



    sys.stdout.write("\r[%-20s] %d%%" % ('='*0, 0))



    # Find all unique values separeted by semicolon

    unique_values = set()

    for i in feat:

        if type(i) == str:

            vals = i.split(sep)

            for v in vals:

                unique_values.add(v.strip())



    def contains(target, term):

        return int(term.casefold() in str(target).casefold())



    progress=0

    total_uniques = len(unique_values)

    expanded = {}

    added_names = []

    for uvalue in unique_values: # Creates a new column for each unique value

        # Replace unwanted characters

        new_feat_name = uvalue.replace(" ", "_").replace("/", "_").replace("'", "")

        new_feat_name = re.sub(r'[)(-,.:]', '', new_feat_name)

        

        # Define the new name

        new_feat_name = template_name.format(column, new_feat_name)

        added_names.append(new_feat_name)



        # Create the new column according to the content of the old column

        result = pd.Series(feat.apply(contains, args=[uvalue]), index=feat.index)

        expanded[new_feat_name] = result



        if replace:

            df[new_feat_name] = result



        # Print progress

        sys.stdout.write("\r[%-20s] %d%%" % ('='*(progress*20//total_uniques), progress*100//total_uniques))

        sys.stdout.flush()

        progress += 1



    if replace:

        df.drop(columns=column, inplace=True)



    # Print progress

    sys.stdout.write("\r[%-20s] %d%%" % ('='*(20), 100))

    print()

    return pd.DataFrame(expanded), added_names

# List with all expanded new features

expanded_feats = []



df_expanded = df_first.drop(columns=multiple_choice) # Remove the multiple choice columns



for column in multiple_choice: # Expand each multiple-choice column

    print(column)

    new_expanded, new_feats = expand_categories(df_first, column)

    

    # Merge the result with the dataframe withou multiple-choice columns

    df_expanded = pd.merge(df_expanded, new_expanded, left_index=True, right_index=True, how='outer')

    expanded_feats.extend(new_feats) # Update the list
# Checking what is new

expanded_feats
df_expanded.YearsCode.value_counts(dropna=False)


to_dummy_without_na = ['Hobbyist']

to_dummy_with_na = ['BetterLife',

                    'BlockchainIs',

                    'BlockchainOrg',

                    'CodeRev',

                    'CompFreq',

                    'Country',

                    'CurrencyDesc',

                    'CurrencySymbol',

                    'Dependents',

                    'Employment',

                    'EntTeams',

                    'Extraversion',

                    'FizzBuzz',

                    'ITperson',

                    'MainBranch',

                    'OffOn',

                    'OpSys',

                    'PurchaseHow',

                    'ResumeUpdate',

                    'ScreenName',

                    'SocialMedia',

                    'Trans',

                    'UndergradMajor',

                    'UnitTests',

                    'WorkLoc']



# Categorical values representing intervals (replace by average)

interval_to_numerical = ['CareerSat',

                         'EdLevel',

                         'ImpSyn',

                         'JobSat',

                         'JobSeek',

                         'LastHireDate',

                         'MgrMoney',

                         'MgrWant',

                         'MgrIdiot',

                         'OpenSource',

                         'OpenSourcer',

                         'OrgSize',

                         'PurchaseWhat',

                         'SOAccount',

                         'SOComm',

                         'SOFindAnswer',

                         'SOHowMuchTime',

                         'SOJobs',

                         'SOPartFreq',

                         'SOTimeSaved',

                         'SOVisitFreq',

                         'Student',

                         'SurveyEase',

                         'SurveyLength',

                         'WelcomeChange',

                         'WorkPlan',

                         'WorkRemote']



# Columns with dtype object holding numbers (convert string to number)

object_to_num = ['Age1stCode', 'SOVisit1st', 'YearsCode', 'YearsCodePro']



# Numerical columns (nothing to be done)

numerical = ['Age',

             'CodeRevHrs',

             'CompTotal',

             'ConvertedComp',

             'WorkWeekHrs']
def print_uniques(df, columns):

    """ Prints the unique values of the column.

    

    It prints in a way to facilitates the create of a dictionary to trasnforme are unique value.

    """

    for c in df.sort_index(axis=1).columns:

        if c in columns :

            

            print("\n=========", c, "=========")

            print("(", df[c].dtype, ")")

            

            for i in df[c].value_counts(dropna=False).index:

                if i is np.NaN :

                    print("np.nan : np.nan,".format(i))#Keep nans as nans

                else:

                    print("'{}' : ,".format(i))# Prints an empty entry
print_uniques(df_expanded, to_dummy_with_na)
print_uniques(df_expanded, interval_to_numerical)
replacemens_interval_num = {

'CareerSat': {

    np.nan:np.nan,

    'Very satisfied':4,

    'Slightly satisfied':3,

    'Neither satisfied nor dissatisfied':2,

    'Slightly dissatisfied':1,

    'Very dissatisfied':0

},

'EdLevel' : {

    np.nan : np.nan,

    'Other doctoral degree (Ph.D, Ed.D., etc.)' : 8,

    'Master’s degree (MA, MS, M.Eng., MBA, etc.)' : 7,

    'Bachelor’s degree (BA, BS, B.Eng., etc.)' : 6,

    'Professional degree (JD, MD, etc.)' : 5,

    'Associate degree' : 4,

    'Some college/university study without earning a degree' : 3,

    'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)' : 2,

    'Primary/elementary school' : 1,

    'I never completed any formal education' : 0,

},

'ImpSyn' : {

    np.nan : np.nan,

    'Far above average' : 2,

    'A little above average' : 1,

    'Average' : 0,

    'A little below average' : -1,

    'Far below average' : -2,

},

'JobSeek' : {

    np.nan : np.nan,

    'I am actively looking for a job' : 2,

    'I’m not actively looking, but I am open to new opportunities' : 1,

    'I am not interested in new job opportunities' : 0,

},



'JobSat' :  {

    np.nan:np.nan,

    'Very satisfied' : 4,

    'Slightly satisfied' : 3,

    'Neither satisfied nor dissatisfied' : 2,

    'Slightly dissatisfied' : 1,

    'Very dissatisfied' : 0

},

'LastHireDate' : {

    np.nan : np.nan,

    'More than 4 years ago' : 6, # 6 years

    '3-4 years ago' : 3.5, # 3.5 years

    '1-2 years ago' : 1.5, #1.5 years

    'Less than a year ago' : 0.5, # half year

    'I\'ve never had a job' : 0,

    'NA - I am an independent contractor or self employed' : -1,# Create a boolean feature for this -1

},

'MgrIdiot' : {

    np.nan : np.nan,

    'Very confident' : 3,

    'Somewhat confident' : 2,

    'Not at all confident' : 1,

    'I don\'t have a manager' : -1,# Create a boolean feature for this -1

},

'MgrMoney' :{

    'Yes' : 1,

    np.nan : 0.5,

    'Not sure' : 0.5,

    'No' : 0,

},

'MgrWant' : {

    'Yes' : 1,

    np.nan : 0.5,

    'Not sure' : 0.5,

    'No' : 0,

    'I am already a manager' : -1,# Create a boolean feature for this -1

},

'OpenSource' : {

    np.nan : np.nan,

    'OSS is, on average, of HIGHER quality than proprietary / closed source software' : 1,

    'The quality of OSS and closed source software is about the same' : 0,

    'OSS is, on average, of LOWER quality than proprietary / closed source software' : -1,

},    

'OpenSourcer' : {

    'Once a month or more often' : 18/12,# 18 times per yers

    'Less than once a month but more than once per year' : 6/12, # 6 times per yers

    'Less than once per year' : 0.5/12, # less than 1 per year

    'Never' : 0,

},

'OrgSize' : {

    np.nan : np.nan, # keep as nan

    '10,000 or more employees':(10000 * 10)/2, # between 10'000 and 100'000

    '5,000 to 9,999 employees' :(5000 + 9999)/2,# mean values...

    '1,000 to 4,999 employees':(1000 + 4999)/2,

    '500 to 999 employees':(500 + 999)/2,

    '100 to 499 employees' : (100 + 499)/2,

    '20 to 99 employees' : (99+20)/2,

    '10 to 19 employees':(10 + 19)/2,

    '2-9 employees':(2 + 9)/2,

    'Just me - I am a freelancer, sole proprietor, etc.': 1

},

'PurchaseWhat' : {

    np.nan : np.nan,

    'I have a great deal of influence' : 3,

    'I have some influence' : 1,

    'I have little or no influence' : 0,

},

'SOAccount' : {

    'Yes' : 1,

    'Not sure / can\'t remember' : 0.5,

    np.nan : 0.5,

    'No' : 0,

},    

'SOComm' : {

    np.nan : np.nan,

    'Yes, definitely' : 4,

    'Yes, somewhat' : 3,

    'Neutral' : 2,

    'Not sure' : 2,

    'No, not really' : 1,

    'No, not at all' : 0,

},

'SOFindAnswer' : {

    np.nan : np.nan,

    'More than 10 times per week' : 15, # between 10 and 20 per week

    '6-10 times per week' : 8, # 8 per week

    '3-5 times per week' : 4, # 4 per week

    '1-2 times per week' : 1.5, # 1.5 per week

    'Less than once per week' : 0.5, # 0.5 per week

},

'SOHowMuchTime' : {

    np.nan : np.nan,

    '60+ minutes' : 90,

    '31-60 minutes' : 45.5,

    '11-30 minutes' : 20.5,

    '0-10 minutes' : 5,

},

'SOJobs' : {

    np.nan : np.nan,

    'Yes' : 1,

    'No, I knew that Stack Overflow had a job board but have never used or visited it' : 0.5,

    'No, I didn\'t know that Stack Overflow had a job board' : 0,

},

'SOPartFreq' : {

    np.nan : np.nan,

    'Multiple times per day' : 3,

    'Daily or almost daily' : 1,

    'A few times per week' : 10/30,

    'A few times per month or weekly' : 4/30,

    'Less than once per month or monthly' :1/30 ,

    'I have never participated in Q&A on Stack Overflow' : 0

},

'SOTimeSaved' : {

    np.nan : np.nan,

    'Stack Overflow was much faster' : 2,

    'Stack Overflow was slightly faster' : 1,

    'They were about the same' : 0,

    'The other resource was slightly faster' : -1,

    'The other resource was much faster' : -2,

},

'SOVisitFreq' : {

    np.nan : np.nan,

    'Multiple times per day' : 3, # 3 times a day (=90/30)

    'Daily or almost daily' : 1, # every day (=30/30)

    'A few times per week' : 10/30, # 10 per month

    'A few times per month or weekly' : 5/30,# 5 per month

    'Less than once per month or monthly' : 0.5/30, # 0.5 per month

    'I have never visited Stack Overflow (before today)' : 0,

},

'Student' : {

    np.nan : np.nan,

    'Yes, full-time' : 1,

    'Yes, part-time' : 0.5,

    'No' : 0,

},

'SurveyEase' : {

    np.nan : np.nan,

    'Easy' : 0,

    'Neither easy nor difficult' : 0.5,

    'Difficult' : 1,

},

'SurveyLength' : {

    np.nan : np.nan,

    'Too long' : 1,

    'Appropriate in length' : 0.5,

    'Too short' : 0,

},

'WelcomeChange' : {

    np.nan : np.nan,

    'Not applicable - I did not use Stack Overflow last year' : np.nan,

    'A lot more welcome now than last year' : 2,

    'Somewhat more welcome now than last year' : 1,

    'Just as welcome now as I felt last year' : 0,

    'Somewhat less welcome now than last year' : -1,

    'A lot less welcome now than last year' : -2,

},

'WorkPlan' : {

    np.nan : np.nan,

    'There is a schedule and/or spec (made by me or by a colleague), and I follow it very closely' : 1,

    'There is a schedule and/or spec (made by me or by a colleague), and my work somewhat aligns' : 0.5,

    'There\'s no schedule or spec; I work on what seems most important or urgent' : 0,

},

'WorkRemote' : {

    np.nan : np.nan,

    'It\'s complicated' : np.nan,

    'All or almost all the time (I\'m full-time remote)' : 1,

    'More than half, but not all, the time' : 0.75,

    'About half the time' : 0.5,

    'Less than half the time, but at least one day each week' : 0.4,

    'A few days each month' : 0.2,

    'Less than once per month / Never' : 0,

}

}
print_uniques(df_expanded, object_to_num)
# Mapping the string values that need to be transformed into numbers

replacemens_str_num = {

    np.nan : np.nan,

    'I don\'t remember' : np.nan,

    'Less than 1 year' : 0.5,

    'Younger than 5 years' : 4.0,

    'More than 50 years' : 60.0,

    'Older than 85' : 90.0,

}
def convertTo_num(df, map_str_num, map_inter_num, map_dummies_na, map_dummies_nona, verbose=False, drop_firt_dummy=True):

    """ Convert all the strings in the dataframe into numbers.

    

    This is an impure function. It uses parameters and values defined in other cells.

    

    Parameteres:

    df

        The dataframe to be transformed

    map_str_num

        The dictionary mapping values from string to numbers.

    map_dummies_na

        The dictionary mapping feature to be transformed into dummies. These

        features may contain NaNs.

    map_dummies_nona

        The dictionary mapping feature to be transformed into dummies. These

        features do NOT contain NaNs.

    verbose

        Boolean to indicate if need to print logs.

    drop_firt_dummy

        Boolean indicating if the first dummy variable must be dropped.

    """

    

    def replace_float(value, mapping):

        if value in mapping:

            return mapping[value]

        else:

            return float(value)



    df_new = df.copy()

    for col_name, mapping in map_inter_num.items():



        df_new[col_name] = df_new[col_name].apply(func = lambda value: mapping[value])

        if verbose: print("{}: dtype{}".format(col_name, df_new[col_name].dtype))



    new_feat_self_emp = 'LastHireDate_self_employed'

    df_new[new_feat_self_emp] = df_expanded['LastHireDate'] == (-1)

    df_new['LastHireDate'].replace(-1, np.nan, inplace=True)

    

    new_feat_no_manager = 'MgrIdiot_dont_have_manager'

    df_new[new_feat_no_manager] = df_expanded['MgrIdiot'] == (-1)

    df_new['MgrIdiot'].replace(-1, np.nan, inplace=True)

    

    new_feat_Iam_manager = 'MgrWant_Iam_maneger'

    df_new['MgrWant_Iam_maneger'] = df_expanded['MgrWant'] == (-1)

    df_new['MgrWant'].replace(-1, np.nan, inplace=True)

    

    new_dummies = [new_feat_self_emp, new_feat_no_manager, new_feat_Iam_manager]



    print()

    for col_name in object_to_num:



        df_new[col_name] = df_new[col_name].apply(func = lambda value: replace_float(value, map_str_num))

        if verbose: print("{}: dtype{}".format(col_name, df_new[col_name].dtype))





    new_dummies = []

    dummies = pd.get_dummies(df_new[map_dummies_na], drop_first=drop_firt_dummy)

    new_dummies.extend(dummies.columns)



    df_new = pd.merge(df_new, dummies, right_index=True, left_index=True, how='outer')

    df_new.drop(columns = map_dummies_na, inplace=True)





    dummies = pd.get_dummies(df_new[map_dummies_nona], drop_first=drop_firt_dummy)

    new_dummies.extend(dummies.columns)



    df_new = pd.merge(df_new, dummies, right_index=True, left_index=True, how='outer')

    df_new.drop(columns = map_dummies_nona, inplace=True)

    

    return df_new, new_dummies

# Convert the values of the dataframe into numbers

df_numerical, new_dummies = convertTo_num(df_expanded,

                                          replacemens_str_num,

                                          replacemens_interval_num,

                                          to_dummy_with_na,

                                          to_dummy_without_na)

df_numerical.shape
df_numerical.isna().sum().sort_values(ascending=False).head(10)
df_numerical.isna().sum().sort_values(ascending=False).head(10)


def get_group_dummies(df, name):

    """ This function look for columns starting with a name.

    

    It is very useful to find dummy variables named with a prefix.

    

    Parameters:

    df

        The dataframe to look for columns.

    name

        The prefix of the column.

    """

    return [col for col in df.columns if col.startswith(name)]
new_dummies
df_numerical[new_dummies].sum().sort_values().head(30)
df_numerical.SOTimeSaved.value_counts()
df_last = df_numerical

for c in df_last.sort_index(axis=1).columns:

    #if c not in expanded_feats :

        print("\n\n\n============={}=============".format(c))

        print(df_last[c].dtype)

        print(df_last[c].value_counts(dropna=False).head(30))
# Check how many unique values remain

df_numerical.nunique().sort_values(ascending=False).head(30)
df_numerical.shape
df_numerical.isna().sum().sort_values()/df_numerical.shape[0]
def remove_feat_and_dummies(df, features):

    """ Remove the feature informed and their originated dummies.

    

    Parameters:

    df

        The dataframe from which the features are going to be removed.

    features

        The prefix of the features to be removed. If there are conflic 

        of prefix between different features they all will be removed.

    """

    correlated_columns = []

    for feat in features:

        correlated_columns.extend(get_group_dummies(df, feat))

    df_corr = df.drop(columns=correlated_columns)

    return df_corr
# Remove unwanted highly correlated columns

# Currency symbol and description are highly correlated to country

# Total compensation and compensation frequency are values used to calculate converted compensatoin

# Age, YearsCode and YearsCodePro are correlated features, let's drop the first two

# Employment_Employed part-time is correlated to WorkWeekHrs

feats_corr = ['CurrencyDesc_', 'CurrencySymbol_', 'CompFreq_', 'CompTotal', 'YearsCode', 'Age', 'Employment_Employed part-time']

df_corr = remove_feat_and_dummies(df_numerical, features=feats_corr)



for fe in feats_corr:

    assert(len(get_group_dummies(df_corr, fe)) == 0)
print("NaNs %")

df_corr.isna().mean().sort_values().tail()
def clean_data(df, dependent_var_name, nans_tresh=0.3, verbose=True):

    df = df.dropna(axis=0, subset=[dependent_var_name])# Remove nans in the dependent variable

    

    df = df.dropna(how='all', axis=0) # Remove rows complete with nans

    df = df.dropna(how='all', axis=1) # Remove columns complete with nans

    

    if verbose: print("removing {} columns by unique values".format(( df.nunique() <= 1).sum()))

    df = df.loc[:, df.nunique() > 1] # Keep columns with at least two unique values

    

    if verbose: print("removing {} columns by nans %".format((df.isna().mean(0) >= nans_tresh).sum()))

    df = df.loc[:, df.isna().mean(0) < nans_tresh] # Columns with nans% < nans_tresh

            

    

    return df.copy()
df_clean = clean_data(df_corr, 'ConvertedComp')



df_last = df_clean

df_last.shape
df_corr.shape
previous_length = len(df_corr)

nans_in_dependent = df_corr.ConvertedComp.isnull().sum()

print("The dataframe had {} rows.".format(previous_length))

print("There were {} NaNs in the dependet variable.".format(nans_in_dependent))

print("These rows with NaNs were removed and {} remains.".format(previous_length-nans_in_dependent))
def drop_all_nans(df, column_treshold = 0.1, verbose=True):

    """ Remove columns with a certain threshold of nans and then remove all nans

    

    Parameters:

    df

        The dataframe to remove nans

    column_treshold

        The limit to drop the column if it has more nans than expected.

    verbose

        Boolean indicating if must print information

    """

    

    if verbose:

        print("Removing {} columns by nans %".format((df.isna().mean() >= column_treshold).sum()))

        

    # Keep columns with nans% < column_treshold

    df_new = df.loc[:, (df.isna().mean() < column_treshold)]



    # Drop nans

    if verbose:

        removing_rows = (df_new.isna().sum(1) > 0).sum()

        previous_rows = len(df_new)

        print("Removing {} rows with NaNs values from the total {} rows.".format(removing_rows, previous_rows))

    

    df_new = df_new.dropna(axis=0, how='any')

    

    if verbose:

        print("Final shape {}".format(df_new.shape))

    

    return df_new
# Drop nans

df_dense = drop_all_nans(df_clean)



df_last = df_dense

df_last.shape
drop_all_nans(df_corr).shape
def reduce_cuttoff(df, cutoff, verbose=False):

    """ Drop columns with a certaing amout of zeros.

    

    Parameters:

    df

        The dataframe to drop columns

    cutoff

        The limit (%) of zeros allowed in the columns.

    

    Returns:

    The dataframe with columns removed.

    """

    

    reduce_X = df.loc[:, (df == 0).mean(0) < cutoff]

    for col_name in to_dummy_with_na:

        dummies = get_group_dummies(reduce_X, col_name)

        

        non_zeros = (reduce_X[dummies] != 0).sum().sum()

        if len(dummies) >= 2 and non_zeros > (reduce_X.shape[0] * .99):

            # Drop first dummy if non-zeros values represent more than 99% or rows

            if verbose:

                print("Dropping dummy {}".format(dummies[0]))

                

            reduce_X = reduce_X.drop(columns=dummies[0])

            

    return reduce_X
def train_cuttoffs(X, y, cutoffs, test_size = 0.3, random_state=50, plot=True):

    '''

    Parameters:

    X

        The dataframe of independent variables (predictors).

    y

        The dataframe of dependent variable to be predicted.

    cutoffs

        List of floats of percentage of zeros allowed in the columns.

    test_size

        Float between 0 and 1, default 0.3, determines the proportion of data as test data

    random_state

        Int, default 50, controls random state for train_test_split

    plot

        Boolean, True to plot result



    Returns:

    scores

        the scores of all trains

    lm_model

        The best linear regression

    data_split

        The data split of the best model

    best_cutoff_st

        The best cutoff

    '''

    r2_scores_test, r2_scores_train, num_feats, results = [], [], [], dict()

    for cutoff in cutoffs:

        print("Cutoff:", cutoff)



        reduce_X = reduce_cuttoff(X, cutoff, verbose=False)

        print("num_feats: {}".format(reduce_X.shape[1]))

        num_feats.append(reduce_X.shape[1])



        #split the data into train and test

        X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)



        #fit the model and obtain pred response

        lm_model = LinearRegression(normalize=True)

        lm_model.fit(X_train, y_train)

        

        y_test_preds = lm_model.predict(X_test)

        y_train_preds = lm_model.predict(X_train)

        

        r2_test = r2_score(y_test, y_test_preds)

        r2_train = r2_score(y_train, y_train_preds)

        mse_test = mean_squared_error(y_test, y_test_preds)

        mse_train = mean_squared_error(y_train, y_train_preds)



        scores = {

            "r2_scores_test" : r2_test,

            "r2_scores_train" : r2_train,

            "mse_scores_test" : mse_test,

            "mse_scores_train" : mse_train,

        }

        

        print(scores)



        #append the r2 value from the test set

        r2_scores_test.append(r2_test)

        r2_scores_train.append(r2_train)

        results[str(cutoff)] = r2_score(y_test, y_test_preds)



    best_cutoff_st = max(results, key=results.get)

    print("\nBest cutoff:", best_cutoff_st)



    if plot:

        #plt.plot(num_feats, r2_scores_test, label="Test", alpha=.5, )

        #plt.plot(num_feats, r2_scores_train, label="Train", alpha=.5)

        plt.plot(cutoffs, r2_scores_test, label="Test", alpha=.5, )

        plt.plot(cutoffs, r2_scores_train, label="Train", alpha=.5)

        plt.xlabel('Cutoff')

        plt.ylabel('Rsquared')

        plt.title('Rsquared by Number of Features')

        plt.legend(loc=1)

        plt.grid(True)

        plt.show()



    reduce_X = reduce_cuttoff(X, float(best_cutoff_st))

    num_feats.append(reduce_X.shape[1])



    #split the data into train and test

    X_train, X_test, y_train, y_test = train_test_split(reduce_X, y, test_size = test_size, random_state=random_state)

    

    data_split = {

        "X_train":X_train,

        "X_test": X_test,

        "y_train": y_train,

        "y_test": y_test,

    }



    #fit the model

    lm_model = LinearRegression(normalize=True)

    lm_model.fit(X_train, y_train)

    

    y_test_preds = lm_model.predict(X_test)

    y_train_preds = lm_model.predict(X_train)

    

    r2_test = r2_score(y_test, y_test_preds)

    r2_train = r2_score(y_train, y_train_preds)

    mse_test = mean_squared_error(y_test, y_test_preds)

    mse_train = mean_squared_error(y_train, y_train_preds)



    scores = {

        "r2_scores_test" : r2_test,

        "r2_scores_train" : r2_train,

        "mse_scores_test" : mse_test,

        "mse_scores_train" : mse_train,

    }

    



    return scores, lm_model, data_split, float(best_cutoff_st)

df_dense.shape
from sklearn.ensemble import RandomForestRegressor



def fit_RF(df, y_colum, max_depth=5, n_estimators=100, test_size = 0.3, random_state=0):

    """Trains a RandomForest.

    df

        The dataframe with independent variables to be trained (predictors).

    y_colum

        The dataframe with dependent variables to be predicted.

    max_depth

        default=5

        Mas depth of the tress in the RandomForests

    n_estimators 

        default=100

        The number of tress to be trained.

    test_size

        default=0.3

        The percentage of date used to test the model (between 0 and 1).

    random_state

        default=0

        Controls random state for train_test_split

    """

    

    y = df[y_colum] # define y

    X = df.drop(columns=[y_colum]) # remove y from dataset



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=random_state)

    data_split = {

        'X_train' : X_train,

        'X_test' : X_test,

        'y_train' : y_train,

        'y_test' : y_test

    }



    regr = RandomForestRegressor(max_depth=max_depth, random_state=random_state,

                                 n_estimators=n_estimators)

    regr.fit(X_train, y_train)

    

    y_test_preds = regr.predict(X_test)

    y_train_preds = regr.predict(X_train)

    

    preds = {

        'y_test_preds' : y_test_preds,

        'y_train_preds' : y_train_preds

    }



    #append the r2 value from the test set

    r2_scores_test = r2_score(y_test, y_test_preds)

    r2_scores_train = r2_score(y_train, y_train_preds)

    mse_scores_test = mean_squared_error(y_test, y_test_preds)

    mse_scores_train = mean_squared_error(y_train, y_train_preds)

    

    scores = {

        "r2_scores_test" : r2_scores_test,

        "r2_scores_train" : r2_scores_train,

        "mse_scores_test" : mse_scores_test,

        "mse_scores_train" : mse_scores_train,

    }



    print(scores)



    print(regr.feature_importances_)

    

    return regr, scores, data_split, preds
# Train the RandomForest

regr_comp_RF, scores_comp_RF, data_split_comp_RF, preds_comp_RF = fit_RF(df_dense,

                                                                         y_colum = 'ConvertedComp',

                                                                         max_depth=5,

                                                                         n_estimators=50,

                                                                         random_state=10)

def DF_feature_importance(regr, X_train):

    """ Defines the feature importance according to the trained model.

    

    Parameters:

    regr

        The trained RandomForest regressor.

    X_train

        The dataframe in which the regressor was trained.

    

    Returns

    A datafram with the importance of each feature.

    """

    coefs_df_RF = pd.DataFrame()

    coefs_df_RF['Feature'] = X_train.columns

    coefs_df_RF['Importance'] = regr.feature_importances_

    return coefs_df_RF
coefs_df_comp_RF = DF_feature_importance(regr_comp_RF, data_split_comp_RF['X_train'])

plot_df = coefs_df_comp_RF.sort_values(by='Importance', ascending=False).head(5)

print(plot_df)

plot_df
def retrieve_question(df, question_col, df_question):

    """ Retrives the text of the questions.

    

    Parameters:

    df

        The dataframe containing the columns to recover the question.

    question_col

        What columns are going to be used to recover the text.

    df_question

        The dataframe with the text of questions.

    

    Returns:

    A list of questions.

    """

    questions = []

    

    for name in df[question_col]:

        #print(name)

        print(name)

        name_split = name.split('_')

        text = df_question.loc[df_question.Column == name_split[0]]

        if len(text) > 0:

            quest = text['QuestionText'].values[0] + "\n" + " ".join(name_split[1:])

            questions.append(quest)

            print(quest)

        print()

    

    return questions



df_questions_text = reload_questions()

plot_df['Questions'] = retrieve_question(plot_df, "Feature", df_questions_text)

sns.set(font_scale=2.3)

f, ax = plt.subplots(figsize=(16, 8))



ax = sns.barplot(x='Importance', y='Questions', data=plot_df, color="r")

ax.legend(ncol=2, loc="lower right", frameon=True, )

def coef_weights(coefficients, X_train):

    ''' Retrive the coefficient weights of each feature.

    

    Parameters:

    coefficients

        the coefficients of the linear model 

    X_train

        the training data, so the column names can be used

    

    Returns:

    coefs_df

        a dataframe holding the coefficient, estimate, and abs(estimate)

    '''

    coefs_df = pd.DataFrame()

    coefs_df['Feature'] = X_train.columns

    coefs_df['Coefficient'] = coefficients

    coefs_df['Coefficient (absolut)'] = np.abs(coefficients)

    coefs_df = coefs_df.sort_values('Coefficient (absolut)', ascending=False)

    return coefs_df
cutoffs = [1, 0.8] #Percentage of zeros allowed in the columns



y_colum = 'ConvertedComp' # Dependent variable

y = df_dense[y_colum] # define y

X = df_dense[plot_df.Feature.values] # Only used the important features from the RandomForest



scores, lm_model, data_split, best_cutoff = train_cuttoffs(X, y, cutoffs, test_size=0.3, random_state=1000, plot=False)
# Get the coefficents

coef_df = coef_weights(lm_model.coef_, data_split['X_train'])



# Normalize the coefficents

coef_df['Coefficient_norm'] = coef_df['Coefficient']/np.abs(coef_df['Coefficient'].sum())

coef_df.sort_values(by='Coefficient (absolut)', ascending=False).head(10)



coef_df['Coefficient'] = coef_df['Coefficient']/coef_df['Coefficient'].sum()





df_questions_text = reload_questions()

coef_df['Questions'] = retrieve_question(coef_df, "Feature", df_questions_text)



sns.set(font_scale=2.3)

f, ax = plt.subplots(figsize=(16, 8))

#sns.set_color_codes("pastel")

ax = sns.barplot(x='Coefficient_norm', y='Questions', data=coef_df.sort_index(), color="blue")

ax.legend(ncol=2, loc="lower right", frameon=True, )

ax.set_xlabel('Importance')

ax.set_ylabel('Coefficient')
cutoffs = [0.99, 0.98, 0.95]



y_colum = 'ConvertedComp'

y = df_dense[y_colum] # define y

X = df_dense.drop(columns=[y_colum]) # remove y from dataset



scores, lm_model, data_split, best_cutoff = train_cuttoffs(X, y, cutoffs, test_size=0.3, random_state=0)
#Use the function

coef_df = coef_weights(lm_model.coef_, data_split['X_train'])



#A quick look at the top results

coef_df_sort = coef_df.sort_values(by='Coefficient (absolut)', ascending=False).head(10)

coef_df_sort


#coef_df = coef_df[coef_df['Feature'].isin(plot_df['Feature'].values)]



coef_df_sort = coef_df_sort.head(5)



df_questions_text = reload_questions()

coef_df_sort['Questions'] = retrieve_question(coef_df_sort, "Feature", df_questions_text)



sns.set(font_scale=2.3)

f, ax = plt.subplots(figsize=(16, 8))

#sns.set_color_codes("pastel")

ax = sns.barplot(x='Coefficient', y='Questions', data=coef_df_sort, color="blue")

ax.legend(ncol=2, loc="lower right", frameon=False, )

#ax.set_xlabel('totalCount')
y_colum_job = 'JobSat'



df_clean_job = clean_data(df_corr, y_colum_job)



# JobSat is highly cross correlated to CareerSat

df_clean_job = df_clean_job.drop(columns='CareerSat')



df_last_job = df_clean_job

df_last_job.shape



df_dense_job = drop_all_nans(df_clean_job)





df_last_job = df_dense_job

df_last_job.shape
# Train the RandomForest to the job satisfaction

regr_job, scores_job, data_split_job, preds_job = fit_RF(df_dense_job,

                                                                     y_colum = y_colum_job,

                                                                     max_depth=5,

                                                                     n_estimators=50,

                                                                     random_state=200)

# Look into the feature importance

coefs_df_RF_job = DF_feature_importance(regr_job, data_split_job['X_train'])

coefs_df_RF_job_sort = coefs_df_RF_job.sort_values(by='Importance', ascending=False).head(6)

#print(coefs_df_RF_job_sort)

coefs_df_RF_job_sort
# Define the data about importance to plot

plot_df = coefs_df_RF_job.sort_values(by='Importance', ascending=False).head(5)
# Plot the feature importance about the job satisfaction.

df_questions_text = reload_questions()

plot_df['Questions'] = retrieve_question(plot_df, "Feature", df_questions_text)



sns.set(font_scale=2.3)

f, ax = plt.subplots(figsize=(16, 8))



ax = sns.barplot(x='Importance', y='Questions', data=plot_df, color="r")

cutoffs = [1, 0.8]





y_colum_job = 'JobSat'

y = df_dense_job[y_colum_job] # define y

X = df_dense_job[plot_df.Feature.values] # remove y from dataset



scores, lm_model, data_split, best_cutoff = train_cuttoffs(X, y, cutoffs, test_size=0.3, random_state=1000, plot=False)
# Retrieve the weights of the coefficents

coef_df = coef_weights(lm_model.coef_, data_split['X_train'])





coef_df.sort_values(by='Coefficient (absolut)', ascending=False).head(10)
coef_df = coef_df[coef_df['Feature'].isin(plot_df['Feature'].values)]



coef_df['Coefficient_norm'] = coef_df['Coefficient']/np.abs(coef_df['Coefficient'].sum())





df_questions_text = reload_questions()

coef_df['Questions'] = retrieve_question(coef_df, "Feature", df_questions_text)



sns.set(font_scale=2.3)

f, ax = plt.subplots(figsize=(16, 8))

#sns.set_color_codes("pastel")

ax = sns.barplot(x='Coefficient_norm', y='Questions', data=coef_df.sort_index(), color="blue")

#ax.legend(ncol=2, loc="lower right", frameon=True, )

ax.set_xlabel('Importance')
# What are the occupations of people with high job satisfaction?

df_numerical, new_dummies = convertTo_num(df_expanded,

                                          replacemens_str_num,

                                          replacemens_interval_num,

                                          to_dummy_with_na,

                                          to_dummy_without_na)

def calculate_corr(df, feats, feat_corrwith):

    """Calculates the correlation between a set of features and another feature.

    

    Parameters:

    feats

        A list with the names of features to calculate the correlation.

    feat_corrwith

        The name of the other feature to calculate the correlation

        

    Returns:

    A dataframe with the correlation.

    """

    df_corrwith = pd.DataFrame()

    df_corrwith['corr'] = df[get_group_dummies(df, feats)].corrwith(df[feat_corrwith]).sort_values()

    df_corrwith = df_corrwith.reset_index()

    df_questions_text = reload_questions()

    df_corrwith['Questions'] = retrieve_question(df_corrwith, "index", df_questions_text)

    return df_corrwith
# Calculates the correlation between all the types of dev and the job satisfaction

df_corr_occupation = calculate_corr(df_dense, 'DevType', 'JobSat')
df_corr_occupation
# Retrives the questions' text

df_questions_text = reload_questions()

df_corr_occupation['Questions'] = retrieve_question(df_corr_occupation, "index", df_questions_text)



df_corr_occupation['Questions2'] = df_corr_occupation['Questions'].apply(lambda x : x.rsplit('\n')[1])

df_corr_occupation['Questions2']

df_corr_occupation
# Plot the correlation

sns.set(font_scale=2.3)

f, ax = plt.subplots(figsize=(16, 33))

#sns.set_color_codes("pastel") # dark deep

ax = sns.barplot(x='corr', y='Questions2', data=df_corr_occupation.sort_values(by='corr', ascending=False))#, color="blue"

#ax.legend(ncol=2, loc="lower right", frameon=True, )

ax.set_xlabel('Correlation')

ax.set_ylabel('Occupation')
# Get the dummy variable of developer type 

feats_dev_job = get_group_dummies(df_dense, 'DevType')



# Add job satisfaction to the list

feats_dev_job.append('JobSat')

df_dense_corr = df_dense[feats_dev_job]



# Group the dev types by job satisfaction

group = df_dense_corr.groupby(['JobSat']).sum()/df_dense_corr.sum().drop('JobSat')*100

group = group.reset_index()

group = group.melt(id_vars = 'JobSat', var_name="variable")



#df_questions_text = reload_questions()

group['Occupation'] = retrieve_question(group, "variable", df_questions_text)



group['Occupation'] = retrieve_question(group, "variable", df_questions_text)



# Creates another column only with the answers, the second element after \n (break line)

group['Occupation2'] = group['Occupation'].apply(lambda x : x.rsplit('\n')[1])

group['Occupation2']
# Retrives equivalent text of the numerical job satisfaction

map_job_sat = {

    4: 'Very satisfied',

    3: 'Slightly satisfied',

    2: 'Neither satisfied nor dissatisfied',

    1: 'Slightly dissatisfied',

    0: 'Very dissatisfied'

}

group['Job Satisfaction'] = group['JobSat'].apply(lambda x : map_job_sat[x])

group.value.max()
map_job_sat.values()
group
# Plot the relative frequencies

group = group.sort_values(by=['JobSat', 'value'], ascending=False)

sns.set(font_scale=2.3)

f, ax = plt.subplots(figsize=(10, 40))

sns.set_color_codes("pastel") # dark deep



ax = sns.barplot(x='value', y='Occupation2', data=group, hue='Job Satisfaction', hue_order=map_job_sat.values(), palette="viridis_r")#, color=colors)#, color="blue"



plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

ax.xaxis.set_minor_locator(plt.MultipleLocator(5))

ax.grid(b=True, axis='x', which='major', color='#FFFFFF', linewidth=1.8)

ax.grid(b=True, axis='x', which='minor', color='#FFFFFF', linewidth=1.3, linestyle='--')# '-', '--', '-.'





ax.set_xlabel('Relative Frequency (%)')

ax.set_ylabel('Occupation')
# Define the names of the dummy variables of EduOther

feats_edu_job = get_group_dummies(df_dense, 'EduOther')



# Add job satisfaction

feats_edu_job.append('JobSat')

df_dense_corr = df_dense[feats_edu_job]



# Group the education by job satisfactoin

group = df_dense_corr.groupby(['JobSat']).sum()/df_dense_corr.sum().drop('JobSat')*100

group = group.reset_index()

group = group.melt(id_vars = 'JobSat', var_name="variable")



# Retrive the text of the questions

df_questions_text = reload_questions()

group['Education'] = retrieve_question(group, "variable", df_questions_text)



# Define another columns with the answer

group['Education_answer'] = group['Education'].apply(lambda x : x.rsplit('\n')[1])

group['Education_answer']
# Retrieves the equivalent text of the numerical job satisfaction

map_job_sat = {

    4: 'Very satisfied',

    3: 'Slightly satisfied',

    2: 'Neither satisfied nor dissatisfied',

    1: 'Slightly dissatisfied',

    0: 'Very dissatisfied'

}

group['Job Satisfaction'] = group['JobSat'].apply(lambda x : map_job_sat[x])

group
group[group["Education_answer"] == 'Participated in a hackathon'].sum()
# Plot the relative frequency of education by job satisfaction

group = group.sort_values(by=['JobSat', 'value'], ascending=False)

sns.set(font_scale=2.3)

f, ax = plt.subplots(figsize=(10, 40))

sns.set_color_codes("pastel") # dark deep



#colors = plt.cm.GnBu_r(5) #RdYlGn

ax = sns.barplot(x='value', y='Job Satisfaction', data=group, hue='Education_answer', palette="tab10")#, color=colors)#, color="blue"

#ax.legend(ncol=1, loc=2, frameon=True, )#loc="lower right"

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#ax.grid(True, which='both', axis='x')



ax.xaxis.set_minor_locator(plt.MultipleLocator(5))

ax.grid(b=True, axis='x', which='major', color='#FFFFFF', linewidth=2.5)

ax.grid(b=True, axis='x', which='minor', color='#FFFFFF', linewidth=1.3, linestyle='--')# '-', '--', '-.'





ax.set_xlabel('Learning Method Relative Frequency (%)')

ax.set_ylabel('Job Satisfaction')