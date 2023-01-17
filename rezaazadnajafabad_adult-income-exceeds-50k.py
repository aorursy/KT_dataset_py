# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def msubpartition():

    ''' Add subpartition to the output of the code in the terminal :) '''

    print('\n**************************************\n')

def load_and_head_dataframe(str_address):

    ''' 

    load data frame, print head and shape | \n

    inputs, str_address: address of the data frame | \n

    return the loaded data frame

    '''

    #mpartition()

    print('loading data set,print head and shape')

    # read the csv file, df as the data frame

    df = pd.read_csv(str_address)

    # print the the first few line of the data frame

    msubpartition()

    print('\nhead of the data frame:\n\n', df.head())

    # print the shape of the data set

    msubpartition()

    print('The shape of the data frame:', df.shape, '\n')



    return df
# load the data

# source web link: http://archive.ics.uci.edu/ml/datasets/Adult

df = load_and_head_dataframe('/kaggle/input/adult-data-set/income_exceeds_50k_modeling.csv')
# print data frame information

df.info()
def describe_dataframe(df):

    '''

    print describtion of the data frame on the termianl both for numeric and

    the categorical columns\n

    input:\n

    df: data frame

    '''

    print('The discription of the data frame')



    # print the description for the numeric columns

    msubpartition()

    print('Description of the numeric columns of the data frame:\n\n', df.describe())

    # print the description of the categorical columns of the data frame

    msubpartition()

    print('Description of the categorical columns of the data frame:\n\n', df.describe(include=['O']))
# describe the data frame

describe_dataframe(df)
# drop the feature: Education-num. Since education is present in the data frame

df = df.drop(['Education-num'], axis=1)
def handle_missing_data(df):

    ''' 

    handel missing data, according to the number of missing data and your insight

    choose any necessary action from step 1 to 4 to handle the missing data  |\n 

    input, df: data frame | \n

    return modified data frame 

    '''

    print('handle missing data')



    # print the number of missing data for each column in a decreasing sorted way

    msubpartition()

    print('the number of missing data for each column:\n\n', (df.isnull().sum()).sort_values(ascending=False))



    # step 1

    # In case of any missing data in order to eliminate the rows with null members:

    df = df.dropna(how='any',axis=0) # It will erase every row (axis=0) that has "any" Null value in it.

    msubpartition()

    print('rows including missing data is deleted')

    print('\nshape of the data frame after removing rows with missing data:', df.shape)

    print('\nthe number of missing data for each column after data cleaning\n\n', (df.isnull().sum()).sort_values(ascending=False))



    # # step 2

    # # missing values can be filled with exact prefered value.

    # # define a dictionary like:

    # values = {'column_1_name': 'prefered_value', 

    #           'column_2_name': 'prefered_value'} # continue this pattern for all columns with missing data

    # df.fillna(value=values)

    # print('\nthe number of missing data for each column after data cleaning\n\n', (df.isnull().sum()).sort_values(ascending=False))





    # # step 3

    # # missing values in a column can be filled with most repeated value in the column

    # # df['weekday'].mode()[0] gets the most repeated element in the column

    # df['prefered_column'].fillna(df['prefered_column'].mode()[0], inplace=True)

    # print('\nthe number of missing data for each column after data cleaning\n\n', (df.isnull().sum()).sort_values(ascending=False))



    # # step 4

    # # missing values in a column can be filled with mean of the values in the column

    # df['prefered_column'].fillna(df['prefered_column'].mean(), inplace=True)

    # print('\nthe number of missing data for each column after data cleaning\n\n', (df.isnull().sum()).sort_values(ascending=False))



    return df
# handel missing data

df = handle_missing_data(df)
def modify_columns_name(df):

    '''Renaming the columns of the data set for more conviniance, based on the

    information provided on the sourcewebsite of the data frame and

    prints describtion of the data frame after remaning some columns | \n

    inputs, df: data frame |\n

    return data frame after renaming some columns

    '''

    

    print('rename columns and print discription')

    msubpartition()

    print('***WARNING***: This function must be modified before each implemetation')



    df.rename(columns={

                       'fnlwgt': 'Final-weight'

                      }, inplace = True)

    

    # print the discription of the data set

    msubpartition()

    print('description of the data set after renameing some columns:\n\n', df.describe())



    return df
# rename the columns

df = modify_columns_name(df)
def modify_data_type(df, list_categorical_features_name):

    ''' 

    Modify data type of some coulumns. | \n

    input:\n

    df: data frame | \n

    list_categorical_features_name: list containing name of the features,

    which you prefer to change their type to categorical\n

    return\n

    data frame after modifying the data type of some columns

    '''

    # mpartition()

    print('data type modification')



    # print data type of the columns of the data set

    msubpartition()

    print('\ndata type of the columns:\n\n', df.dtypes)

        

    # convert type of some of the columns to <catogorical type> for

    # ease of modification and computation

    # astype() method is used to cast a pandas object to a specified 

    # dtype. astype() function also provides the capability to convert

    # any suitable existing column to categorical type. 

    # DataFrame.astype() function comes very handy when we want to

    # case a particular column data type to another data type

    

    for member in list_categorical_features_name:

        df[member] = df[member].astype('category')

    

    # print the type of the columns of the data set after the type modification

    msubpartition()

    print('data type of the columns after type modification:\n\n', df.dtypes)



    return df
# data type modification

df = modify_data_type(df, ['workclass', 'education', 'Marital-status',

 'occupation', 'relationship', 'race', 'sex', 'Native-country', 

 'income_exceeds_50K'])
def outliers_status(dataframe_column):

    ''' returns the number of outliers of input and accociated percentage

        input: a non categorical column of the data frame'''

    # outlier counter

    outlier_counter = 0

    # compute first quantile

    Q1 = dataframe_column.quantile(0.25)

    # compute third quantile

    Q3 = dataframe_column.quantile(0.75)

    # compute the interquartile range (IQR)

    IQR = Q3 - Q1

    # low thershold

    l_t = Q1 - 1.5 * IQR   

    # high threshold

    h_t = Q3 + 1.5 * IQR    



    # define a list for input   

    list_input = list(dataframe_column)



    # compute number of outliers

    for i in range( len(list_input) ):

        if list_input[i]>h_t or list_input[i]<l_t:

            outlier_counter = outlier_counter + 1



    # compute the percentage of the outliers

    percentage = outlier_counter / len(dataframe_column) * 100



    return outlier_counter, percentage



#------------------------------------------------------------------------------------------------------------------



def get_IQR_and_count_outliers(df):

    '''

    compute IQR, the number of outliers and associated percentage for numerical columns\n

    inputs:\n

    df: the data frame

    '''



    print('compute IQR, the number of outliers and associated percentage for numerical columns')

    msubpartition()

    print('***ATTENTION***: This function should be manipulated prior to hot encoding')

    msubpartition()

    print('***ATTENTION***: This function should be modified before manipulation')



    # create a new data frame with countable features plus target variable

    # define a list including countable features

    features_countable = ['age', 'Final-weight', 'Capital-gain',

                        'Capital-loss', 'Hours-per-week'] 

    # create a dataframe with countable features

    df_with_countable_features = df[features_countable]

    

    df_with_count_features_indep_variable = df_with_countable_features

    # compute the first and third quantile of the data frame:

    Q1 = df_with_count_features_indep_variable.quantile(0.25)

    Q3 = df_with_count_features_indep_variable.quantile(0.75)

    # compute the interquartile range (IQR)

    IQR = Q3 - Q1

    # print the IQR

    msubpartition() 

    print('the interquantile for the data frame:\n\n', IQR)

    

    # get the number and percentage of the outliers of each numerical column

    msubpartition() 

    print('the number of outliers and associated percentage for numerical columns:\n')



    # print the number of outliers and accociated percentage

    for member in features_countable:

        # get the number of outliers and accociated percentage

        outliers_data = outliers_status(df_with_count_features_indep_variable[member])

        print('number of outliers of %s and accociated percentage: %d, %.2f%%' % 

             (member, outliers_data[0], outliers_data[1]))
# compute IQR, the number of outliers and associated percentage for numerical columns

get_IQR_and_count_outliers(df)
import matplotlib.pyplot as plt

import seaborn as sns



def uni_variate_box_plot(df_target, str_target_var_name):

    '''

    plot univarte box plot for targat variable to visulaize the outliers\n

    inputs:\n

    df_target: the column of the data frame associated with the target. i.e. df['total_count']\n

    str_target_var_name: the name of the target variable

    '''

    

    print('Plot univariet box plot for %s inorder to Deal with Outliers' % str_target_var_name)



    # plot box for target variable

    plt.figure('Box Plot for %s' % (str_target_var_name))

    # set the sytle of the plot: gird

    sns.set_style('whitegrid') 

    # plot the diagram

    sns.boxplot(x=df_target) 

    plt.title('Box Plot for %s' % (str_target_var_name))
# box plot for target variable to visualize the outliers (before hot encoding)

for member in ['age', 'Final-weight', 'Capital-gain', 'Capital-loss', 'Hours-per-week']:

    uni_variate_box_plot(df[member], member)

def plot_correlation_heatmap(df):

    '''

    'plot heat map for numerical features, this function automatically

    consider the numverical features. Also pirnts the corrlation table in the termianl\n

    inputs:\n

    df: the data farame

    '''



    print('heat map for numerical features')

    msubpartition()

    print('***ATTENTION***: This functions needs to be modified before maniputlation')

    msubpartition()

    print('If id column exixts in the data frame, it should ve omitted. Since correlation of \nid with other features makes no sense')



    df_with_no_id_column = df

    # set decimial precision to the all computation to 2

    pd.set_option('precision', 2)

    # set the figure size

    plt.figure('heat map for numerical features', figsize=(8, 8))

    # plot the heat map. 

    # annot is to print the value of corre in each cell of the plot

    sns.heatmap(df_with_no_id_column.corr(), square=True, annot=True)

    

    # show the plot in a tight layout (margins are automatically set to display

    # all of the titles)

    plt.tight_layout() 

    

    # print the correlation of the features

    msubpartition()

    print('correlation for the numerical features:\n\n', df_with_no_id_column.corr())
# plot heat map for correlation between numerical features

plot_correlation_heatmap(df)
from scipy.stats import shapiro, bartlett, levene, ttest_ind, f_oneway, kruskal, pearsonr

from sklearn.feature_selection import chi2



def replace_categerical_features_content_with_numbers(df):

    '''

    This function Automaically replaces all the content of the categorical features\n

    which are oiginally in string format, with numerical features. The order of numbering

    is not under control\n

    Use this function to produce feed for the significace_test(df, str_target_name) function\n

    inputs:\n

    df: data frame\n

    return:\n

    data frame with replaced content

    '''



    print('replace the content of the categorical features with numeric value')



    # get the name of the columns of the data frame

    columns_name = df.columns



    #

    for member in columns_name:

        # check if the feature is categorical type

        if df[member].dtypes.name == 'category':

            # get the number of unique values in each column

            number_of_unique_elements_in_column = len(df[member].unique())

            msubpartition()

            print('Number of unique elements in %s : %d' % (member, number_of_unique_elements_in_column))

            # define a key list for the replacing dictionay

            mkeys = list(df[member].unique())

            # define a valude list for replacing dictionary

            mvalues = range(0, number_of_unique_elements_in_column)

            print('key for dictionary:', mkeys)

            print('value for dictionary:', mvalues)

            # create a dictionary for replacement

            replacing_dictionary = dict(zip(mkeys, mvalues))

            print('the dic:', replacing_dictionary)

            

            # replace the values

            df[member].replace(replacing_dictionary, inplace=True)



    return df



#------------------------------------------------------------------------------------------------------------------------------



def print_statistical_analysis_result(p_value, str_feature_1_name, str_feature_2_name):

    '''

    Auxiliri function for siginificance_test() in order to print the results

    '''

    print('Null Hypothesis: There is no statistical significance between %s and %s' % (str_feature_1_name, str_feature_2_name))

    print('The p-value for %s vs %s is %f' % (str_feature_1_name, str_feature_2_name, p_value))

    if p_value >= .05:

        print('The null hypothesis is Approved')

        print('Result: There is no statistical significance between %s and %s' % (str_feature_1_name, str_feature_2_name))

    else:

        print('the null hypothesis is Rejected')

        print('Result: There is a statistical significance between %s and %s' % (str_feature_1_name, str_feature_2_name))



#------------------------------------------------------------------------------------------------------------------------------



def print_correlation_result(correlation_value, str_feature_1_name, str_feature_2_name):

    '''

    This is an Auxility function to print the result of computed correlation value

    in significance_test() function.

    '''

    # state 1

    if -1 <= correlation_value and correlation_value < -0.75:

        print('%s and %s has a excellent negative correlation, correlation value: %f' % (str_feature_1_name, str_feature_2_name, correlation_value))

    # state 2

    if -0.75 <= correlation_value and correlation_value < -0.5:

        print('%s and %s has a good negative correlation, correlation value: %f' % (str_feature_1_name, str_feature_2_name, correlation_value))

    # state 3

    if -0.5 <= correlation_value and correlation_value < -0.25:

        print('%s and %s has a medium negative correlation, correlation value: %f' % (str_feature_1_name, str_feature_2_name, correlation_value))

    # state 4

    if -0.25 <= correlation_value and correlation_value < 0:

        print('%s and %s has a weak negative correlation, correlation value: %f' % (str_feature_1_name, str_feature_2_name, correlation_value))

    # state 5

    if 0 <= correlation_value and correlation_value < 0.25:

        print('%s and %s has a weak positive correlation, correlation value: %f' % (str_feature_1_name, str_feature_2_name, correlation_value))

    # state 6

    if 0.25 <= correlation_value and correlation_value < 0.5:

        print('%s and %s has a medium positive correlation, correlation value: %f' % (str_feature_1_name, str_feature_2_name, correlation_value))

    # state 7

    if 0.5 <= correlation_value and correlation_value < 0.75:

        print('%s and %s has a good positive correlation, correlation value: %f' % (str_feature_1_name, str_feature_2_name, correlation_value))

    # state 8

    if 0.75 <= correlation_value and correlation_value <= 1:

        print('%s and %s has a excellent positive correlation, correlation value: %f' % (str_feature_1_name, str_feature_2_name, correlation_value))



#------------------------------------------------------------------------------------------------------------------------------



def test_feature_has_normal_distributed_population(df_feature_1, df_feature_2):

    '''

    An Auxiliry function to check whether the inputs have normally distributed

    populations or not. To do such a test scipy.stats.shapiro() is used

    The Shapiro-Wilk test tests the null hypothesis that the data was drawn 

    from a normal distribution.

    inputs:

    two feature that we are interested in

    retruns:

    True: if both of pvalues have pvlaue>.05 (null hypothesis is approved)

    False : if one of pvalues have pvlaue<.05 (null hypothesis is rejected)

    '''

    

    # get shpiro pvalue for feature 1

    _, pvalue_shapiro_1 = shapiro(df_feature_1)

    # get shpiro pvalue for feature 1

    _, pvalue_shapiro_2 = shapiro(df_feature_2)



    if pvalue_shapiro_1 > .05 and pvalue_shapiro_2 > .05:

        return True

    else:

        return False



#------------------------------------------------------------------------------------------------------------------------------



def test_two_features_have_same_population_varince(df_feature_1, df_feature_2):

    '''

    An auxiliry function for significance_test() to find out if given two features

    have the same population variance or not. This is done by performing

    scipy.stats.levene() and scipy.stats.bartlett(*args) test.

    inputs: 

    data frame features which we are intersted in

    '''



    # get the normal population test result

    normal_population_test = test_feature_has_normal_distributed_population(df_feature_1, df_feature_2)



    if (normal_population_test == True):

        # bartlett test is uesed when feature have nomally distributed populations

        _, pvalue = bartlett(df_feature_1, df_feature_2)

    else:

        # levene test is uesed when feature Do Not have nomally distributed populations

        _, pvalue = levene(df_feature_1, df_feature_2)



    if pvalue > .05:

        # the null hypothesis is approved

        return True

    else:

        # the null hypothesis is rejected

        return False



#------------------------------------------------------------------------------------------------------------------------------



def significance_test_all_features_vs_target(df, str_target_name):

    '''

    Perform an statistical significance test for feature1 vs feature2.

    compute p-value for all featrues vs target. It automaically selects which 

    algorithm to use\n

    inputs:\n

    df: data frame\n

    str_target_name: tha name of the colunm feature as string

    '''

    

    print('Significance test for all features vs target')

    msubpartition()

    print('***WARNING:***: the category of all features should be int64, float64 or category\nOtherwise the code must be modified')

    

    # get the name of the columns of the data frame

    columns_name = df.columns

    # get the type of the target feature

    type_of_target_feature = df[str_target_name].dtypes.name



    #

    for member in columns_name:

        # get the type of the feature

        type_of_the_feature = df[member].dtypes.name

        # state 1

        if type_of_the_feature == 'category' and type_of_target_feature == 'category' and member!=str_target_name:

            msubpartition()

            print('%s vs %s' % (member, str_target_name))

            print('Since both %s and %s are cotegorical featrues then Chi-Squre test is used for Significance test' %(member, str_target_name))



            # reshape the feature since chi 2 requies 2D array and feature list is 1D

            # for this matter I used numpay arraies

            feature_array = np.array(df[member])

            feature_array = feature_array.reshape((feature_array.shape[0], 1))

            # apply chi2, get the p-value

            _, p_value = chi2(feature_array, df[str_target_name])

            

            # print results

            print_statistical_analysis_result(p_value, member, str_target_name)

        # state 2

        if (type_of_the_feature == 'int64' or type_of_the_feature == 'float64') and type_of_target_feature == 'category' and member!=str_target_name:

            msubpartition()

            print('%s vs %s' % (member, str_target_name))

            print('type of %s is numreric and type of %s is category' % (member, str_target_name))

            # check wether target featur has up to two unique value or more

            if len(df[str_target_name].unique()) <= 2:

                print('Since target feature has up to 2 unique value scipy.stats.ttest_ind() will be uesed')

                # test whether both features have same variance

                same_varicane_test = test_two_features_have_same_population_varince(df[member], df[str_target_name])

                # apply ttest_ind(), get the pvalue

                _, p_value = ttest_ind(df[member], df[str_target_name], equal_var=same_varicane_test)

                # print results

                print_statistical_analysis_result(p_value, member, str_target_name)



            elif len(df[str_target_name].unique()) > 2:

                print('Since target feature has more than 2 unique value one way ANOVA (f_oneway() or kruskal()) will be uesed')

                # check if both features normolly distributed Population or not:

                both_features_normally_distributed_population = test_feature_has_normal_distributed_population(df[member], df[str_target_name])

                # check if both features have same Population variance

                features_have_same_population_variance = test_two_features_have_same_population_varince(df[member], df[str_target_name])



                # if both features have same population variace and are form a normally distributed population

                # we can use ANOVA TEST: scipy.stats.f_oneway() 

                # other wise we have to use: scipy.stats.kruskal()

                # BASED ON THE SKLEARN DOCUMENTATION



                if both_features_normally_distributed_population == True and features_have_same_population_variance == True:

                    # apply scipy.stats.f_oneway() 

                    _, p_value = f_oneway(df[member], df[str_target_name])

                    # print the results

                    print_statistical_analysis_result(p_value, member, str_target_name)

                else:

                    # apply scipy.stats.kruskal()

                    _, p_value = kruskal(df[member], df[str_target_name], nan_policy='raise')

                    # print the results

                    print_statistical_analysis_result(p_value, member, str_target_name)

        # state 3

        if (type_of_the_feature == 'int64' or type_of_the_feature == 'float64') and (type_of_target_feature == 'int64' or type_of_target_feature == 'float64') and member!=str_target_name:

            msubpartition()

            print('%s vs %s' % (member, str_target_name))

            print('type of both %s and %s are numeric. So, scipy.stats.pearsonr() will be used for significance test' % (member, str_target_name))

            # apply scipy.stats.pearsonr() 

            correletion_value, p_value = pearsonr(df[member], df[str_target_name])

            # print the results

            print_statistical_analysis_result(p_value, member, str_target_name)

            print_correlation_result(correletion_value, member, str_target_name)

    

        #

        if type_of_the_feature == 'category' and (type_of_target_feature == 'int64' or type_of_target_feature == 'float64') and member!=str_target_name:

            msubpartition()

            print('%s vs %s' % (member, str_target_name))

            print('type of %s is category and type of %s is numeric' % (member, str_target_name))

            # check wether target featur has up to two unique value or more

            if len(df[member].unique()) <= 2:

                print('Since categorical feature has up to 2 unique value scipy.stats.ttest_ind() will be uesed')

                # test whether both features have same variance

                same_varicane_test = test_two_features_have_same_population_varince(df[member], df[str_target_name])

                # apply ttest_ind(), get the pvalue

                _, p_value = ttest_ind(df[member], df[str_target_name], equal_var=same_varicane_test)

                # print results

                print_statistical_analysis_result(p_value, member, str_target_name)



            elif len(df[member].unique()) > 2:

                print('Since categorical feature has more than 2 unique value one way ANOVA (f_oneway() or kruskal()) will be uesed')

                # check if both features normolly distributed Population or not:

                both_features_normally_distributed_population = test_feature_has_normal_distributed_population(df[member], df[str_target_name])

                # check if both features have same Population variance

                features_have_same_population_variance = test_two_features_have_same_population_varince(df[member], df[str_target_name])



                # if both features have same population variace and are form a normally distributed population

                # we can use ANOVA TEST: scipy.stats.f_oneway() 

                # other wise we have to use: scipy.stats.kruskal()

                # BASED ON THE SKLEARN DOCUMENTATION



                if both_features_normally_distributed_population == True and features_have_same_population_variance == True:

                    # apply scipy.stats.f_oneway() 

                    _, p_value = f_oneway(df[member], df[str_target_name])

                    # print the results

                    print_statistical_analysis_result(p_value, member, str_target_name)

                else:

                    # apply scipy.stats.kruskal()

                    _, p_value = kruskal(df[member], df[str_target_name], nan_policy='raise')

                    # print the results

                    print_statistical_analysis_result(p_value, member, str_target_name)





#------------------------------------------------------------------------------------------------------------------------------

# Implementing Significace Test will be done in three steps



#Step 1

# replace the content of the categorical features with numeric value

# otherwise you can just change the content of the featuers in an spread sheet 

df_significace_test_feed = replace_categerical_features_content_with_numbers(df)



# Step 2

# modify data type

# since the content of feed data frame is replaced type modification is required

df_significace_test_feed = modify_data_type(df_significace_test_feed, 

    ['workclass', 'education', 'Marital-status',

    'occupation', 'relationship', 'race', 'sex', 'Native-country', 

    'income_exceeds_50K'])



# Step 3

# significance test for all featrues vs target

significance_test_all_features_vs_target(df_significace_test_feed, 'income_exceeds_50K')

def create_dummies(df, str_target_name):

    '''

    create dummy variables for categorical variables\n

    inputs:\n

    df: data frame\n

    str_target_name: name of the target feature as string\n

    \nreturn:

    a data frame with dummy variables included and index and categorical variables excluded

    '''



    print('Hot Encoding: create dummies')

    msubpartition()

    print('***ATTENTION***: mannual modification is needed to use this function before manipulation\nIf the type of the features are any thing except int64, float64 and category')

    

    # define an empty data frame as out put of the functioa

    df_final = pd.DataFrame() 

    

    # get the name of the columns of the data frame

    columns_name = df.columns

    

    # add dummy categorical features to df_final

    for member in columns_name:

        # include only categoical features 

        # getattr(df, member is equel to df.member

        if getattr(df, member).dtypes.name == 'category' and member != str_target_name:

            dummy_df_i = pd.get_dummies(df[member], prefix=member)

            df_final = pd.concat([df_final, dummy_df_i], axis=1)



    # add numeric features to df_final

    for member in columns_name:       

        # include only numeric features 

        # getattr(df, member is equel to df.member

        if (df[member].dtypes.name == 'int64' or df[member].dtypes.name == 'float64') and (member != str_target_name):

            df_final = pd.concat([df_final, df[member]], axis=1)

            

    # add target feature to df_final

    df_final = pd.concat([df_final, df[str_target_name]], axis=1)

    

    msubpartition()

    #print('df_final head:\n', df_final.head())

    print('the shape of the df_final', df_final.shape)

    msubpartition()

    print('the columns name of the df_final:\n', df_final.columns)



    return df_final
# hot encoding: creating dummy variables

df = create_dummies(df, 'income_exceeds_50K')
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectKBest, RFE

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score

from operator import itemgetter



def get_features_target_df(df, str_target_name):

    '''

    split data frame to features and target data frame\n

    inputs:\n

    df: data frame\n

    str_target_name: the name of the target column as string\n

    return:\n

    df_features: features data frame\n

    df_target: target data frame (type: pandas.core.series.Series)

    '''



    print('split data frame to two data frame: features and target data frame')



    # get features data frame

    df_features = df.drop([str_target_name], axis=1)

    # get target data frame

    df_target = df[str_target_name]



    return df_features, df_target



#-------------------------------------------------------------------------------------------------------------------------



def train_test_splitor(df_features, df_target, train_2_test_ratio, bool_verbose=True):

    '''

    split features and target data frame to train and test data frames\n

    inputs:\n

    df_features: featurs data frame\n

    df_target: target data frame\n

    return:\n



    '''

    if bool_verbose:

        print('split the features and target data frames to 4 data frames:\n train, test featrues and target data frames')



    # get features and target data frames for train and test

    df_features_train, df_features_test, df_target_train, df_target_test = train_test_split(df_features, df_target, train_size=train_2_test_ratio, test_size=1-train_2_test_ratio, random_state=1, shuffle=True)



    if bool_verbose:

        # print the shapes of the shapes of the created data frame

        msubpartition()

        print('the shape of the df_features_train', df_features_train.shape)

        print('the shape of the df_features_test', df_features_test.shape)

        print('the shape of the df_target_train', df_target_train.shape)

        print('the shape of the df_target_test', df_target_test.shape)



    return df_features_train, df_features_test, df_target_train, df_target_test



#-------------------------------------------------------------------------------------------------------------------------



def mpearson_feature_selector(df_features, df_target, prefered_number_of_features, bool_verbose=True):

    '''

    find the n (prefered_number_of_features) of the features with hisghest 

    correlation with the target | \n

    inputs:\n 

    df_features: feature's data frame. how to get: df_features=df.drop([str_target_value], axis=1)\n

    df_target: target data frame. How to get: df_target=df[str_target_value]\n

    prefered_number_of_features: number of features with highest correlation with the

    target | \n

    returns 2:\n

    selected_features: a list of the feature's name with highest correlation with

    the target in ascending order

    associated_correlation_value: correlation value of the selected features\n

    Also prints the returned values in the terminal

    '''

    if bool_verbose:

        print('Running pearson feature selector ...')

    

    # define a list for appending the corralation of each feature with Y (the target)

    correlation_list = []

    # get the name of the features in the df_features

    features_name = df_features.columns.tolist()



    # compute the correlation with df_target for each feature

    for i in features_name:

        cor = np.corrcoef(df_features[i], df_target)[0, 1]

        correlation_list.append(cor)



    # replace NaN with 0

    correlation_list = [0 if np.isnan(i) else i for i in correlation_list]



    # how to selcet the n (prefered_number_of_features) of the features with highest correlation?

    # 1. get the absolute of the coorelation_list: np.abs(correlation_list)

    # 2. get the soreted argoman of the above nparray: np.argsort(np.abs(correlation_list))

    # 3. get the subset of the df_features including the above columns:

    #    df_features.iloc[ :,np.argsort(np.abs(correlation_list))[-prefered_number_of_features:] ]

    # 4. get the column names of the above data frame: 

    #    df_features.iloc[ :,np.argsort(np.abs(correlation_list))[-prefered_number_of_features:] ].columns.tolist()

    selected_features = df_features.iloc[ :,np.argsort(np.abs(correlation_list))[-prefered_number_of_features:] ].columns.tolist()



    # get index for the n high correlation value

    high_corr_indexes = np.argsort(np.abs(correlation_list))[-prefered_number_of_features:]

    # define a list for the n high correlation value

    associated_correlation_value = [correlation_list[i] for i in high_corr_indexes]

    

    if bool_verbose:

        msubpartition()

        print('selected features:\n', selected_features)

        msubpartition()

        print('associated correlation values with selected features:\n', associated_correlation_value)



    return selected_features, associated_correlation_value



#-------------------------------------------------------------------------------------------------------------------------



def KBest_feature_selector(df_features, df_target, prefered_number_of_features, bool_verbose=True):

    ''' 

    This functoion is a feature selection function, that selects n (prefered_number_of_features) features with highest score. The score function should be selected wisely. | \n 

    inputs:\n 

    df_features: feature's data frame. how to get: df_features=df.drop([str_target_value], axis=1)\n

    df_target: target data frame. How to get: df_target=df[str_target_value]\n

    prefered_number_of_features: number of features with highest correlation with the target | \n

    returns:\n

    \nreturns a list including the selected features\n

    Also prints the returned values in the terminal

    '''

    

    if  bool_verbose:

        print('KBest feature seloctor ...')

        msubpartition()

        print('***WARINING***: This function needs modification before manipultion')

        msubpartition()

        print('***WARINIG***\nPay careful attention in selection of the score function (score_func)')



    # scale each feature in range between zero and one.

    X_norm = MinMaxScaler().fit_transform(df_features)

    # SelectKBest() selects features according to the k highest scores

    #

    # ***WARNING***: pay careful attention in selection of the score function (score_func)

    # available score functions:

    # For regression: f_regression, mutual_info_regression

    # For classification: chi2, f_classif, mutual_info_classif

    # ***WARNING***: ecah score function should be imported first

    # define the SelectKBest

    KBest_selector = SelectKBest(chi2, k=prefered_number_of_features)

    # fit to data. Run score function on (df_features, df_target) and get the appropriate features

    KBest_selector.fit(X_norm, df_target)

    # Get a mask, or integer index, of the features selected

    boolean_mask = KBest_selector.get_support()

    # define a list for selected features

    selected_features = df_features.loc[:,boolean_mask].columns.tolist()



    if  bool_verbose:

        # print the name of the score function 

        msubpartition()

        print('the seceted score function:', KBest_selector.score_func)

        # print selected features

        msubpartition()

        print('selected features by KBest:\n', selected_features)    

 

    return selected_features



#-------------------------------------------------------------------------------------------------------------------------



def RFE_feature_selector(df_features, df_target, prefered_number_of_features, bool_verbose=True):

   ''' 

    Feature ranking with recursive feature elimination. 

    inputs:\n

    df_features: feature's data frame. how to get: df_features=df.drop([str_target_value], axis=1)\n

    df_target: target data frame. How to get: df_target=df[str_target_value]\n

    prefered_number_of_features: number of features with highest correlation with the target | \n

    returns:\n

    \nreturns a list including the selected features 

    \nAlso prints the returned values in the terminal

   '''

   if bool_verbose:

        print('RFE featrue selector ...')



   # scale each feature in range between zero and one.

   X_norm = MinMaxScaler().fit_transform(df_features)

   # define RFE selector

   # estimator= LogisticRegression() | LinearRegression()

   # logisticRegression taks too much long to respond

   rfe_selector = RFE(estimator=LinearRegression(), n_features_to_select=prefered_number_of_features, step=1)

   # fit to data

   rfe_selector.fit(X_norm, df_target)

   # Get a mask, or integer index, of the features selected

   boolean_mask = rfe_selector.get_support()

   # define a list for selected features

   selected_features = df_features.loc[:,boolean_mask].columns.tolist()

   

   if bool_verbose:

        # print the selected features

        msubpartition()

        print('selected features by RFE:\n', selected_features)

   

   return selected_features



#-------------------------------------------------------------------------------------------------------------------------



def get_selected_features(df_features, df_target):

    '''

    Find the best set of selected features with three methods: Pearson, KBest, RFE\n

    Inorder to feed them to logestic regression model

    '''

    print('Find the best set of selected features')

    msubpartition()

    print('This may take a while to compute ...')

    

    # define a list for data:

    data = []



    for i in [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:

        # selected features from Pearson method

        selected_features1, _ = mpearson_feature_selector(df_features, df_target, 

                                                        prefered_number_of_features = i,

                                                        bool_verbose=False)



        # add selected_features1 to data

        data.append(['pearson', i, selected_features1])



        # selected features from KBest method

        selected_features2 = KBest_feature_selector(df_features, df_target,

                                                    prefered_number_of_features=i,

                                                    bool_verbose=False)

        # add selected_features2 to data

        data.append(['KBest', i, selected_features2])

    

        # selected features form RFE

        selected_features3 = RFE_feature_selector(df_features, df_target,

                                                  prefered_number_of_features = i,

                                                  bool_verbose=False)

        # add selected_features3 to data

        data.append(['RFE', i, selected_features3])



    return data



#-------------------------------------------------------------------------------------------------------------------------



def mlogistic_regression(df_features_train, df_target_train, df_features_test, df_target_test, maximimu_iteriation=200, bool_verbose=False, random_seed=1, bool_plot_curve=True, bool_mverbose=True):

    '''

    Logistic regression from scikit learn library...

    to be continued...

    '''

    if bool_mverbose:

        print('Logistic Regression ...')



    # define the logistic regression model

    logistic_regression_model = LogisticRegression(max_iter=100, 

                                                   verbose=bool_verbose, 

                                                   random_state=random_seed)



    # train the logistic regression model 

    logistic_regression_model.fit(df_features_train, df_target_train)

    

    # compute the prediction of the model for the df_features_test

    model_prediction = logistic_regression_model.predict(df_features_test)



    # # print the score of the traing data

    # train_data_score = logistic_regression_model.score(df_features_train, df_target_train)

    # msubpartition()

    # print('training data score:', train_data_score)



    # # print the score of the test data

    # test_data_score = logistic_regression_model.score(df_features_test, df_target_test)

    # msubpartition()

    # print('test data score:', test_data_score)





    #Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores

    area_under_ROC_curve = roc_auc_score(df_target_test, model_prediction)

    if bool_mverbose:

        # print area_under_ROC_curve

        msubpartition()

        print('area_under_ROC_curve:', area_under_ROC_curve)



    #Compute Receiver operating characteristic (ROC)

    false_positive_rates, true_positive_rates, thresholds = roc_curve(df_target_test, logistic_regression_model.predict_proba(df_features_test)[:, 1])



    # obtain the confusion matrix

    my_confusion_matrix = confusion_matrix(df_target_test, model_prediction)

    if bool_mverbose:

        # print the confusion matrix

        msubpartition()

        print('the confusion matrix:\n', my_confusion_matrix)

    

    if bool_mverbose:

        # print the accuracy_score

        msubpartition()

        print('accuracy_score of the model:', accuracy_score(df_target_test, model_prediction))



    if bool_plot_curve:

        plt.figure('Receiver Operating Characteristic Curve', figsize=(8, 6))

        plt.grid(False)

        # the blues curve

        plt.plot(false_positive_rates, true_positive_rates, label='logistic regression (area=%0.2f)'% area_under_ROC_curve)

        # the red curve

        plt.plot([0, 1], [0, 1], 'r--')

        plt.axis([0.0, 1.0, 0.0, 1.05])

        plt.xlabel('False positive rate')

        plt. ylabel('True positive rate')

        plt.title('Receiver Operating Characteristic Curve')

        plt.legend(loc='lower right')

    

    return area_under_ROC_curve



#-------------------------------------------------------------------------------------------------------------------------



def train_model_compute_auc(df_features, df_target, data_selected_features):

    '''

    Training the model with input selected features

    '''

    print('Training the model with input selected features')

    msubpartition()

    print('This may take a while to compute ...')

    

    # create a list for out data

    # infact we are going to update the input data list with Area Under Cover

    output_list = []



    for member in data_selected_features:



        # get the selected features from data_selected_features 

        selected_features = member[2]



        # get the selected featrues data frame to feed it to the model

        df_selected_features = df_features[selected_features]



        # get data frames for train and test based on df_selected_features

        df_features_train, df_features_test, df_target_train, df_target_test = train_test_splitor(df_selected_features, df_target, .7, bool_verbose=False)



        # train the logistic model

        # get the area under curve for each selected feture

        auc = mlogistic_regression(df_features_train, df_target_train, df_features_test, df_target_test, maximimu_iteriation=200, bool_verbose=False, random_seed=1, bool_plot_curve=False, bool_mverbose=False)



        # append the auc to the output data list

        # each member of the list = [method_name, #_selected_features, [list of selected features], auc]

        output_list.append([member[0], member[1], member[2], auc])

    

    return output_list



#-------------------------------------------------------------------------------------------------------------------------



def find_the_best_model(data_list):

    '''

    Find the best model with respect to Area Under Curve\n

    inputs:\n

    data_list: list, each member of the input list = [method_name, #_selected_features, [list of selected features], auc]\n

    return:\n

    list: (The best AUC, Method Name, Number of selected features, Selected features)

    '''

    print('Find the best model based on the provided AUC')



    # sort data list with respect to AUC

    data_list = sorted(data_list, key=itemgetter(3))

    

    # print the best result

    msubpartition()

    print('The best AUC: %.3f\n' % data_list[-1][3],

          'Method Name: %s\n' % data_list[-1][0],

          'Number of selected features: %d\n' % data_list[-1][1], 

          'Selected features: %s' % data_list[-1][2]

        )



    # get Pearson method feature count and auc to plot them. each member: [count_of_selected_features, auc]

    Pearson_featureCount_auc = [[member[1], member[-1]] for member in data_list if member[0]=='pearson']

    # sort the list with respect to selectedfeature count

    # (originally it should be sorted, but to be sure)

    Pearson_featureCount_auc = sorted(Pearson_featureCount_auc, key=itemgetter(0))

    Pearson_featureCount = [member[0] for member in Pearson_featureCount_auc]

    Pearson_auc = [member[1] for member in Pearson_featureCount_auc]



    # get KBest method feature count and auc to plot them.  each member: [count_of_selected_features, auc]

    KBest_featureCount_auc = [[member[1], member[-1]] for member in data_list if member[0]=='KBest']

    # sort the list with respect to selectedfeature count

    # (originally it should be sorted, but to be sure)

    KBest_featureCount_auc = sorted(KBest_featureCount_auc, key=itemgetter(0))

    KBest_featureCount = [member[0] for member in KBest_featureCount_auc]

    KBest_auc = [member[1] for member in KBest_featureCount_auc]



    # get RFE method feature count and auc to plot them. each member: [count_of_selected_features, auc]

    RFE_featureCount_auc = [[member[1], member[-1]] for member in data_list if member[0]=='RFE']

    # sort the list with respect to selectedfeature count

    # (originally it should be sorted, but to be sure)

    RFE_featureCount_auc = sorted(RFE_featureCount_auc, key=itemgetter(0))

    RFE_featureCount = [member[0] for member in RFE_featureCount_auc]

    RFE_auc = [member[1] for member in RFE_featureCount_auc]



    # plot the AUC vs number of selected features

    plt.figure('AUC vs number of selected features', figsize=(12, 6))

    plt.grid(False)

    # plot Pearson

    if len(Pearson_featureCount_auc) > 0:

        plt.plot(Pearson_featureCount, Pearson_auc, label='Pearson', marker='o')

    # plot KBest

    if len(KBest_featureCount_auc) > 0:

        plt.plot(KBest_featureCount, KBest_auc, label='KBest', marker='v')

    # plot RFE

    if len(RFE_featureCount_auc) > 0:

        plt.plot(RFE_featureCount, RFE_auc, label='RFE', marker='+')



    

    plt.xlabel('Number of Selected Features')

    plt.ylabel('Area Under Curve')

    plt.legend()

    plt.title('Area Under Curve vs Number of Selected Features')



    # return: The best AUC, Method Name, Number of selected features, Selected features

    return data_list[-1][3], data_list[-1][0], data_list[-1][1], data_list[-1][2]
# get the features and target data frames:

df_features, df_target = get_features_target_df(df, 'income_exceeds_50K')



# Find the best set of selecting features method and number of selected features in 3 steps.



# Step 1

# get the selected features data list

# each member = [method_name, #_selected_features, [list of selected features]]

selected_features_data = get_selected_features(df_features, df_target)



# Step 2

# get the Area Under Curve

# each member = [method_name, #_selected_features, [list of selected features], auc]

data_list = train_model_compute_auc(df_features, df_target, selected_features_data)



# Step 3

# find the best model based on the AUC value

# input data format: [method_name, #_selected_features, [list of selected features], auc]

Best_AUC, Best_Method_Name, Number_of_selected_features, Best_Selected_features = find_the_best_model(data_list)
# get the selected featrues data frame to feed it to the model

df_selected_features = df_features[Best_Selected_features]



# get data frames for train and test based on df_selected_features

df_features_train, df_features_test, df_target_train, df_target_test = train_test_splitor(df_selected_features, df_target, .7)



# Logistic regression model and train

mlogistic_regression(df_features_train, df_target_train, df_features_test, df_target_test, maximimu_iteriation=200, bool_verbose=False, random_seed=1, bool_plot_curve=True)