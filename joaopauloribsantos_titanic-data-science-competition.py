## Data Analysis and Munging/ Wrangling

import pandas as pd

import numpy as np



## Data Visualization

import seaborn as sns

import matplotlib.pyplot as plt

import seaborn as sns



## Machine Learning Models

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_curve, auc

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from collections import OrderedDict

from IPython.display import Image  

from sklearn.tree import export_graphviz

## Location/ Path where documents are stored

path = "/kaggle/input/titanic/"



## Dataframes that correspond to each spreadsheet

df_train = pd.read_csv(path + 'train.csv')

df_test = pd.read_csv(path + 'test.csv')

df_gender_submission = pd.read_csv(path + 'gender_submission.csv')
## Dataframe 'df_train'

df_train.head(3)
## Dataframe 'df_test'

df_test.head(3)
## Dataframe 'df_gender_submission'

df_gender_submission.head(3)
## Set the passenger id as the index of the dataframes

df_train.set_index(['PassengerId'], inplace = True)

df_test.set_index(['PassengerId'], inplace = True)

df_gender_submission.set_index(['PassengerId'], inplace = True)
dict_columns = {'Pclass': 'Pass_Class', 

                'Ticket': 'Ticket_Id', 

                'Fare': 'Pass_Fare',

                'Cabin': 'Cabin_Id', 

                'Embarked' : 'Port_Embark',

                'Parch' : 'Par_Child_Aboard', 

                'SibSp': 'Sibli_Aboard'}



df_train.rename(columns = dict_columns, inplace = True)



df_test.rename(columns = dict_columns, inplace = True)
## Checking the dataframes structures like number of rows and columns

print("df_train: ", df_train.shape, 

      "\ndf_test: ", df_test.shape, 

      "\ndf_gender_submission: ", df_gender_submission.shape)
## Percentagem of null values in dataframe df_train

df_train.isnull().sum() / len(df_train) * 100
## Percentagem of null values in dataframe df_train

df_test.isnull().sum() / len(df_test) * 100
## Percentage of null values in dataframe df_train

df_gender_submission.isnull().sum() / len(df_gender_submission) * 100
## Check if there is any relationship in the unnamed cabins, with the types of accommodation



# Creating a dataframe with only the information from the columns 'Pclass' and 'Cabin'

df_train_null_values_Cabin =  df_train.loc[:, ['Pass_Class', 'Cabin_Id']]



# Creating a new column to identify whether the field, which has any string, is null or not

df_train_null_values_Cabin['is_null'] = np.where(df_train_null_values_Cabin['Cabin_Id'].isnull(),1,0 )



# Creating a crosstab with the information of the classes of the cabins, and the number 

# of cabins with unknown name / number

pd.crosstab(df_train_null_values_Cabin["Pass_Class"],df_train_null_values_Cabin["is_null"],margins=True)
## Replacing the null values in the column 'Cabin' with the value 'Unknown'

df_train['Cabin_Id'].fillna('Unknown', inplace = True)

df_train.head(3)
## Creating the column Deck

df_train['Deck'] = df_train['Cabin_Id'].apply(lambda nm : nm[0])
## Since the field only contains some specific types of values, more precisely three, 

## it is possible to verify the distribution of these data

df_train['Port_Embark'].value_counts(dropna = False)
## Replacing the null values in the column 'Embarked' with the value 'Not Informed'/ 'Not Info' 



## Applying the changes

df_train['Port_Embark'].fillna(value = 'Not Info', inplace = True)

df_train[df_train['Age'].isnull()]
## Since there are many passengers without age information, the median will be adopted to replace these null values.

df_train['Age'].fillna(df_train['Age'].median(), inplace = True)

df_train.head(3)
df_test.isnull().sum()
## It seems that there is only one passenger of the third class, 

## so we apply the median based on the values of the third class.

df_test[df_test['Pass_Fare'].isnull()]
median = df_test.loc[df_test['Pass_Class'] == 3, 'Pass_Fare'].mean()

## Assign the passenger Thomas Storey, the fare value equal to the average of his class

df_test['Pass_Fare'].fillna(median, inplace = True)



## Replacing the null values in the column 'Cabin' with the value 'Unknown'

df_test['Cabin_Id'].fillna('Unknown', inplace = True)



## Since there are many passengers without age information, the median will be adopted to replace these null values.

df_test['Age'].fillna(df_test['Age'].median(), inplace = True)



## Applying the treatment related to column 'Deck'

df_test['Deck'] = df_test['Cabin_Id'].apply(lambda nm : nm[0])
## Check if there are still null values in the dataframes.

print("Are there null values in the dataframe df_train ?: ", df_train.isnull().values.any())

print("Are there null values in the dataframe df_test ?: ", df_test.isnull().values.any())
# The previous code showed the boxplot of all variables / columns.

plt.figure(figsize = (18,13))

sns.boxplot(data = df_train)



plt.show()
def fn_validating_dataframe(p_df_dataframe):

    """

        Description:

            Validates information related to the 

            dataframe.



        Keyword arguments:

            p_df_dataframe -- the dataframe 



        Return:

            None



        Exception:

            Validates whether the object passed is a pandas dataframe;

            Validates that the dataframe is empty.

    """

    

    if not (isinstance(p_df_dataframe, pd.DataFrame)):

            raise Exception("The past object is not a Pandas Dataframe")

            

    if p_df_dataframe.empty:

            raise Exception("The dataframe is empty")
def fn_number_of_outliers_per_dataframe(p_df_dataframe):

    """

        Description:

            Validates the number of outliers on a dataframe



        Keyword arguments:

            p_df_dataframe -- the dataframe 



        Return:

            Object with the number of outliers per column



        Exception:

            Validates whether the object passed is a pandas dataframe;

            Validates that the dataframe is empty.

    """

    

    fn_validating_dataframe(p_df_dataframe)

        

    Q1 = p_df_dataframe.quantile(0.25)

    Q3 = p_df_dataframe.quantile(0.75)

    IQR = Q3 - Q1

    sr_out = ((p_df_dataframe < (Q1 - 1.5 * IQR)) | (p_df_dataframe > (Q3 + 1.5 * IQR))).sum()

    return sr_out
fn_number_of_outliers_per_dataframe(df_train)
fn_number_of_outliers_per_dataframe(df_test)
  

def fn_catching_outliers(p_df_dataframe, p_column):

    """

    Description:

        Function that locates outliers in an informed dataframe.



    Keyword arguments:

        p_df_dataframe -- the dataframe 

        p_column -- the dataframe column



    Return:

        df_with_outliers -- Dataframe with the outliers located

        df_without_outliers -- Dataframe without the outilers

    

    Exception:

        None

    """

    # Check if the information passed is valid.

    fn_number_of_outliers_per_dataframe(p_df_dataframe)

    

    # Calculate the first and the third qurtile of the dataframe  

    Q1 = p_df_dataframe[p_column].quantile(0.25)

    Q3 = p_df_dataframe[p_column].quantile(0.75)    

  

    

    # Calculate the interquartile value

    IQR = Q3 - Q1

    

    #sr_out = ((p_df_dataframe < (Q1 - 1.5 * IQR)) | (p_df_dataframe > (Q3 + 1.5 * IQR))).sum()

    

    # Generating the fence hig and low values

    fence_high = Q3 + (1.5 * IQR)

    fence_low = Q1 - (1.5 * IQR)

    

    # And Finally we are generating two dataframes, onde with the outliers values and the second with the values within values

    df_with_outliers = p_df_dataframe[((p_df_dataframe[p_column] < fence_low) | (p_df_dataframe[p_column] > fence_high))]

    

    if df_with_outliers.empty:

        print("No outliers were detected.")

    

    return df_with_outliers
# Column 'Age'

df_out = fn_catching_outliers(df_train, 'Age')

df_out.head(5)
## Creating a function to assist in classification.

def age_definition(p_age):

    """

        Description:

            Function that classifies someone's age



        Keyword arguments:

            p_age -- Number that matches someone's age



        Return:

            Returns the age classification as Young, Child and etc.



        Exception:

            None

    """

    age = int(p_age)

    if age <= 2:

        return 'Infant'

    elif age <= 12:

        return 'Childreen'

    elif age <= 17:

        return 'Young'

    elif age <= 24:

        return 'Young Adult'

    elif age <= 44:

        return 'Adult'

    elif age <= 59:

        return 'Middle-Age'

    elif age <= 74:

        return 'Senior'

    elif age <= 90:

        return "Elder"

    else:

        return 'Extreme Old Age'



## Creating the new column

df_train['Age_Gr'] = df_train['Age'].apply(age_definition)



## Catching the outlier again

df_out = fn_catching_outliers(df_train, 'Age')



## Age classification based on outliers 

df_out['Age_Gr'].value_counts()
## Catching the outlier from dataframe 'Test'

df_out = fn_catching_outliers(df_test, 'Age')

df_out.head(3)
## Creating the new column

df_test['Age_Gr'] = df_test['Age'].apply(age_definition)



## Catching the outlier again

df_out = fn_catching_outliers(df_test, 'Age')



## Age classification based on outliers 

df_out['Age_Gr'].value_counts()
# Column 'Parents_Or_Childreeens_Aboard'

df_out = fn_catching_outliers(df_train, 'Par_Child_Aboard')

df_out.head(5)
df_out['Par_Child_Aboard'].value_counts()
## Column 'Siblings_Aboard'

df_out = fn_catching_outliers(df_train, 'Sibli_Aboard')

df_out.head(5)                   
df_out['Sibli_Aboard'].value_counts()
##Passenger_Fare   

df_out = fn_catching_outliers(df_train, 'Pass_Fare')

df_out.head(5)
## Main statistical metrics related to passenger tariffs

df_train['Pass_Fare'].describe()
df_train['Crew'] = np.where(df_train['Pass_Fare'] == 0, 1, 0)



df_test['Crew'] = np.where(df_test['Pass_Fare'] == 0, 1, 0)
## Check the quartiles

quantiles = [0.10, 0.25, 0.50, 0.70 ,0.75, 0.80, 0.90, 0.95, 0.99]

for q in quantiles:

    print("Quantile[", q*100,"%] :", df_train['Pass_Fare'].quantile(q))
## Main statistical metrics related to outliers, more specifically those related to passenger fares

df_out['Pass_Fare'].describe()
## Group the amount of outliers by social class

df_out['Pass_Class'].value_counts()
df_out[df_out['Pass_Class'] == 3 ]
## Number of people with the same ticket, more precisely the ticket 'CA. 2343 '

len(df_out[df_out['Ticket_Id'] == 'CA. 2343'])
# Create dictionary with the number of tickets on the ship

dt_ticket_identification = df_train['Ticket_Id'].value_counts().to_dict()

# Create column showing the number of passengers using the same ticket

df_train['Num_Same_Ticket'] = df_train['Ticket_Id'].map(lambda x: dt_ticket_identification[x])

# Update column value

df_train['Pass_Fare'] = np.where((df_train['Pass_Fare'] != 0)

                                      ,df_train['Pass_Fare']/df_train['Num_Same_Ticket']

                                      ,df_train['Pass_Fare'])
# Create dictionary with the number of tickets on the ship

dt_ticket_identification = df_test['Ticket_Id'].value_counts().to_dict()

# Create column showing the number of passengers using the same ticket

df_test['Num_Same_Ticket'] = df_test['Ticket_Id'].map(lambda x: dt_ticket_identification[x])

# Update column value

df_test['Pass_Fare'] = np.where((df_test['Pass_Fare'] != 0)

                                      ,df_test['Pass_Fare']/df_test['Num_Same_Ticket']

                                      ,df_test['Pass_Fare'])
## Number of people with the same ticket, more precisely the ticket 'CA. 2343 '

df_train[df_train['Ticket_Id'] == 'CA. 2343']
## Identify whether outliers still exist

df_out = fn_catching_outliers(df_train, 'Pass_Fare')

len(df_out)
## Show the first three rows of the dataframe

df_out.head(3)
## Show the most commom statistics about the dataframe of outliers

df_out['Pass_Fare'].describe()
## Group the amount of outliers by social class

df_out['Pass_Class'].value_counts()
df_out['Port_Embark'].value_counts()
df_out['Cabin_Id'].value_counts()
df_train['Is_Alone'] = np.where((df_train['Sibli_Aboard'] > 0) | 

                                (df_train['Par_Child_Aboard'] > 0), 0, 1)



df_train.head(3)
df_survivors = df_train['Survived'].value_counts().reset_index()



df_survivors.rename(columns = {'index': 'Survived', 

                               'Survived': 'Number'}, 

                    inplace = True)



df_survivors['Survived'] = np.where(df_survivors['Survived'] == 1, 'Yes', 'No')



df_survivors
ax = sns.barplot(x = "Survived", 

                 y = "Number", 

                 data = df_survivors)



plt.xlabel('Survivors') # add to x-label to the plot

plt.ylabel('Number of Passengers') # add y-label to the plot

plt.title('RMS Titanic passenger information') # add title to the plot



for p in ax.patches:

        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),

                    ha='center', va='bottom',

                    color= 'black')

plt.show()
df_gender_survivor = df_train.pivot_table( values = 'Survived',index = 'Sex', aggfunc = 'sum')

df_gender_survivor
ax = df_gender_survivor['Survived'].plot(kind='bar', figsize=(10, 6), color = ['orange', 'turquoise'])



ax.set_xticklabels(ax.get_xticklabels(), 

                   rotation = 0)



plt.xlabel('Survivors')



plt.ylabel('Number\n of\n Passengers', 

           labelpad = 50, 

           rotation = 0)



plt.title('Survivors by Gender')



for p in ax.patches:

        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.25, p.get_height()),

                    ha='center', va='bottom',

                    color= 'black')



plt.show()
df_gender_survivor = df_train.groupby(['Sex', 'Survived'])[['Survived']].count()



df_gender_survivor.rename(columns = {'Survived': 'Number'}, 

                          index={0: 'No', 

                                 1:'' 'Yes',

                                 'male' : 'Male',

                                 'female': 'Female'},

                    inplace = True)



df_gender_survivor
df = df_gender_survivor.reset_index()



plt.figure(figsize=(10, 10))



ax = sns.barplot(x = "Sex", y = "Number", hue = "Survived", data = df, palette="vlag")



ax.tick_params(axis = 'both', 

               which = 'major', 

               labelsize = 14)



for p in ax.patches:

        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.2, p.get_height()),

                    ha='center', va='bottom',

                    color= 'black', fontsize=14)



plt.xlabel('Gender',

           fontsize = 16)



plt.ylabel('Number\n of\n Passengers', 

           labelpad = 50, 

           rotation = 0, 

           fontsize = 16)



plt.title('List of Survivors and Deaths by Gender', 

          fontsize = 20)



plt.show()
df_gender_survivor = df_train.groupby(['Sex', 'Survived', 'Is_Alone'])[['Survived']].count()



df_gender_survivor.rename(columns = {'Survived': 'Number'}, 

                          index={0: 'No', 

                                 1:'' 'Yes',

                                 'male' : 'Male',

                                 'female': 'Female'},

                    inplace = True)



df_gender_survivor
df = df_gender_survivor.reset_index()



ax = sns.factorplot(x='Is_Alone', 

                    y='Number', 

                    hue='Survived', 

                    col='Sex', 

                    data = df, 

                    kind='bar',

                   palette = 'hls')





ax.set_xlabels('Passenger is Alone', fontsize = 14)



ax.set_ylabels('Number\n of\n Passengers', 

               labelpad = 60, 

               rotation = 0, 

               fontsize = 14)



axes = ax.axes.flatten()

axes[0].set_title("Survivor information for MALE passengers")

axes[1].set_title("Survivor information for FEMALE passengers")



plt.show()
df = df_train.loc[df_train['Survived'] == 1]



df_gender_survivor = df.groupby(['Sex', 'Age_Gr'])[['Survived']].count()



df_gender_survivor.rename(columns = {'Survived': 'Number'}, 

                          index={0: 'No', 

                                 1:'' 'Yes',

                                 'male' : 'Male',

                                 'female': 'Female'},

                    inplace = True)



df_gender_survivor
df = df_gender_survivor.reset_index()



plt.figure(figsize=(10, 10))



ax = sns.barplot(x = "Sex", y = "Number", hue = "Age_Gr", data = df, palette="hls")



ax.tick_params(axis = 'both', 

               which = 'major', 

               labelsize = 14)



for p in ax.patches:

        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.05, p.get_height()),

                    ha='center', va='bottom',

                    color= 'black', fontsize=14)



plt.xlabel('Gender',

           fontsize = 16)



plt.ylabel('Number\n of\n Passengers', 

           labelpad = 50, 

           rotation = 0, 

           fontsize = 16)







plt.title('Survivors according to their Age Group', 

          fontsize = 20)





plt.show()
df_gender_survivor = df_train.groupby(['Age_Gr'])[['Survived']].count()



df_gender_survivor.rename(columns = {'Survived': 'Number'}, 

                           inplace = True)



df_gender_survivor
df = df_gender_survivor.reset_index()



plt.figure(figsize=(15, 10))



ax = sns.barplot(x = "Age_Gr", y = "Number", data = df, palette="hls")



ax.tick_params(axis = 'both', 

               which = 'major', 

               labelsize = 14)



for p in ax.patches:

        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),

                    ha='center', va='bottom',

                    color= 'black', fontsize=14)



plt.xlabel('Age Group',

           fontsize = 16)



plt.ylabel('Number\n of\n Passengers', 

           labelpad = 50, 

           rotation = 0, 

           fontsize = 16)



plt.title('Passengers according to their Age Group', 

          fontsize = 20)



plt.show()
df_gender_survivor = df_train.groupby(['Deck', 'Survived'])[['Survived']].count()



df_gender_survivor.rename(columns = {'Survived': 'Number'}, 

                          index={0: 'No', 

                                 1:'' 'Yes'},

                           inplace = True)



df_gender_survivor
df = df_gender_survivor.reset_index()



ax = sns.factorplot(x='Deck', 

                    y='Number', 

                    col='Survived', 

                    data = df, 

                    kind='bar',

                   palette = 'hls')





ax.set_xlabels('Deck', fontsize = 14)



ax.set_ylabels('Number\n of\n Passengers', 

               labelpad = 60, 

               rotation = 0, 

               fontsize = 14)



axes = ax.axes.flatten()

axes[0].set_title("Number of DEATHS per deck")

axes[1].set_title("Number of SURVIVORS per deck")



plt.show()
df_gender_survivor = df_train.groupby(['Pass_Class', 'Survived'])[['Survived']].count()



df_gender_survivor.rename(columns = {'Survived': 'Number'}, 

                           inplace = True)



df_gender_survivor
df = df_gender_survivor.reset_index()

df['Survived'] = np.where(df['Survived'] == 1, 'Yes', 'No')



plt.figure(figsize=(10, 10))



ax = sns.barplot(x = "Pass_Class", y = "Number", hue = "Survived", data = df, palette="vlag")



ax.tick_params(axis = 'both', 

               which = 'major', 

               labelsize = 14)



for p in ax.patches:

        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.2, p.get_height()),

                    ha='center', va='bottom',

                    color= 'black', fontsize=14)



plt.xlabel('Passenger Class',

           fontsize = 16)



plt.ylabel('Number\n of\n Passengers', 

           labelpad = 50, 

           rotation = 0, 

           fontsize = 16)



plt.title('List of Survivors and Deaths by Passenger\'s Class' , 

          fontsize = 20)



plt.show()
## Show columns data types

df_train.dtypes
## Column Sex

df_train['Sex'].unique()
## Column Age_Group

df_train['Age_Gr'].unique()
df_train_treat = pd.get_dummies(df_train, 

                                columns=['Sex', 'Age_Gr', 'Port_Embark'], 

                                drop_first = True, 

                                prefix = ['Sex', 'Age_Gr', 'Port_Embark'],

                                prefix_sep='_')



df_train_treat.head(3)
## Show columns data types

df_train_treat.dtypes
def fn_generate_histogram(p_df_dataframe, p_colum, p_num_desc_bar_adjust = 0):

    

    plt.title("Histograma of column [{}]".format(p_colum), 

              fontsize = 16)

    count, bin_edges = np.histogram(p_df_dataframe[p_colum])   

    ax =  p_df_dataframe[p_colum].plot(kind = 'hist', 

                         xticks = bin_edges, 

                         figsize=(10, 10))

    plt.ylabel('Frequency', 

           labelpad = 50, 

           rotation = 0, 

           fontsize = 16)

    

    plt.xlabel('{}'.format(p_colum), 

           fontsize = 16)

    

    for p in ax.patches:

        ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+p_num_desc_bar_adjust, p.get_height()),

                    ha='center', va='bottom',

                    color= 'black', fontsize=14)

    

    

    plt.show()
fn_generate_histogram(df_train_treat, 'Age', 4)
fn_generate_histogram(df_train_treat, 'Pass_Fare', 11)
df_train_treat_corr = df_train_treat.corr()

df_train_treat_corr
## Size of figure

plt.figure(figsize = (30, 18))



## Creating the heatmap

ax = sns.heatmap(df_train_treat_corr, 

                       vmin = -1, 

                       cmap = 'coolwarm',

                       annot = True)



## Configuring some characteristics of the chart, such as axis rotation and font size

ax.set_xticklabels(ax.get_xticklabels(), 

                         rotation = 35)



ax.tick_params(axis = 'both', 

               which = 'major', 

               labelsize = 14)



## Title of Plot

plt.title('Heat Map of Correlation' , 

          fontsize = 26)



## Show Figure

plt.show()
## Dependent Variable

target = df_train_treat["Survived"]



## Independents Variables

expl = df_train_treat.drop(columns = ['Survived','Name', 'Ticket_Id', 'Cabin_Id', 'Deck'], 

                           axis=1)
def fn_split_bases_training_teste(p_df_x_var, p_df_y_var, p_test_size ,p_random_state):

    """

        Description:

            Function that separates a database into two bases, more precisely the training and test bases.

            The bases are separated in a similar proportion in relation to the target variable



        Keyword arguments:

            p_df_x_var -- Object containing only the target variables

            p_df_y_var -- Object containing only the explanatory variables

            p_test_size -- What percentage of data should be assigned to the test base

            p_random_state -- Seed to the random generator



        Return:

            x_train -- Training base that corresponds to the independent variables.

            x_test -- Test base that corresponds to the independent variables.

            y_train -- Training base that corresponds to the dependent / target variable

            y_test -- Test base that corresponds to the independent/ target variable



        Exception:

            None

    """

    y_all = p_df_x_var

    x_all = p_df_y_var

    

    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size = p_test_size ,random_state = p_random_state)

    

    print('Number of observations in the Training base: {} \nNumber of observations in the Test base: {}'.format(len(x_train),len(x_test)))

    

    return  x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test = fn_split_bases_training_teste(target, expl, 0.3, 123)
## Distribution of the target variable in the training and test bases

print("\n Training")

print(y_train.value_counts() / len(y_train))

print("\n Test")

print(y_test.value_counts() / len(y_test))
x_train
def fn_calc_model_accuracy(p_y_train, 

                           p_y_test, 

                           p_y_pred_train, 

                           p_y_score_train, 

                           p_y_pred_test, 

                           p_y_score_test, 

                           model = 'Not Informed',

                           p_first_index = False):

    """

        Description:

            Function that calculates the accuracy of a model



        Keyword arguments:

            p_y_train -- Training base that corresponds to the dependent / target variable

            p_y_test -- Test base that corresponds to the dependent / target variable

            p_y_pred_train -- Object with the predicted value of the target variable, from the training base

            p_y_score_train -- Object with the estimated probabilities of the training base

            p_y_pred_test -- Object with the predicted value of the target variable, from the test base

            p_y_score_test -- Object with the estimated probabilities of the test base

            model(Defaul Value) -- Name of the model used in predictions



        Return:

            acc_train -- Accuracy of training base 

            gini_train -- Gini coefficient from training base 

            roc_auc_train -- Roc Cruve value from training base 

            acc_test --  Accuracy of test base 

            gini_test -- Gini coefficient from test base 

            roc_auc_test -- Roc Cruve value from test base 



        Exception:

            None

    """

    acc_train = round(accuracy_score(p_y_pred_train, p_y_train) * 100, 2)

    

    acc_test = round(accuracy_score(p_y_pred_test, p_y_test) * 100, 2)

    

    y_score_train = p_y_score_train[:, 1] if (p_first_index == False) else p_y_score_train

    y_score_teste = p_y_score_test[:, 1] if (p_first_index == False) else p_y_score_test

    

    fpr_train, tpr_train, thresholds = roc_curve(p_y_train, y_score_train)

    roc_auc_train = 100 * round(auc(fpr_train, tpr_train), 2)

    gini_train = 100 * round((2 * roc_auc_train/ 100 - 1), 2)



    # 

    fpr_test, tpr_test, thresholds = roc_curve(p_y_test, y_score_teste)

    roc_auc_test = 100 * round(auc(fpr_test, tpr_test), 2)

    gini_test = 100 * round((2 * roc_auc_test/100 - 1), 2)

    

    print('Model - ', model)

    print('----Taining Base----\nAccuracy: {} \nGini: {} \nROC Curve: {}'.format(acc_train, \

                                                                                     gini_train, \

                                                                                     roc_auc_train))



    print('\n----Test Base----\nAccuracy: {} \nGini: {} \nROC Curve: {}'.format(acc_test, \

                                                                                      gini_test, \

                                                                                      roc_auc_test))

    

    return acc_train, gini_train, roc_auc_train, acc_test, gini_test, roc_auc_test
def fn_model_gaussian_naive(p_x_train, p_y_train, p_x_test):

    """

        Description:

            Function that executes the gaussian model according to the passed information



        Keyword arguments:

            p_x_train -- Training base that corresponds to the independent variables

            p_y_train -- Training base that corresponds to the dependent / target variable

            p_x_test -- Test base that corresponds to the independent variables

           



        Return:

            y_pred_gaussian_train -- Object with the predicted value of the target variable, from the training base

            y_score_gaussian_train -- Object with the estimated probabilities of the training base 

            y_pred_gaussian_test -- Object with the predicted value of the target variable, from the test base

            y_score_gaussian_test --  Object with the estimated probabilities of the test base

            



        Exception:

            None

    """

    gaussian = GaussianNB()

    gaussian.fit(p_x_train, p_y_train)



    # 

    y_pred_gaussian_train = gaussian.predict(p_x_train)

    y_score_gaussian_train = gaussian.predict_proba(p_x_train)



    # 

    y_pred_gaussian_test = gaussian.predict(p_x_test)

    y_score_gaussian_test = gaussian.predict_proba(p_x_test)

    

    return y_pred_gaussian_train, y_score_gaussian_train, y_pred_gaussian_test, y_score_gaussian_test
def fn_model_logistic_regression(p_x_train, p_y_train, p_x_test):

    """

        Description:

            Function that executes the logistic regression model according to the passed information



        Keyword arguments:

            p_x_train -- Training base that corresponds to the independent variables

            p_y_train -- Training base that corresponds to the dependent / target variable

            p_x_test -- Test base that corresponds to the independent variables

           



        Return:

            y_pred_logreg_train -- Object with the predicted value of the target variable, from the training base

            y_score_logreg_train -- Object with the estimated probabilities of the training base 

            y_pred_logreg_test -- Object with the predicted value of the target variable, from the test base

            y_score_logreg_test --  Object with the estimated probabilities of the test base

            



        Exception:

            None

    """

    logreg = LogisticRegression(solver = 'liblinear')

    logreg.fit(p_x_train, p_y_train)



    # 

    y_pred_logreg_train = logreg.predict(p_x_train)

    y_score_logreg_train = logreg.predict_proba(p_x_train)



    # 

    y_pred_logreg_test = logreg.predict(p_x_test)

    y_score_logreg_test = logreg.predict_proba(p_x_test)

    

    return y_pred_logreg_train, y_score_logreg_train, y_pred_logreg_test, y_score_logreg_test
def fn_model_SVM(p_x_train, p_y_train, p_x_test):

    """

        Description:

            Function that executes the SVM model according to the passed information



        Keyword arguments:

            p_x_train -- Training base that corresponds to the independent variables

            p_y_train -- Training base that corresponds to the dependent / target variable

            p_x_test -- Test base that corresponds to the independent variables

           



        Return:

            y_pred_svc_train -- Object with the predicted value of the target variable, from the training base

            y_score_svc_train -- Object with the estimated probabilities of the training base 

            y_pred_svc_test -- Object with the predicted value of the target variable, from the test base

            y_score_svc_test --  Object with the estimated probabilities of the test base

            



        Exception:

            None

    """

    svc = SVC()



    svc.fit(p_x_train, p_y_train)



    y_pred_svc_train = svc.predict(p_x_train)

    y_score_svc_train = 1/(1+np.exp(-svc.decision_function(p_x_train)))



    y_pred_svc_test = svc.predict(p_x_test)

    y_score_svc_test = 1/(1+np.exp(-svc.decision_function(p_x_test)))

    

    return y_pred_svc_train, y_score_svc_train, y_pred_svc_test, y_score_svc_test   

    
def fn_model_descicion_tree(p_x_train, p_y_train, p_x_test, p_max_depth, p_random_state ):

    """

        Description:

            Function that executes the Decision Tree model according to the passed information



        Keyword arguments:

            p_x_train -- Training base that corresponds to the independent variables

            p_y_train -- Training base that corresponds to the dependent / target variable

            p_x_test -- Test base that corresponds to the independent variables

            p_max_depth -- The maximum depth of the tree

            p_random_state -- Seed to the random generator

           



        Return:

            y_pred_dectree_train -- Object with the predicted value of the target variable, from the training base

            y_score_dectree_train -- Object with the estimated probabilities of the training base 

            y_pred_dectree_test -- Object with the predicted value of the target variable, from the test base

            y_score_dectree_test --  Object with the estimated probabilities of the test base

            



        Exception:

            None

    """

    dectree = DecisionTreeClassifier(criterion = 'entropy',

                                     max_depth = p_max_depth,

                                     random_state = p_random_state)

    

    dectree.fit(p_x_train, p_y_train)





    # Treino

    y_pred_dectree_train = dectree.predict(p_x_train)

    y_score_dectree_train = dectree.predict_proba(p_x_train)[:,1]



    # Teste

    y_pred_dectree_test = dectree.predict(p_x_test)

    y_score_dectree_test = dectree.predict_proba(p_x_test)[:,1]

    

    return y_pred_dectree_train, y_score_dectree_train, y_pred_dectree_test, y_score_dectree_test   

    
def fn_model_random_forest(p_x_train, p_y_train, p_x_test, p_max_depth, p_random_state ):

    """

        Description:

            Function that executes the Random Forest model according to the passed information



        Keyword arguments:

            p_x_train -- Training base that corresponds to the independent variables

            p_y_train -- Training base that corresponds to the dependent / target variable

            p_x_test -- Test base that corresponds to the independent variables

            p_max_depth -- The maximum depth 

            p_random_state -- Seed to the random generator

           



        Return:

            y_pred_rndforest_train -- Object with the predicted value of the target variable, from the training base

            y_score_rndforest_train -- Object with the estimated probabilities of the training base 

            y_pred_rndforest_test -- Object with the predicted value of the target variable, from the test base

            y_score_rndforest_test --  Object with the estimated probabilities of the test base

            



        Exception:

            None

    """

    rndforest = RandomForestClassifier(criterion = 'entropy',

                                       max_depth = p_max_depth,

                                       random_state = p_random_state)

    

    rndforest.fit(p_x_train, p_y_train)



    # Treino

    y_pred_rndforest_train = rndforest.predict(p_x_train)

    y_score_rndforest_train = rndforest.predict_proba(p_x_train)[:,1]



    # Teste

    y_pred_rndforest_test = rndforest.predict(p_x_test)

    y_score_rndforest_test = rndforest.predict_proba(p_x_test)[:,1]

    

    return y_pred_rndforest_train, y_score_rndforest_train, y_pred_rndforest_test, y_score_rndforest_test   

    
def fn_model_gradient_boosting(p_x_train, p_y_train, p_x_test, p_min_samples_leaf ):

    """

        Description:

            Function that executes the Random Forest model according to the passed information



        Keyword arguments:

            p_x_train -- Training base that corresponds to the independent variables

            p_y_train -- Training base that corresponds to the dependent / target variable

            p_x_test -- Test base that corresponds to the independent variables

            p_min_samples_leaf -- Number of samples required to be at a leaf node

           



        Return:

            y_pred_rndforest_train -- Object with the predicted value of the target variable, from the training base

            y_score_rndforest_train -- Object with the estimated probabilities of the training base 

            y_pred_rndforest_test -- Object with the predicted value of the target variable, from the test base

            y_score_rndforest_test --  Object with the estimated probabilities of the test base

            



        Exception:

            None

    """

    



    gbc = GradientBoostingClassifier(min_samples_leaf = p_min_samples_leaf)



    gbc.fit(p_x_train, p_y_train)



    # Treino

    y_pred_gbc_train = gbc.predict(p_x_train)

    y_score_gbc_train = gbc.predict_proba(p_x_train)[:,1]



    # Teste

    y_pred_gbc_test = gbc.predict(p_x_test)

    y_score_gbc_test = gbc.predict_proba(p_x_test)[:,1]

    

    return y_pred_gbc_train, y_score_gbc_train, y_pred_gbc_test, y_score_gbc_test   

    
y_pred_gaussian_train, y_score_gaussian_train, y_pred_gaussian_test, y_score_gaussian_test = fn_model_gaussian_naive(x_train, 

                                                                                                                     y_train, 

                                                                                                                     x_test)
acc_gau_train, gini_gau_train, roc_gau_train, acc_gau_test, gini_gau_test, roc_auc_gau_test = fn_calc_model_accuracy(y_train, 

                                                                                                 y_test, 

                                                                                                 y_pred_gaussian_train,

                                                                                                 y_score_gaussian_train,

                                                                                                 y_pred_gaussian_test,

                                                                                                 y_score_gaussian_test,

                                                                                                 'GAUSSIAN NAIVE BAYES')
columns = ['Pass_Class', 'Age', 'Is_Alone', 'Pass_Fare', 'Sex_male']

x_train_treat = x_train.loc[:,columns]

x_test_treat = x_test.loc[:, columns]



y_pred_gaussian_train, y_score_gaussian_train, y_pred_gaussian_test, y_score_gaussian_test = fn_model_gaussian_naive(x_train_treat, 

                                                                                                                     y_train, 

                                                                                                                     x_test_treat)



acc_gau_train, gini_gau_train, roc_gau_train, acc_gau_test, gini_gau_test, roc_auc_gau_test = fn_calc_model_accuracy(y_train, 

                                                                                                 y_test, 

                                                                                                 y_pred_gaussian_train,

                                                                                                 y_score_gaussian_train,

                                                                                                 y_pred_gaussian_test,

                                                                                                 y_score_gaussian_test,

                                                                                                 'GAUSSIAN NAIVE BAYES')
y_pred_log_train, y_score_log_train, y_pred_log_test, y_score_log_test = fn_model_logistic_regression(x_train,

                                                                                                      y_train, 

                                                                                                      x_test)



acc_log_train, gini_log_train, roc_log_train, acc_log_test, gini_log_test, roc_auc_log_test = fn_calc_model_accuracy(y_train, 

                                                                                                 y_test, 

                                                                                                 y_pred_log_train,

                                                                                                 y_score_log_train,

                                                                                                 y_pred_log_test,

                                                                                                 y_score_log_test,

                                                                                                 'LOGISTIC REGRESSION')
y_pred_svm_train, y_score_svm_train, y_pred_svm_test, y_score_svm_test = fn_model_SVM(x_train,

                                                                                      y_train, 

                                                                                      x_test)



acc_svm_train, gini_svm_train, roc_svm_train, acc_svm_test, gini_svm_test, roc_auc_svm_test = fn_calc_model_accuracy(y_train, 

                                                                                                                     y_test, 

                                                                                                                     y_pred_svm_train,

                                                                                                                     y_score_svm_train,

                                                                                                                     y_pred_svm_test,

                                                                                                                     y_score_svm_test,

                                                                                                                     'SUPPORT VECTOR MACHINE',

                                                                                                                     True)
y_pred_tree_train, y_score_tree_train, y_pred_tree_test, y_score_tree_test = fn_model_descicion_tree(x_train,

                                                                                                     y_train,

                                                                                                     x_test,

                                                                                                     5,

                                                                                                     42)



acc_tree_train, gini_tree_train, roc_tree_train, acc_tree_test, gini_tree_test, roc_auc_tree_test = fn_calc_model_accuracy(y_train, 

                                                                                                                           y_test, 

                                                                                                                           y_pred_tree_train,

                                                                                                                           y_score_tree_train,

                                                                                                                           y_pred_tree_test,

                                                                                                                           y_score_tree_test,

                                                                                                                           'DECISION TREE',

                                                                                                                           True)
y_pred_rndfor_train, y_score_rndfor_train, y_pred_rndfor_test, y_score_rndfor_test = fn_model_random_forest(x_train,

                                                                                                            y_train,

                                                                                                            x_test,

                                                                                                            4,

                                                                                                            42)



acc_rndfor_train, gini_rndfor_train, roc_rndfor_train, acc_rndfor_test, gini_rndfor_test, roc_auc_rndfor_test = fn_calc_model_accuracy(y_train, 

                                                                                                                            y_test, 

                                                                                                                           y_pred_rndfor_train,

                                                                                                                           y_score_rndfor_train,

                                                                                                                           y_pred_rndfor_test,

                                                                                                                           y_score_rndfor_test,

                                                                                                                           'RANDOM FOREST',

                                                                                                                           True)
y_pred_gbc_train, y_score_gbc_train, y_pred_gbc_test, y_score_gbc_test = fn_model_gradient_boosting(x_train,

                                                                                                    y_train,

                                                                                                    x_test,

                                                                                                    6)



acc_gbc_train, gini_gbc_train, roc_gbc_train, acc_gbc_test, gini_gbc_test, roc_auc_gbc_test = fn_calc_model_accuracy(y_train, 

                                                                                                                            y_test, 

                                                                                                                           y_pred_gbc_train,

                                                                                                                           y_score_gbc_train,

                                                                                                                           y_pred_gbc_test,

                                                                                                                           y_score_gbc_test,

                                                                                                                           'GRADIENT BOOSTING',

                                                                                                                           True)




models = pd.DataFrame({

    'Model': ['Decision Tree', 

               'Random Forest', 

               'Gradient Boosting',

               'Support Vector Machine(SVM)',

               'Logistic Regression',

               'Gaussian Naive Bayes'],

    

     'Train Accuracy': [acc_tree_train,

                        acc_rndfor_train,

                        acc_gbc_train,

                        acc_svm_train,

                        acc_log_train, 

                        acc_gau_train],   

    

    

    'Test Accuracy': [acc_tree_test,

                      acc_rndfor_test,

                      acc_gbc_test,

                      acc_svm_test,

                      acc_log_test,

                      acc_gau_test]



})

model_comp = models.sort_values(by = 'Test Accuracy', 

                                ascending = False)

model_comp = model_comp[['Model','Train Accuracy','Test Accuracy']]

model_comp



x_train = x_train.drop(columns = ['Port_Embark_Not Info'], 

                           axis=1)



df_test['Is_Alone'] = np.where((df_test['Sibli_Aboard'] > 0) | 

                                (df_test['Par_Child_Aboard'] > 0), 0, 1)



df_test_treat = pd.get_dummies(df_test, 

                                columns=['Sex', 'Age_Gr', 'Port_Embark'], 

                                drop_first = True, 

                                prefix = ['Sex', 'Age_Gr', 'Port_Embark'],

                                prefix_sep='_')



df_test_treat = df_test_treat.drop(columns = ['Name', 'Ticket_Id', 'Cabin_Id', 'Deck'], 

                           axis=1)



df_test_treat.head(3)
y_pred_gbc_train, y_score_gbc_train, y_pred_gbc_test, y_score_gbc_test = fn_model_gradient_boosting(x_train,

                                                                                                    y_train,

                                                                                                    df_test_treat,

                                                                                                    6)
df_test_treat = df_test_treat.reset_index()



submission = pd.DataFrame({

        "PassengerId": df_test_treat["PassengerId"],

        "Survived": y_pred_gbc_test

    })



#submission.to_csv('submission.csv', index = False)