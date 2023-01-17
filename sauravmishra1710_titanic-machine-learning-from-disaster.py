# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import the required libraries



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt 

import seaborn as sns



from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import recall_score, accuracy_score, confusion_matrix, f1_score

from sklearn.metrics import precision_score, auc, roc_auc_score, roc_curve, precision_recall_curve



from sklearn.ensemble import AdaBoostClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier



# Ignore Warnings

import warnings

warnings.filterwarnings('ignore')



# To display all the columns

pd.options.display.max_columns = None



# To display all the rows

pd.options.display.max_rows = None



# To map Empty Strings or numpy.inf as Na Values

pd.options.mode.use_inf_as_na = True



pd.options.display.expand_frame_repr =  False



%matplotlib inline



# Set Style

sns.set(style = "whitegrid")
# train data

titanic_data = pd.read_csv('/kaggle/input/titanic/train.csv', low_memory = False, skipinitialspace = True, float_precision = 2)



# data glimpse

titanic_data.head()
# test data

titanic_test_data = pd.read_csv('/kaggle/input/titanic/test.csv', low_memory = False, skipinitialspace = True, float_precision = 2)



# data glimpse

titanic_test_data.head()
print("Train Data Shape:", titanic_data.shape)

print("Test Data Shape:", titanic_test_data.shape)
# train set columns

print("Columns in the train set :")

print(titanic_data.columns)



print('\n')



# test set columns

print("Columns in the test set :")

print(titanic_test_data.columns)
# the columns 'PassengerId' and 'Ticket' willl be of no value in the analysis.

# Lets drop those columns from both the train ans test set.



# train set

titanic_data.drop(columns = ['PassengerId', 'Ticket'], axis = 1, inplace = True)



# test set

titanic_test_passengerId = titanic_test_data[['PassengerId']]

titanic_test_data.drop(columns = ['PassengerId', 'Ticket'], axis = 1, inplace = True)



# data glimpse

titanic_data.head()
# Check the total missing values in each column.

print("Total NULL Values in each columns")

print("*********************************")

print(titanic_data.isnull().sum())
# Lets check the percentage of missing values column-wise in the train data



(titanic_data.isnull().sum()/ len(titanic_data)) * 100
# Lets check the percentage of missing values column-wise in the test data



(titanic_test_data.isnull().sum()/ len(titanic_test_data)) * 100
# With more than 75% of values missing, we can drop the column - 'Cabin' but before doing that let us have a look 

# at it if we can get something out of almost nothing - 



titanic_data.Cabin.value_counts()
titanic_test_data.Cabin.value_counts()
# Observations from the above cell - 

# 1. We see above that there are about 147 unique values and some rows have multiple cabins allocated.



# 2. Each cabin has 2 parts. Digging into the cabin numbers in the titanic, wikipedia shows the first letter

#    in the cabin is the Deck and the number is the room number.



# We will extract the deck information from the cabin



titanic_data["deck"] = titanic_data["Cabin"].str.slice(0,1)



titanic_test_data["deck"] = titanic_test_data["Cabin"].str.slice(0,1)



# data glimpse

titanic_data.head()
# Lets now check the value counts for the newly created column - 'deck'



print('Deck Value Counts for train set - ')

print(titanic_data.deck.value_counts())



print('\n')



print('Deck Value Counts for test set - ')

print(titanic_test_data.deck.value_counts())
# Lets assign a default deck - say 'Z' for both the train and the test sets -



# train set

titanic_data["deck"] = titanic_data["deck"].fillna("Z")



# also replace the single T-deck with G

titanic_data['deck'].replace(['T'], ['G'], inplace=True)



# test set

titanic_test_data["deck"] = titanic_test_data["deck"].fillna("Z")



# data glimpse

titanic_data.head()
# Now we have the deck information, We can drop the cabin column



# train set

titanic_data.drop(columns = ['Cabin'], axis = 1, inplace = True)



# test set

titanic_test_data.drop(columns = ['Cabin'], axis = 1, inplace = True)



# data glimpse

titanic_data.head()
# Let's have a look at the age values wrt -

# 1. gender

# 2. salutation (in the name column every person has a salutaion prefix. This can be a factor in

# imputing the missing age values)



# lets check the gender wise age distribution in the train set



print('Train Set')

print('Age of the oldest Passenger was:', titanic_data['Age'].max(),'Years')

print('Age of the youngest Passenger was:', titanic_data['Age'].min(),'Years')

print('Average Age on the ship:', titanic_data['Age'].mean(),'Years\n')



print(titanic_data.groupby(by = ['Sex']).Age.describe())



print('\n-----------------------------------------------------------------------------\n')

print('Test Set')

# lets check the gender wise age distribution in the test set



print('Age of the oldest Passenger was:', titanic_test_data['Age'].max(),'Years')

print('Age of the youngest Passenger was:', titanic_test_data['Age'].min(),'Years')

print('Average Age on the ship:', titanic_test_data['Age'].mean(),'Years\n')



print(titanic_test_data.groupby(by = ['Sex']).Age.describe())

# 2. salutation (in the name column every person has a salutaion prefix. This can be a factor in

# imputing the missing age values)



# We will extract the salution from every person's name in the Name column.



salutation = titanic_data.Name.str.split(', ').str[1].str.split('.').str[0]

titanic_data['salutation'] = salutation

titanic_data.salutation.value_counts()
# applying the above in the test set to extract the salutation there -



salutation = titanic_test_data.Name.str.split(', ').str[1].str.split('.').str[0]

titanic_test_data['salutation'] = salutation

titanic_test_data.salutation.value_counts()
# Salutations inputations in train data



titanic_data['salutation'].replace(['Mme','Lady','the Countess','Jonkheer'], ['Mrs','Mrs', 'Mrs','Mrs'], inplace=True)



titanic_data['salutation'].replace(['Mlle', 'Jonkheer', 'Ms' ], ['Miss','Miss', 'Miss'], inplace=True)



titanic_data['salutation'].replace(['Sir'], ['Mr'], inplace=True)



titanic_data['salutation'].replace(['Major', 'Col', 'Capt'], ['army_rank', 'army_rank', 'army_rank'], inplace=True)
# age distribution with respect to the name salutations



print(titanic_data.groupby(by = ['salutation']).Age.describe())
# Salutations inputations in test data



titanic_test_data['salutation'].replace(['Mme','Lady','the Countess','Jonkheer'], ['Mrs','Mrs', 'Mrs','Mrs'], inplace=True)



titanic_test_data['salutation'].replace(['Mlle', 'Jonkheer', 'Ms' ], ['Miss','Miss', 'Miss'], inplace=True)



titanic_test_data['salutation'].replace(['Sir'], ['Mr'], inplace=True)



titanic_test_data['salutation'].replace(['Major', 'Col', 'Capt'], ['army_rank', 'army_rank', 'army_rank'], inplace=True)



titanic_test_data['salutation'].replace(['Dona'], ['Don'], inplace=True)
# age distribution with respect to the name salutations



print(titanic_test_data.groupby(by = ['salutation']).Age.describe())
# Missing age value for each category - 



print('Number of presons with salutation type Mr having missing age value - ', 

      titanic_data.loc[(titanic_data['Age'].isnull()) & (titanic_data['salutation'] == 'Mr')].shape[0])



print('Number of presons with salutation type Capt having missing age value - ', 

      titanic_data.loc[(titanic_data['Age'].isnull()) & (titanic_data['salutation'] == 'Capt')].shape[0])



print('Number of presons with salutation type Col having missing age value - ', 

      titanic_data.loc[(titanic_data['Age'].isnull()) & (titanic_data['salutation'] == 'Col')].shape[0])



print('Number of presons with salutation type Don having missing age value - ', 

      titanic_data.loc[(titanic_data['Age'].isnull()) & (titanic_data['salutation'] == 'Don')].shape[0])



print('Number of presons with salutation type Dr having missing age value - ', 

      titanic_data.loc[(titanic_data['Age'].isnull()) & (titanic_data['salutation'] == 'Dr')].shape[0])



print('Number of presons with salutation type Major having missing age value - ', 

      titanic_data.loc[(titanic_data['Age'].isnull()) & (titanic_data['salutation'] == 'Major')].shape[0])



print('Number of presons with salutation type Master having missing age value - ', 

      titanic_data.loc[(titanic_data['Age'].isnull()) & (titanic_data['salutation'] == 'Master')].shape[0])



print('Number of presons with salutation type Miss having missing age value - ', 

      titanic_data.loc[(titanic_data['Age'].isnull()) & (titanic_data['salutation'] == 'Miss')].shape[0])



print('Number of presons with salutation type Mrs having missing age value - ', 

      titanic_data.loc[(titanic_data['Age'].isnull()) & (titanic_data['salutation'] == 'Mrs')].shape[0])



print('Number of presons with salutation type Rev having missing age value - ', 

      titanic_data.loc[(titanic_data['Age'].isnull()) & (titanic_data['salutation'] == 'Rev')].shape[0])
# the values (mean & median) are taken from the - 'age distribution with respect to the name salutations' cell above.



titanic_data.loc[(titanic_data['Age'].isnull()) & (titanic_data['salutation'] == 'Mr'),'Age'] = 30

titanic_data.loc[(titanic_data['Age'].isnull()) & (titanic_data['salutation'] == 'Dr'),'Age'] = 42

titanic_data.loc[(titanic_data['Age'].isnull()) & (titanic_data['salutation'] == 'Master'),'Age'] = 3.5

titanic_data.loc[(titanic_data['Age'].isnull()) & (titanic_data['salutation'] == 'Miss'),'Age'] = 21

titanic_data.loc[(titanic_data['Age'].isnull()) & (titanic_data['salutation'] == 'Mrs'),'Age'] = 35
# Repeating the above for the test set as well -



# Missing age value for each category - 



print('Number of presons with salutation type Mr having missing age value - ', 

      titanic_test_data.loc[(titanic_test_data['Age'].isnull()) & (titanic_test_data['salutation'] == 'Mr')].shape[0])



print('Number of presons with salutation type Capt having missing age value - ', 

      titanic_test_data.loc[(titanic_test_data['Age'].isnull()) & (titanic_test_data['salutation'] == 'Capt')].shape[0])



print('Number of presons with salutation type Col having missing age value - ', 

      titanic_test_data.loc[(titanic_test_data['Age'].isnull()) & (titanic_test_data['salutation'] == 'Col')].shape[0])



print('Number of presons with salutation type Don having missing age value - ', 

      titanic_test_data.loc[(titanic_test_data['Age'].isnull()) & (titanic_test_data['salutation'] == 'Don')].shape[0])



print('Number of presons with salutation type Dr having missing age value - ', 

      titanic_test_data.loc[(titanic_test_data['Age'].isnull()) & (titanic_test_data['salutation'] == 'Dr')].shape[0])



print('Number of presons with salutation type Major having missing age value - ', 

      titanic_test_data.loc[(titanic_test_data['Age'].isnull()) & (titanic_test_data['salutation'] == 'Major')].shape[0])



print('Number of presons with salutation type Master having missing age value - ', 

      titanic_test_data.loc[(titanic_test_data['Age'].isnull()) & (titanic_test_data['salutation'] == 'Master')].shape[0])



print('Number of presons with salutation type Miss having missing age value - ', 

      titanic_test_data.loc[(titanic_test_data['Age'].isnull()) & (titanic_test_data['salutation'] == 'Miss')].shape[0])



print('Number of presons with salutation type Mrs having missing age value - ', 

      titanic_test_data.loc[(titanic_test_data['Age'].isnull()) & (titanic_test_data['salutation'] == 'Mrs')].shape[0])



print('Number of presons with salutation type Rev having missing age value - ', 

      titanic_test_data.loc[(titanic_test_data['Age'].isnull()) & (titanic_test_data['salutation'] == 'Rev')].shape[0])
# the values (mean & median) are taken from the - 'age distribution with respect to the name salutations' cell above.



titanic_test_data.loc[(titanic_test_data['Age'].isnull()) & (titanic_test_data['salutation'] == 'Mr'),'Age'] = 28.5

titanic_test_data.loc[(titanic_test_data['Age'].isnull()) & (titanic_test_data['salutation'] == 'Master'),'Age'] = 7

titanic_test_data.loc[(titanic_test_data['Age'].isnull()) & (titanic_test_data['salutation'] == 'Miss'),'Age'] = 21.77

titanic_test_data.loc[(titanic_test_data['Age'].isnull()) & (titanic_test_data['salutation'] == 'Mrs'),'Age'] = 36.5
# Lets have a look at the percentage of null values in the data set now - 



(titanic_data.isnull().sum()/ len(titanic_data)) * 100
# Lets have a look at the percentage of null values in the data set now - 



(titanic_test_data.isnull().sum()/ len(titanic_test_data)) * 100
# Now lets check on the missing values for the 'Embarked' column -



titanic_data.loc[(titanic_data['Embarked'].isnull())]
titanic_data.loc[(titanic_data['Fare'] >= 79) & (titanic_data['Fare'] <= 82)]
titanic_data.loc[(titanic_data['Embarked'].isnull()) & (titanic_data['Age'] == 38),'Embarked'] = 'S'

titanic_data.loc[(titanic_data['Embarked'].isnull()) & (titanic_data['Age'] == 62),'Embarked'] = 'C'
# Lets have a look at the percentage of null values in the train set now - 



(titanic_data.isnull().sum()/ len(titanic_data)) * 100
# the test set has some missing values in the fare column. lets examine that



titanic_test_data.Fare.describe()
# as mean > median value for Fare in test set, lets impute the missing value with median - 



# test set

titanic_test_data["Fare"] = titanic_test_data["Fare"].fillna(14.45)
# Lets have a look at the percentage of null values in the test set now - 



(titanic_test_data.isnull().sum()/ len(titanic_test_data)) * 100
# dropping the name column -



titanic_data.drop(['Name'], axis=1, inplace = True)

titanic_test_data.drop(['Name'], axis=1, inplace = True)
# data glimpse



titanic_data.head()
# data glimpse



titanic_test_data.head()
# Custom Function for Default Plotting variables



# Function Parameters  - 



# figure_title         -    The title to use for the plot.

# xlabel               -    The x-axis label for the plot.

# ylabel               -    The y-axis label for the plot.

# xlabel_rotation      -    The degree of rotation for the x-axis ticks (values).

# legend_flag          -    Boolean flag to check if a legend is required to be their in the plot.

# legend               -    Place legend on axis subplots.



def set_plotting_variable(figure_title, xlabel, ylabel, xlabel_rotation, legend_flag, legend):

    

    plt.title(figure_title)

    plt.xticks(rotation = xlabel_rotation)

    plt.xlabel(xlabel, labelpad = 15)

    plt.ylabel(ylabel, labelpad = 10)

    

    if legend_flag == True:

        plt.legend(loc = legend)
# Function Parameters   -



# category              -      The category of the variable in consideration - Categorical or Continuous.

# plot_type             -      The type of the plot - Unordered Categorical (-lineplot) or Ordered Categorical (-countplot).

# series                -      The series/column from the data frame for which the univariate analysis is being considered for.

# figsize_x             -      The width of the plot figure in inches.

# figsize_y             -      The height of the plot figure in inches.

# subplot_x             -      The rows for the subplot.

# subplot_y             -      The columns for the subplot.

# xlabel                -      The x-axis label for the plot.

# ylabel                -      The y-axis label for the plot.

# x_axis                -      The series/variable to be plotted along the x-axis.

# hue                   -      The variable (categorical) in the data for which the plot is being considered for.

# data                  -      The data frame.

# legend                -      Place legend on axis subplots.



# hspace                -      The amount of height reserved for space between subplots,

#                              expressed as a fraction of the average axis height



# wspace                -      The amount of width reserved for space between subplots,

#                              expressed as a fraction of the average axis width



# xlabel_rotation       -      The degree of rotation for the x-axis ticks (values).



def plot_univariate(category, plot_type, series, figsize_x, figsize_y, subplot_x, subplot_y,

                    xlabel, ylabel, x_axis, hue, data, legend, hspace, wspace, xlabel_rotation):

    

    plt.figure(figsize = (figsize_x, figsize_y))

    

    if category == 'Categorical':

        

        title_1 = "Frequency Plot of " + xlabel

        title_2 = title_1 + " across Survival Status"

        

        # Subplot - 1

        plt.subplot(subplot_x, subplot_y, 1)

        

        if plot_type == 'Unordered Categorical':

            sns.lineplot(data = series)

        

        elif plot_type == 'Ordered Categorical':

            sns.countplot(x = x_axis, order = series.sort_index().index, data = data)

        

        # Call Custom Function

        set_plotting_variable(title_1, xlabel, ylabel, xlabel_rotation, False, legend)

        

        # Subplot - 2

        plt.subplot(subplot_x, subplot_y, 2)

        

        sns.countplot(x = x_axis, hue = hue, order = series.sort_index().index, data = data)

        # Call Custom Function

        set_plotting_variable(title_2, xlabel, ylabel, xlabel_rotation, True, legend)

    

    elif category == 'Continuous':

        

        title_1 = "Distribution Plot of " + xlabel

        title_2 = "Box Plot of " + xlabel

        title_3 = title_2 + " across Survival Status"

        

        # Subplot - 1

        plt.subplot(subplot_x, subplot_y, 1)

        

        sns.distplot(data[x_axis], hist = True, kde = True, color = 'g')

        # Call Custom Function

        set_plotting_variable(title_1, xlabel, ylabel, xlabel_rotation, False, legend)

        

        # Subplot - 2

        plt.subplot(subplot_x, subplot_y, 2)

        

        sns.boxplot(x = x_axis, data = data, color = 'm')

        # Call Custom Function

        set_plotting_variable(title_2, xlabel, ylabel, xlabel_rotation, False, legend)

           

    plt.subplots_adjust(hspace = hspace)

    plt.subplots_adjust(wspace = wspace)

    plt.show()
series = titanic_data.Survived.value_counts(dropna = False)



print(series.sort_index())

print('\n')



plt.figure(figsize = (8, 6))



plt.title('Frequency Plot of Persons Survived')

sns.countplot(x = 'Survived',  

              order = series.sort_index().index, 

              data = titanic_data)

plt.xlabel('Survived Status', labelpad = 15)

plt.ylabel('Frequency', labelpad = 10)



plt.subplots_adjust(wspace = 0.6)

plt.show()



# Survived - 1 means the person survived and 0 means the person did not survive the Titanic mishap.
# Rank-Frequency Plot of Unordered Categorical Variable: Embarked



print('Embarked - ' + 'Port of Embarkation' + '\n' 

      +'-------------------------------' + '\n' + 'C = Cherbourg \nQ = Queenstown \nS = Southampton'  + '\n'

      +'-------------------------------')



series = titanic_data.Embarked.value_counts(dropna = False)



print(series.sort_index())

print('\n')

print(titanic_data.groupby(by = 'Survived').Embarked.value_counts(dropna = False).sort_index())

print('\n')



# Call Custom Function

plot_univariate(category = 'Categorical',

                plot_type = 'Unordered Categorical',

                series = series,

                figsize_x = 15,

                figsize_y = 6,

                subplot_x = 1,

                subplot_y = 2,

                xlabel = "Port of Embarkation",

                ylabel = "Frequency",

                x_axis = 'Embarked',

                hue = 'Survived',

                data = titanic_data,

                legend = 'upper center',

                hspace = 0,

                wspace = 0.3,

                xlabel_rotation = 0)
series = titanic_data.salutation.value_counts(dropna = False)



print(series.sort_index())

print('\n')

print(titanic_data.groupby(by = 'Survived').salutation.value_counts(dropna = False).sort_index())

print('\n')



# Call Custom Function

plot_univariate(category = 'Categorical',

                plot_type = 'Unordered Categorical',

                series = series,

                figsize_x = 15,

                figsize_y = 6,

                subplot_x = 1,

                subplot_y = 2,

                xlabel = "Salutations",

                ylabel = "Frequency",

                x_axis = 'salutation',

                hue = 'Survived',

                data = titanic_data,

                legend = 'upper center',

                hspace = 0,

                wspace = 0.3,

                xlabel_rotation = 0)
print('Pclass - ' + 'Passenger Class' + '\n' 

      +'-------------------------------' + '\n' + 'Ticket class \n1 = 1st \n2 = 2nd \n3 = 3rd' + '\n'

      +'-------------------------------')



series = titanic_data.Pclass.value_counts(dropna = False)



print(series.sort_index())

print('\n')

print(titanic_data.groupby(by = 'Survived').Pclass.value_counts(dropna = False).sort_index())

print('\n')



# Call Custom Function

plot_univariate(category = 'Categorical',

                plot_type = 'Unordered Categorical',

                series = series,

                figsize_x = 15,

                figsize_y = 6,

                subplot_x = 1,

                subplot_y = 2,

                xlabel = "Passenger Class",

                ylabel = "Frequency",

                x_axis = 'Pclass',

                hue = 'Survived',

                data = titanic_data,

                legend = 'upper center',

                hspace = 0,

                wspace = 0.3,

                xlabel_rotation = 0)
series = titanic_data.Sex.value_counts(dropna = False)



print(series.sort_index())

print('\n')

print(titanic_data.groupby(by = 'Survived').Sex.value_counts(dropna = False).sort_index())

print('\n')



# Call Custom Function

plot_univariate(category = 'Categorical',

                plot_type = 'Unordered Categorical',

                series = series,

                figsize_x = 15,

                figsize_y = 6,

                subplot_x = 1,

                subplot_y = 2,

                xlabel = "Gender",

                ylabel = "Frequency",

                x_axis = 'Sex',

                hue = 'Survived',

                data = titanic_data,

                legend = 'upper center',

                hspace = 0,

                wspace = 0.3,

                xlabel_rotation = 0)
abc = titanic_data.groupby(['Sex', 'Survived']).Pclass.value_counts(dropna = False).sort_index()

print(abc)



abc.unstack().plot()
print(titanic_data.Fare.describe())

print('\n')

print(titanic_data.groupby(by = 'Survived').Fare.describe().sort_index())

print('\n')



# Call Custom Function

plot_univariate(category = 'Continuous',

                plot_type = 'Quantitative',

                series = [1, 0],

                figsize_x = 15,

                figsize_y = 12,

                subplot_x = 2,

                subplot_y = 2,

                xlabel = "Fare",

                ylabel = "Distribution",

                x_axis = 'Fare',

                hue = 'Survived',

                data = titanic_data,

                legend = 'best',

                hspace = 0.4,

                wspace = 0.3,

                xlabel_rotation = 0)
f,sub_plt = plt.subplots(1,2,figsize=(20,10))

x_bins = list(range(0,85,5))



# Survived Status 0

titanic_data[titanic_data['Survived'] == 0].Age.plot.hist(ax=sub_plt[0], bins=20)

sub_plt[0].set_title('Survived = 0')

sub_plt[0].set_xticks(x_bins)



# Survived Status 1

titanic_data[titanic_data['Survived'] == 1].Age.plot.hist(ax=sub_plt[1], bins=20)

sub_plt[1].set_title('Survived = 1')

sub_plt[1].set_xticks(x_bins)



plt.show()
print(titanic_data.groupby(by = 'Survived').SibSp.value_counts(dropna = False).sort_index())



series = titanic_data.SibSp.value_counts(dropna = False)



plot_univariate(category = 'Categorical',

                plot_type = 'Unordered Categorical',

                series = series,

                figsize_x = 15,

                figsize_y = 6,

                subplot_x = 1,

                subplot_y = 2,

                xlabel = "SibSp",

                ylabel = "Frequency",

                x_axis = 'SibSp',

                hue = 'Survived',

                data = titanic_data,

                legend = 'upper center',

                hspace = 0,

                wspace = 0.3,

                xlabel_rotation = 0)
print(titanic_data.groupby(by = 'Survived').Parch.value_counts(dropna = False).sort_index())



series = titanic_data.Parch.value_counts(dropna = False)



plot_univariate(category = 'Categorical',

                plot_type = 'Unordered Categorical',

                series = series,

                figsize_x = 15,

                figsize_y = 6,

                subplot_x = 1,

                subplot_y = 2,

                xlabel = "Parch",

                ylabel = "Frequency",

                x_axis = 'Parch',

                hue = 'Survived',

                data = titanic_data,

                legend = 'upper center',

                hspace = 0,

                wspace = 0.3,

                xlabel_rotation = 0)
# creation of the new feature - familysize

titanic_data['familysize'] = titanic_data['SibSp'] + titanic_data['Parch']



titanic_test_data['familysize'] = titanic_test_data['SibSp'] + titanic_test_data['Parch']



# lets now drop thae columns 'SibSp' and 'Parch' as they would be correlated and predict the 'familysize'



# train set

titanic_data.drop(columns = ['SibSp', 'Parch'], axis = 1, inplace = True)



# test set

titanic_test_data.drop(columns = ['SibSp', 'Parch'], axis = 1, inplace = True)



# data glimpse

titanic_data.head()
titanic_data['familysize'].describe()
print(titanic_data.groupby(by = 'Survived').familysize.value_counts(dropna = False).sort_index())



series = titanic_data.familysize.value_counts(dropna = False)



plot_univariate(category = 'Categorical',

                plot_type = 'Unordered Categorical',

                series = series,

                figsize_x = 15,

                figsize_y = 6,

                subplot_x = 1,

                subplot_y = 2,

                xlabel = "familysize",

                ylabel = "Frequency",

                x_axis = 'familysize',

                hue = 'Survived',

                data = titanic_data,

                legend = 'upper center',

                hspace = 0,

                wspace = 0.3,

                xlabel_rotation = 0)
# train set



titanic_data['age_category'] = 0

titanic_data.loc[titanic_data['Age'] <= 15,'age_category'] = 0

titanic_data.loc[(titanic_data['Age'] > 15) & (titanic_data['Age'] <= 30),'age_category'] = 1

titanic_data.loc[(titanic_data['Age'] > 30) & (titanic_data['Age'] <= 45),'age_category'] = 2

titanic_data.loc[(titanic_data['Age'] > 45) & (titanic_data['Age'] <= 60),'age_category'] = 3

titanic_data.loc[(titanic_data['Age'] > 60),'age_category'] = 4



# dropping the 'Age' feature

titanic_data.drop(['Age'], axis = 1, inplace = True)



# data glimpse

titanic_data.head()
# test set



titanic_test_data['age_category'] = 0

titanic_test_data.loc[titanic_test_data['Age'] <= 15,'age_category'] = 0

titanic_test_data.loc[(titanic_test_data['Age'] > 15) & (titanic_test_data['Age'] <= 30),'age_category'] = 1

titanic_test_data.loc[(titanic_test_data['Age'] > 30) & (titanic_test_data['Age'] <= 45),'age_category'] = 2

titanic_test_data.loc[(titanic_test_data['Age'] > 45) & (titanic_test_data['Age'] <= 60),'age_category'] = 3

titanic_test_data.loc[(titanic_test_data['Age'] > 60),'age_category'] = 4



# dropping the 'Age' feature

titanic_test_data.drop(['Age'], axis = 1, inplace = True)



# data glimpse

titanic_test_data.head()
titanic_data.Fare.describe(percentiles = [.20, .40, .60, .80])
# train set



titanic_data['fare_category'] = 0

titanic_data.loc[titanic_data['Fare'] <= 7.85,'fare_category'] = 0

titanic_data.loc[(titanic_data['Fare'] > 7.85) & (titanic_data['Fare'] <= 10.50),'fare_category'] = 1

titanic_data.loc[(titanic_data['Fare'] > 10.50) & (titanic_data['Fare'] <= 21.67),'fare_category'] = 2

titanic_data.loc[(titanic_data['Fare'] > 21.67) & (titanic_data['Fare'] <= 39.68),'fare_category'] = 3

titanic_data.loc[(titanic_data['Fare'] > 39.68),'fare_category'] = 4



# dropping the 'Age' feature

titanic_data.drop(['Fare'], axis = 1, inplace = True)



# data glimpse

titanic_data.head()
# test set



titanic_test_data['fare_category'] = 0

titanic_test_data.loc[titanic_test_data['Fare'] <= 7.85,'fare_category'] = 0

titanic_test_data.loc[(titanic_test_data['Fare'] > 7.85) & (titanic_test_data['Fare'] <= 10.50),'fare_category'] = 1

titanic_test_data.loc[(titanic_test_data['Fare'] > 10.50) & (titanic_test_data['Fare'] <= 21.67),'fare_category'] = 2

titanic_test_data.loc[(titanic_test_data['Fare'] > 21.67) & (titanic_test_data['Fare'] <= 39.68),'fare_category'] = 3

titanic_test_data.loc[(titanic_test_data['Fare'] > 39.68),'fare_category'] = 4



# dropping the 'Age' feature

titanic_test_data.drop(['Fare'], axis = 1, inplace = True)



# data glimpse

titanic_test_data.head()
titanic_data['Sex'] = titanic_data.Sex.map({'female':0, 'male':1})



# data glimpse

titanic_data.head()
titanic_test_data['Sex'] = titanic_test_data.Sex.map({'female':0, 'male':1})



# data glimpse

titanic_test_data.head()
# train data



# Use get_dummies

dummies_for_decks = pd.get_dummies(titanic_data['deck'], prefix = 'deck', drop_first = False)



# data glimpse

dummies_for_decks.head()
# lets drop the 'deck_T'column

dummies_for_decks.drop(['deck_G'], axis = 1, inplace = True)



# Merging to the master data frame 

titanic_data = titanic_data.join(dummies_for_decks)



# dropping the 'deck' column from the master frame

titanic_data.drop(['deck'], axis = 1, inplace = True)



# data glimpse

titanic_data.head()
# test data



# Use get_dummies

dummies_for_decks = pd.get_dummies(titanic_test_data['deck'], prefix = 'deck', drop_first = False)



# data glimpse

dummies_for_decks.head()
# lets drop the 'deck_T'column

dummies_for_decks.drop(['deck_G'], axis = 1, inplace = True)



# Merging to the master data frame 

titanic_test_data = titanic_test_data.join(dummies_for_decks)



# dropping the 'deck' column from the master frame

titanic_test_data.drop(['deck'], axis = 1, inplace = True)



# data glimpse

titanic_test_data.head()
# train data



titanic_data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)



#data glimpse

titanic_data.head()
# test data



titanic_test_data['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)



#data glimpse

titanic_test_data.head()
# train data



# we have a total of 8 different salutation present in the data as seen in the above cell

# let's encode these to a number range 0 to 7 - 

titanic_data['salutation'].replace(['Mr', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev', 'army_rank', 'Don' ],

                                   [0, 1, 2, 3, 4, 5, 6, 7], inplace=True)



#data glimpse

titanic_data.head()
# test data



# we have a total of 8 different salutation present in the data as seen in the above cell

# let's encode these to a number range 0 to 7 - 

titanic_test_data['salutation'].replace(['Mr', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev', 'army_rank', 'Don' ],

                                   [0, 1, 2, 3, 4, 5, 6, 7], inplace=True)



#data glimpse

titanic_test_data.head()
# Custom Function to get Scores and plots

def get_scores(scores, reg, X_test):

    

    # Plot ROC and PR curves using all models and test data

    fig, axes = plt.subplots(1, 2, figsize = (14, 6))



    pred_test = reg.predict(X_test.values)



    pred_test_probs = reg.predict_proba(X_test.values)[:, 1:]



    fpr, tpr, thresholds = roc_curve(y_test.values.ravel(), pred_test)

    p, r, t = precision_recall_curve(y_test.values.ravel(), pred_test_probs)



    model_f1_score = f1_score(y_test.values.ravel(), pred_test)

    model_precision_score = precision_score(y_test.values.ravel(), pred_test)

    model_recall_score = recall_score(y_test.values.ravel(), pred_test)

    model_accuracy_score = accuracy_score(y_test.values.ravel(), pred_test)

    model_auc_roc = auc(fpr, tpr)

    model_auc_pr = auc(p, r, reorder = True)



    scores.append((model_f1_score,

                   model_precision_score,

                   model_recall_score,

                   model_accuracy_score,

                   model_auc_roc,

                   model_auc_pr,

                   confusion_matrix(y_test.values.ravel(), pred_test)))



    axes[0].plot(fpr, tpr, label = f"auc_roc = {model_auc_roc:.3f}")

    axes[1].plot(r, p, label = f"auc_pr = {model_auc_pr:.3f}")



    axes[0].plot([0, 1], [0, 1], 'k--')

    axes[0].legend(loc = "lower right")

    axes[0].set_xlabel("False Positive Rate")

    axes[0].set_ylabel("True Positive Rate")

    axes[0].set_title("AUC ROC curve")



    axes[1].legend(loc = "lower right")

    axes[1].set_xlabel("recall")

    axes[1].set_ylabel("precision")

    axes[1].set_title("PR curve")



    plt.tight_layout()

    plt.show()

    

    return scores
# Custom Function for hyper parameter tuning



def tune_hyper_parameter(X, y, param_grid, model_type, ml = 'None'):

   

    gc = GridSearchCV(estimator = ml, param_grid = param_grid, scoring = 'roc_auc',

                          n_jobs = 15, cv = 5, verbose = 2, return_train_score=True)

    

    gc = gc.fit(X.values, y.values.ravel())



    return gc
# Custom Function to plot GridSearch Result to get the best value



def hypertuning_plot(scores, parameter):

    

    col = "param_" + parameter

    

    plt.figure()

    

    plt.plot(scores[col], scores["mean_train_score"], label = "training accuracy")

    plt.plot(scores[col], scores["mean_test_score"], label = "test accuracy")

    

    plt.xlabel(parameter)

    plt.ylabel("Accuracy")

    

    plt.legend()

    plt.show()
X = titanic_data.drop('Survived', axis = 1)

y = titanic_data[['Survived']]



X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.85, test_size = 0.15, random_state = 100)



print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
# XGBoost with Default Parameters



xgb = XGBClassifier(n_jobs = -1, random_state = 100)



xgb = xgb.fit(X_train.values, y_train.values.ravel())



# Get the Score Metrics and plots

scores = []



scores = get_scores(scores, xgb, X_test)



# Tabulate results

sampling_results = pd.DataFrame(scores, columns = ['f1', 'precision', 'recall', 'accuracy',

                                                   'auc_roc', 'auc_pr', 'confusion_matrix'])

sampling_results
# GridSearchCV to find optimal max_depth



xgb = XGBClassifier(n_jobs = -1, random_state = 100)



parameter = 'max_depth'



param_grid = {parameter: range(4, 40)}



gcv = tune_hyper_parameter(X_train, y_train, param_grid, 'Individual', xgb)

    

# scores of GridSearch CV

scores = gcv.cv_results_



# Plot the scores

hypertuning_plot(scores, parameter)



# Get the best value

gcv.best_params_
# GridSearchCV to find optimal learning_rate



xgb = XGBClassifier(max_depth = 4, n_jobs = -1, random_state = 100)



parameter = 'learning_rate'



param_grid = {parameter: [0.001, 0.01, 0.1, 0.2, 0.3, 0.6, 0.9, 0.95, 0.99]}



gc = tune_hyper_parameter(X_train, y_train, param_grid, 'Individual', xgb)

    

# scores of GridSearch CV

scores = gc.cv_results_



# Plot the scores

hypertuning_plot(scores, parameter)



# Get the best value

gc.best_params_
# GridSearchCV to find optimal n_estimators



xgb = XGBClassifier(max_depth = 4, learning_rate = 0.1, n_jobs = -1, random_state = 100)



parameter = 'n_estimators'



param_grid = {parameter: range(100, 1100, 100)}



gc = tune_hyper_parameter(X_train, y_train, param_grid, 'Individual', xgb)

    

# scores of GridSearch CV

scores = gc.cv_results_



# Plot the scores

hypertuning_plot(scores, parameter)



# Get the best value

gc.best_params_
# GridSearchCV to find optimal min_child_weight



xgb = XGBClassifier(max_depth = 4, learning_rate = 0.1, n_estimators = 100, n_jobs = -1, random_state = 100)



parameter = 'min_child_weight'



param_grid = {parameter: range(1, 11)}



gc = tune_hyper_parameter(X_train, y_train, param_grid, 'Individual', xgb)

    

# scores of GridSearch CV

scores = gc.cv_results_



# Plot the scores

hypertuning_plot(scores, parameter)



# Get the best value

gc.best_params_
# GridSearchCV to find optimal subsample: 



xgb = XGBClassifier(max_depth = 4, learning_rate = 0.1, n_estimators = 100, min_child_weight = 8,

                    n_jobs = -1, random_state = 100)



parameter = 'subsample'



param_grid = {parameter: np.arange(0.1, 1.1, 0.1)}



gc = tune_hyper_parameter(X_train, y_train, param_grid, 'Individual', xgb)

    

# scores of GridSearch CV

scores = gc.cv_results_



# Plot the scores

hypertuning_plot(scores, parameter)



# Get the best value

gc.best_params_
# GridSearchCV to find optimal colsample_bytree: 



xgb = XGBClassifier(max_depth = 4, learning_rate = 0.1, n_estimators = 100, min_child_weight = 8,

                    subsample = 1.0, n_jobs = -1, random_state = 100)



parameter = 'colsample_bytree'



param_grid = {parameter: np.arange(0.1, 1.1, 0.1)}



gc = tune_hyper_parameter(X_train, y_train, param_grid, 'Individual', xgb)

    

# scores of GridSearch CV

scores = gc.cv_results_



# Plot the scores

hypertuning_plot(scores, parameter)



# Get the best value

gc.best_params_
# Random Forest with best parameters obtained from grid search



xgb = XGBClassifier(max_depth = 4, learning_rate = 0.1, n_estimators = 100, min_child_weight = 8,

                    subsample = 1.0, colsample_bytree = 1.0, n_jobs = -1, random_state = 100)



xgb = xgb.fit(X_train.values, y_train.values.ravel())



# Get the Score Metrics and plots

scores = []



scores = get_scores(scores, xgb, X_test)



# Tabulate results

sampling_results = pd.DataFrame(scores, columns = ['f1', 'precision', 'recall', 'accuracy',

                                                   'auc_roc', 'auc_pr', 'confusion_matrix'])

sampling_results
# LightGBM with Default Parameters



lgb = LGBMClassifier(objective = 'binary', n_jobs = -1, random_state = 100)



lgb = lgb.fit(X_train.values, y_train.values.ravel())



# Get the Score Metrics and plots

scores = []



scores = get_scores(scores, lgb, X_test)



# Tabulate results

sampling_results = pd.DataFrame(scores, columns = ['f1', 'precision', 'recall', 'accuracy',

                                                   'auc_roc', 'auc_pr', 'confusion_matrix'])

sampling_results
# GridSearchCV to find optimal num_leaves



lgb = LGBMClassifier(objective = 'binary', n_jobs = -1, random_state = 100)



parameter = 'num_leaves'



param_grid = {parameter: range(20, 72)}



gc = tune_hyper_parameter(X_train, y_train, param_grid, 'Individual', lgb)

    

# scores of GridSearch CV

scores = gc.cv_results_



# Plot the scores

hypertuning_plot(scores, parameter)



# Get the best value

gc.best_params_
# GridSearchCV to find optimal max_depth



lgb = LGBMClassifier(num_leaves = 20, objective = 'binary', n_jobs = -1, random_state = 100)



parameter = 'max_depth'



param_grid = {parameter: range(8, 72)}



gc = tune_hyper_parameter(X_train, y_train, param_grid, 'Individual', lgb)

    

# scores of GridSearch CV

scores = gc.cv_results_



# Plot the scores

hypertuning_plot(scores, parameter)



# Get the best value

gc.best_params_
# GridSearchCV to find optimal learning_rate



lgb = LGBMClassifier(num_leaves = 20, max_depth = 8, objective = 'binary', n_jobs = -1, random_state = 100)



parameter = 'learning_rate'



param_grid = {parameter: np.arange(0.1, 1.1, 0.1)}



gc = tune_hyper_parameter(X_train, y_train, param_grid, 'Individual', lgb)

    

# scores of GridSearch CV

scores = gc.cv_results_



# Plot the scores

hypertuning_plot(scores, parameter)



# Get the best value

gc.best_params_
# GridSearchCV to find optimal n_estimators



lgb = LGBMClassifier(num_leaves = 20, max_depth = 8, learning_rate = 0.1, objective = 'binary',

                     n_jobs = -1, random_state = 100)



parameter = 'n_estimators'



param_grid = {parameter: range(100, 1100, 100)}



gc = tune_hyper_parameter(X_train, y_train, param_grid, 'Individual', lgb)

    

# scores of GridSearch CV

scores = gc.cv_results_



# Plot the scores

hypertuning_plot(scores, parameter)



# Get the best value

gc.best_params_
# GridSearchCV to find optimal min_child_samples



lgb = LGBMClassifier(num_leaves = 20, max_depth = 8, learning_rate = 0.1, n_estimators = 100,

                     objective = 'binary', n_jobs = -1, random_state = 100)



parameter = 'min_child_samples'



param_grid = {parameter: range(1, 26)}



gc = tune_hyper_parameter(X_train, y_train, param_grid, 'Individual', lgb)

    

# scores of GridSearch CV

scores = gc.cv_results_



# Plot the scores

hypertuning_plot(scores, parameter)



# Get the best value

gc.best_params_
# GridSearchCV to find optimal subsample: 



lgb = LGBMClassifier(num_leaves = 20, max_depth = 8, learning_rate = 0.1, n_estimators = 100, min_child_samples = 4,

                     objective = 'binary', n_jobs = -1, random_state = 100)



parameter = 'subsample'



param_grid = {parameter: np.arange(0.1, 1.1, 0.1)}



gc = tune_hyper_parameter( X_train, y_train, param_grid, 'Individual', lgb)

    

# scores of GridSearch CV

scores = gc.cv_results_



# Plot the scores

hypertuning_plot(scores, parameter)



# Get the best value

gc.best_params_
# GridSearchCV to find optimal colsample_bytree: 



lgb = LGBMClassifier(num_leaves = 20, max_depth = 8, learning_rate = 0.1, n_estimators = 100, min_child_samples = 4,

                     subsample = 0.1, objective = 'binary', n_jobs = -1, random_state = 100)



parameter = 'colsample_bytree'



param_grid = {parameter: np.arange(0.1, 1.1, 0.1)}



gc = tune_hyper_parameter(X_train, y_train, param_grid, 'Individual', lgb)

    

# scores of GridSearch CV

scores = gc.cv_results_



# Plot the scores

hypertuning_plot(scores, parameter)



# Get the best value

gc.best_params_
# LightGBM with best parameters obtained from grid search



lgb = LGBMClassifier(num_leaves = 20, max_depth = 8, learning_rate = 0.1, n_estimators = 100, min_child_samples = 4,

                     subsample = 0.1, colsample_bytree = 0.2, objective = 'binary', n_jobs = -1, random_state = 100)



lgb = lgb.fit(X_train.values, y_train.values.ravel())



# Get the Score Metrics and plots

scores = []



scores = get_scores(scores, lgb, X_test)



# Tabulate results

sampling_results = pd.DataFrame(scores, columns = ['f1', 'precision', 'recall', 'accuracy',

                                                   'auc_roc', 'auc_pr', 'confusion_matrix'])

sampling_results
xgb_pred = xgb.predict(titanic_test_data.values)



xg_pred_file = pd.DataFrame({'PassengerId' : titanic_test_passengerId['PassengerId'],

                       'Survived': xgb_pred.T})

xg_pred_file.to_csv("submit_xg.csv", index=False)
lgb_pred = lgb.predict(titanic_test_data)



lg_pred_file = pd.DataFrame({'PassengerId' : titanic_test_passengerId['PassengerId'],

                       'Survived': lgb_pred.T})

lg_pred_file.to_csv("submit_lgbm.csv", index=False)