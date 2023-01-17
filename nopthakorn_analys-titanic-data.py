%matplotlib inline
import numpy as np
import pandas as pd

#Print you can execute arbitrary python code
train_file = "../input/train.csv"
# utility function
def get_dataframe(csv_file):
    '''read .csv file.

    parameters:
    -----------
    csv_file : a file path in csv format.
    
    return:
    -----------
    return pandas dataframe
    '''
    
    return pd.read_csv(csv_file)
def remove_column(df, key):
    '''
    drop list of key from dataframe, return dataframe that exclude that list key
    
    Parameters:
    -----------
    df  : a dataframe.
    key : a key to remove from dataframe
    
    return:
    -----------
    new_df : data frame without key
    '''
    
    new_df = df.copy()
    
    return new_df.drop(key, axis=1)
def had_family(df):
    '''
    Check is passenger travel with family or alone
    
    Parameters:
    -----------
    df  : a dataframe.
    
    return:
    -----------
    'With family' : if travel with family
    'Alone'       : if travel alone
    '''
        
    if df > 0:
        return 'With family'
    else:
        return 'Alone'

def had_alive(df):
    '''
    Check is passenger who alived or died
    
    Parameters:
    -----------
    df  : a dataframe.
    
    return:
    -----------
    'Alive' : if Survied = 1 
    'Died'  : if Survied = 0
    '''
        
    if df > 0:
        return "alive"
    else:
        return "died"
def had_class(df):
    '''
    Check is passenger class
    
    Parameters:
    -----------
    df  : a dataframe.
    
    return:
    -----------
    '1st class' : if Pclass = 1 
    '2nd class' : if Pclass = 2 
    '3rd class' : if Pclass = 3 
    '''
        
    if df == 1:
        return 'First'
    elif df == 2 :
        return 'Second'
    else:
        return 'Third'
# First let's make a function to sort through the sex 
def had_child(passenger):
    '''
    Treat passenger who age under 16 as a child
    
    Parameters:
    -----------
    passenger  : a objcect.
    
    return:
    -----------
    'child' : if age <= 15
    sex     : if age > 15
 
    '''
        
    # Take the Age and Sex
    age, sex = passenger

    if age <= 15:
        return 'Child'
    else:
        return sex

#------------------------------------------------------------
# read dataframe
#------------------------------------------------------------
titanic = get_dataframe(train_file)

print("Dataset information")
titanic.info()
#------------------------------------------------------------
# Data cleaning
#------------------------------------------------------------

# fill missing "Age" with mean
titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())

# the most common embarkation port is Southampton, so let's assume everyone got on there. 
# replace all the missing values in the Embarked column with S.
titanic['Embarked'] = titanic['Embarked'].fillna('S')

# add "Passenger" column represent who alive or died
titanic['Passenger'] = map(had_alive, titanic['Survived'])

# add "Who" column, to categorize passenger by group of male, female 
# and children as who under 15 as a child, 
titanic['Who'] = titanic[['Age','Sex']].apply(had_child, axis=1)

# add "Class" column represen clss of each passenger
titanic['Class'] = map(had_class, titanic['Pclass'])

# add "Family" column to represent passenger who travel alone or with their family
titanic['Family'] = map(had_family, titanic['Parch'] + titanic['SibSp'])

# drop "Parch" & "SibSp"
titanic = remove_column(titanic, ['Parch', 'SibSp'])


print("Dataset information")
titanic.info()

# import library for the analysis and visualization
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams
from pandas.tools.plotting import scatter_matrix

sns.set(style="white")
%matplotlib inline
# get total passenger
passengercount = float(len(titanic))
# show passenger count
passenger_by_who_class = titanic.groupby(['Who','Class'])
print(passenger_by_who_class.count()['Passenger'])

# set plot title
ax = sns.plt.title('Titanic passengers by Class')

# show the counts of passengers
ax = sns.countplot(x = 'Who', hue = 'Class', data = titanic)

# add percentage for each group
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x(), height + 10, '%1.2f'%((height*100)/passengercount))
# show passenger age
passenger_by_age_who = titanic.groupby(['Who'])
print ("Summary of passenger age")
print (passenger_by_age_who['Age'].mean())
# quick look the distribution
ax = sns.distplot(titanic['Age'], hist = True)
# set plot title
ax = sns.plt.title('Titanic passengers by Age')

# faceplot by passenger type
ax = sns.FacetGrid(titanic, hue = 'Who', aspect = 3)
ax.map(sns.kdeplot,'Age',shade = True)
oldest = titanic['Age'].max()
ax.set(xlim = (0, oldest))
ax.add_legend();
# set title
ax = sns.plt.title('Titanic passengers Age by person')
# map Embarked port character to full name of each town
titanic['Embarked'] = titanic['Embarked'].dropna().map({'C' : 'Cherbourg', 'Q' : 'Queenstown', 'S': 'Southampton' })
# Check passenger who from

passenger_by_embark_who = titanic.groupby('Embarked')
print("Summary passenger by port of embarked")
print(passenger_by_embark_who.count()['Passenger'])

# set plot title
ax = sns.plt.title('Port of Embarkation')

# show the counts of passengers
ax = sns.countplot(x = 'Embarked', data = titanic)

# add percentage for each group
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x(), height + 10, '%1.2f'%((height*100)/passengercount))
# check passengers fare by embarked
ax = sns.FacetGrid(titanic, hue = 'Embarked', aspect = 3)
ax.map(sns.kdeplot,'Fare',shade = True)
oldest = titanic['Fare'].max()
ax.set(xlim = (0,oldest))
ax.add_legend()
ax = sns.plt.title('Ticket fare by port of embarked')
# plot family 

passenger_by_class_who = titanic.groupby(['Embarked', 'Family'])
print(passenger_by_class_who.count()['Passenger'])


# set plot title
ax = sns.plt.title('Titanic passenger family by embarkation')
# show the counts of passengers
ax = sns.countplot(x = 'Embarked', hue = 'Family', data = titanic)

# add percentage for each group
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x(), height + 1, '%1.2f'%((height*100)/passengercount))
# check survial rate
passenger_by_class_who = titanic.groupby('Passenger')
print (passenger_by_class_who.count()['Survived'])

# set plot title
ax = sns.plt.title('Titanic survival passenger')
# show the counts of passengers
ax = sns.countplot(x = 'Passenger', data = titanic)

# add percentage for each group
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x(), height + 2, '%1.2f'%((height*100)/passengercount))
# set range of age for linear plot
age_range = [10,20,40,60,80]

# set range of price
farerange = [0,250,500,750,1000]
# plot survival probability against several variables
ax = sns.factorplot(x = 'Class', y = 'Survived', 
                   hue = 'Who', col = 'Embarked',
                   data = titanic, size = 5, aspect = .6)
# what about if relate gender and age effect survival rate?
ax = sns.lmplot('Age', 'Survived', hue = 'Sex', 
               data = titanic, palette = 'winter', 
               x_bins = age_range)
# check survived rate relate fare by gender
ax = sns.lmplot('Fare', 'Survived', hue = 'Sex',  col = 'Class',
               data = titanic, palette = 'winter', 
               x_bins = farerange)