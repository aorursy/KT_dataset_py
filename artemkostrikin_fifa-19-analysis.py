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
import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

%matplotlib inline
data = pd.read_csv('/kaggle/input/fifa19/data.csv')
data.head()
# Get the Basic info of the dataset

data.describe()
data.info()
num_rows = data.shape[0] # Provide the number of rows in the dataset

num_cols = data.shape[1] # Provide the number of columns in the dataset

print("Row    number: {}".format(num_rows))

print("Column number: {}".format(num_cols))
# To check the column names in the dataset

data.columns
# Data Preparation Step 1: Drop the columns which will not be used in this project

data.drop('Photo',     axis = 1,inplace=True)

data.drop('Flag',      axis = 1,inplace=True)

data.drop('Club Logo', axis = 1,inplace=True)

data.drop('ID',        axis = 1,inplace=True)

data.head()
# Data Preparation Step 2: Check whether any column has missing values

columns_with_missing_values = set(data.columns[data.isnull().mean()!=0])

print(columns_with_missing_values)
# Supporting function to convert string values into numbers

def str2number(amount):

    """

    This function perform convertion from amount values in string type to float type numbers

    

    Parameter:

    amount(str): Amount values in string type with M & K as Abbreviation for Million and Thousands

    

    Returns:

    float: A float number represents the numerical value of the input parameter amount(str)

    """

    if amount[-1] == 'M':

        return float(amount[1:-1])*1000000

    elif amount[-1] == 'K':

        return float(amount[1:-1])*1000

    else:

        return float(amount[1:])
# Data Preparation Step 3: Convert string values into numbers for Value & Wage



# Create New Wage_Number column to store numerical type Wage info

data['Wage_Number']  = data['Wage'].map(lambda x: str2number(x))



#Create New Value_Number column to store numerical type Value info

data['Value_Number'] = data['Value'].map(lambda x: str2number(x))
# Data Preparation Step 4: One-Hot Encoding for Categorical variables such as Club, Nationality, Preferred Positions

# Select only position and stored in New 'Position' column

data['Position'] = data['Position'].str.split().str[0]



# One-hot encode the feature: "Club" , "Nationality" and "Preferred Position"

le = LabelEncoder()

data['Club_onehot_encode']               = le.fit_transform(data['Club'].astype(str))

data['Nationality_onehot_encode']        = le.fit_transform(data['Nationality'].astype(str))

data['Preferred_Position_onehot_encode'] = le.fit_transform(data['Position'].astype(str))
# Question 1: Which Nation has most number of Soccer Players collected in FIFA 19, list the top 20 Nations

nationality_vals = data.Nationality.value_counts()

print(nationality_vals.head(30))



(nationality_vals.head(30)/data.shape[0]).plot(kind="bar");

plt.title("Top 20 FIFA 19 Players Nationality Distribution(in percentage)")
# Question 2: How about the age distribution of the FIFA 19 Players?

age_vals = data.Age.value_counts()

print(age_vals.head(30))



(age_vals.head(30)/data.shape[0]).plot(kind="bar");

plt.title("FIFA 19 Players Age Distribution (in percentage)");
# Question 3: Find out the top 10 clubs with highest total player market value, and the highest average player wage

Value_Wage_DF = data[["Name", "Club", "Value_Number", "Wage_Number"]]

Value_Wage_DF.head()
# Find out the top 10 clubs with the highest average wage

Value_Wage_DF.groupby("Club")["Wage_Number"].mean().sort_values(ascending=False).head(10).plot(kind="bar");

plt.title("Top 10 clubs with the highest average wage")
# Find out the top 10 clubs with the highest total player market value

Value_Wage_DF.groupby("Club")["Value_Number"].sum().sort_values(ascending=False).head(10).plot(kind="bar");

plt.title("Top 10 clubs with the highest total Value")
# Question 4: Choose the best squad

BestSquad_DF = data[['Name', 'Age', 'Overall', 'Potential', 'Position']]

BestSquad_DF.head()
def find_best_squad(position):

    BestSquad_DF_copy = BestSquad_DF.copy()

    BestSquad = []

    for i in position:

        BestSquad.append([i,BestSquad_DF_copy.loc[[BestSquad_DF_copy[BestSquad_DF_copy['Position'] == i]['Overall'].idxmax()]]['Name'].to_string(index = False), BestSquad_DF_copy[BestSquad_DF_copy['Position'] == i]['Overall'].max()])

        BestSquad_DF_copy.drop(BestSquad_DF_copy[BestSquad_DF_copy['Position'] == i]['Overall'].idxmax(), inplace = True)



    return pd.DataFrame(np.array(BestSquad).reshape(11,3), columns = ['Position', 'Player', 'Overall']).to_string(index = False)
# Formation 4-3-3

squad_Formation433 = ['GK', 'LB', 'CB', 'CB', 'RB', 'LM', 'CDM', 'RM', 'LW', 'ST', 'RW']

print ('Best Squad of Formation 4-3-3')

print (find_best_squad(squad_Formation433))
# Formation 3-4-1-2

squad_Formation3412 = ['GK', 'CB', 'CB', 'CB', 'LM', 'CM', 'CM', 'RM', 'CAM', 'ST', 'ST']

print ('Best Squad of Formation 3-4-1-2')

print (find_best_squad(squad_Formation3412))
# Question 5: Correlation between Age, Overall, Potential, Position, Club, Nationality, Special vs Value/Wage

Correlation_DF = data[['Name', 'Age', 'Overall', 'Potential', 'Preferred_Position_onehot_encode', 'Club_onehot_encode', 'Nationality_onehot_encode', 'Special', 'Value_Number', 'Wage_Number']]



Correlation_DF.corr()
colormap = plt.cm.inferno

plt.figure(figsize=(16,12))

plt.title('Correlation between Age, Overall, Potential, Position, Club, Nationality, Special vs Value/Wage', y=1.05, size=15)

sns.heatmap(Correlation_DF.corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)
