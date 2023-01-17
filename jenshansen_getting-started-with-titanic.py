# Import libaries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import missingno as msno

import matplotlib.pyplot as plt # plot graphs

import seaborn as sns # plot graphs

from sklearn import preprocessing



import os # read files

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Random forest model

from sklearn.ensemble import RandomForestClassifier

        

# Ignore warnings

import warnings

warnings.filterwarnings('ignore')

print('-'*25)
# Load train data and display first 5 lines

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data.head(5) # show first 5 rows
# Load test data and display first 5 lines

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head(5) # show first 5 rows
# Save datasets in dictionary

data_dict = {}

data_dict['train data'] = train_data

data_dict['test data']  = test_data;

#data_dict['train data'] 
# Shape of train data

train_data.shape
# Basic info train data

train_data.info()
# Shape of test data

test_data.shape
# Basic info train data

test_data.info()
# Create heatmao from numerical features of train data

corr_numeric = sns.heatmap(train_data[["Survived","SibSp","Parch","Age","Fare"]].corr(),

                           annot=True, fmt = ".2f", cmap = "coolwarm", linewidths=5)
## From https://www.kaggle.com/kpacocha/top-5-titanic-machine-learning-from-disaster

    

def compare_features(data,var_1, var_2):

    return data[[var_1, var_2]][data[var_2].isnull()==False].groupby([var_1], 

                 as_index=False).mean().sort_values(by=var_2, ascending=False)

    

def plot(data,var_1, var_2):

    graph = sns.FacetGrid(data, col=var_2).map(sns.distplot, var_1)

    

def counting_values(data, var_1, var_2):

    return data[[var_1, var_2]][data[var_2].isnull()==False].groupby([var_1], 

           as_index=False)
compare_features(train_data, 'Sex','Survived')
compare_features(train_data, 'Pclass','Survived')
# Create Pointplot with Pclass, Survived, Sex:

ax = sns.pointplot(x="Pclass", y="Survived", hue="Sex",

                   data=train_data, size=5,

                   markers=["o", "o"], # Define marker shape

                   linestyles=["--", "--"], # Define linestyle

                   palette={"male": "#34495e", "female": "#3498db"}, # Custom 2- value color palette

                   )



ax = plt.gca() # Get current axis (gca)



# Insert grid

plt.grid(True, alpha=0.5)



# Remove top and left spines:

ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)



# Name axis

plt.ylabel("Survival rate", fontsize=12)

plt.xlabel("Passenger class", fontsize=12);
# Get table with mean age by passenger class

compare_features(train_data, 'Pclass','Age')
# Using boxplot for age and Pclass

sns.factorplot(data = train_data , x = 'Pclass' , y = 'Age', 

               kind = 'box',  palette={1: "#34495e", 2: "#3498db", 3: "r"});
# Plot age distribution for all three passenger classes

g = sns.FacetGrid(train_data, hue="Pclass", size = 2, aspect=4,

                  palette={1: "#34495e", 2: "#3498db", 3: "r"})



g.map(sns.kdeplot,'Age',shade= True)

g.set(xlim=(0, train_data['Age'].max())) # Set limit to x-axis

g.add_legend()

plt.show()
# Histogram comparison of sex, class, and age by survival

h = sns.FacetGrid(train_data, row = 'Survived', col = 'Pclass', hue = 'Sex', 

                  palette={"male": "#34495e", "female": "#3498db"},

                  size=3.5, aspect=0.8,

                  margin_titles=True);



h.map(plt.hist, 'Age', alpha = 0.75);



# Interate through axes

for ax in h.axes.flat:



    # Make right ylabel more human-readable and larger

    # Only the 2nd and 4th axes have something in ax.texts

    if ax.texts:

       

        txt = ax.texts[0] # Get existing right ylabel text

        survived = int(txt.get_text().split('=')[1]) # Extract survival info from text

        

        # Define new right ylabel text

        if survived == 0:

            text_new = 'Perished'

            

        elif survived == 1:

            text_new = 'Survived'

            

        # Set and format new right ylabel text 

        ax.text(txt.get_unitless_position()[0], txt.get_unitless_position()[1],

                text_new,

                transform=ax.transAxes,

                va='center',

                rotation=-90,

                fontsize=12)

        

        # Remove the original text

        ax.texts[0].remove()

        

    ax.set_xlabel(None) # Remove label from x-axis

    

# Add new labels (only y-axis top left and x-axis bottol right)

h.axes[0,0].set_ylabel('Number of cases', fontsize=12);

h.axes[1,2].set_xlabel('Age', fontsize=12);



# Add legend

h.add_legend();
# Create boxplot for passenger class and fare

g = sns.catplot(x="Fare", y="Survived", row="Pclass",

                kind="box", orient="h", height=1.5, aspect=4,

                data=train_data,

                palette={0: "#34495e", 1: "#3498db"})

g.set(xscale="log");
compare_features(train_data, 'SibSp', 'Survived')
compare_features(train_data, 'Parch', 'Survived')
compare_features(train_data, 'Embarked', 'Survived')
train_data.groupby(['Embarked']).mean().drop(['PassengerId', 'SibSp', 'Parch'], axis=1)
# Quick function to combine 2 datasets:

def concat_data(data_1, data_2):

    

    return pd.concat([data_1, data_2], sort=False).reset_index(drop=True)



# Quick function to divide test and train data

def divide_data(all_data):

    

    return all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)



# Concatenate train and test data

data_all = concat_data(train_data, test_data)

data_fs = [train_data, test_data]
# Create infos of missing data

miss_data_dict = {}



for key, dataset in data_dict.items():

    

    # Create missing data metrices

    miss_abs = dataset.isnull().sum()

    miss_rel = miss_abs / dataset.isnull().count()



    # Write information on missing data to a structure variable

    col_abs = '{}: missing values (absolut)'.format(key)

    col_rel = '{}: missing values (relative in %)'.format(key)

    

    if key == 'test data':

        miss_data_dict[key] =  pd.concat([miss_abs.sort_values(ascending=False),

                                                  miss_rel.sort_values(ascending=False)*100], 

                                                  axis=1, keys=[col_abs, col_rel])

    elif key == 'train data':

        miss_data_dict[key] = pd.concat([miss_abs.sort_values(ascending=False),

                                                  miss_rel.sort_values(ascending=False)*100], 

                                                  axis=1, keys=[col_abs, col_rel])

        
# Display missing train data

msno.matrix(train_data,figsize=(9,2),width_ratios=(10,1))



miss_data_dict['train data'].head()
# Display missing test data

msno.matrix(test_data,figsize=(9,2),width_ratios=(10,1))

miss_data_dict['test data'].head()
# Print average age per subcategory ('Pclass' and 'Sex')

data_all.groupby(['Pclass', 'Sex'])['Age'].mean()
# Get indices of NaN age data

index = data_all['Age'].index[data_all['Age'].apply(np.isnan)]



data_all.loc[index, ['Pclass', 'Sex', 'Age']]
train_data.info()
# Replace missing valuse with medium of subgroup

data_all['Age'] = data_all.groupby(['Pclass', 'Sex'])['Age'].apply(lambda x: x.fillna(x.mean()))



# Save changes to data_dict

train_data, test_data = divide_data(data_all)

data_dict['train data'] = train_data

data_dict['test data']  = test_data



data_all.loc[index, ['Pclass', 'Sex', 'Age']]
# Fill missing data with 'Unknown'

data_all.Cabin = data_all.Cabin.fillna('Unknown')



# Extract first letter form 'Cabin' and save to new feature 'Deck'

data_all['Deck'] = data_all['Cabin'].str[0]
data_all.groupby(['Pclass']).Deck.value_counts()
data_all.groupby(['Deck']).mean().drop(['PassengerId', 'SibSp', 'Parch'], axis=1)
# Plot median fare for different decks

sns.factorplot(data = data_all , x = 'Deck' , y = 'Fare', 

               kind = 'box');
# Create list with average fare for each deck, drop category 'U'

deck_avg_fare = data_all.groupby(['Deck']).Fare.mean().drop('U')

deck_avg_fare
# Get indices with 'U' from data

indices_U = data_all[data_all['Deck'] == 'U'].index



# Loop over indices with 'U'

for i in indices_U:

    

    # Get current fare

    fare = data_all.iloc[i].Fare

    

    # Identify nearest average fare for current fare

    nearest_avg_fare = min(deck_avg_fare, key=lambda x:abs(x-fare))



    # Return deck letter for nearest average fare

    deck = deck_avg_fare[deck_avg_fare == nearest_avg_fare].index[0]

    

    # Write to deck category

    data_all['Deck'].iloc[i] = deck

    

# Save changes to data_dict

train_data, test_data = divide_data(data_all)

data_dict['train data'] = train_data

data_dict['test data']  = test_data
data_all.loc[indices_U, ['Fare', 'Deck']]
# Define Function for handling missing data



def handle_missing_data(data, miss_abs):

    

    for col in data:

        

        # identify proportion of missing data

        miss_prop = data[col].isna().sum()/len(data) 

        

        # Drop entire column if missing data is more than 50%

        #if miss_prop > 0.5:

            

        #    data.drop(col, axis=1, inplace=True)

            

            # Report activity

        #   print("Deleting {} since it is having {:.2f} % of missing rows.".format(col,miss_prop*100) )

        

        if  miss_prop < 0.5:

            

            # Fill missing data with mean value if missing data is numeric

            if data[col].dtype == "float64" and data[col].isnull().sum() > 0:

            

                # Calculate mean

                mean = data[col].mean()

            

                # Fill missing data with mean value

                data[col] = data[col].fillna(mean)

                

                # Report activity

                print("Filling {} missing values in {} with mean value {:.0f}.".format(miss_abs[col], col, mean) )

            

            # Fill missing data with most common value if missing data is non-numeric

            elif data[col].dtype == "object" and data[col].isnull().sum() > 0:

                

                # Get top value (most common)

                top = data[col].describe().top

                

                # Fill missing data with top value

                data[col] = data[col].fillna(top)

                

                # Report activity

                print("Filling {} missing values in {} with top value {}.".format(miss_abs[col], col, top) )

            

    return data
#train_data = pd.read_csv("/kaggle/input/titanic/train.csv")





for key, dataset in data_dict.items():

    

    print('{} operations:'.format(key))

    

    miss_abs = miss_data_dict[key]['{}: missing values (absolut)'.format(key)] 

    miss_rel = miss_data_dict[key]['{}: missing values (relative in %)'.format(key)]

    

    data_dict[key] = handle_missing_data(dataset, miss_abs)
msno.matrix(train_data,figsize=(9,2),width_ratios=(10,1))

train_data.info()
msno.matrix(test_data,figsize=(9,2),width_ratios=(10,1))

test_data.info()
# Accessing title data from 'Name' column

data_all['Name'].str.split(',', expand=True)[1].str.split('.', expand=True)[0]
# Define a function for extracting the title

def title_extract(data):

    

    data['Title'] = data['Name'].str.split(',', expand=True)[1].str.split('.', expand=True)[0]
# Apply title extraction function to the dataset

for key, dataset in data_dict.items():

    title = title_extract(dataset)

    

data_all = concat_data(train_data, test_data)



# Display title value count within training data

data_all['Title'].value_counts()
# Assess (in full dataset) whether a title is rare (True if rare)

title_names = (data_all['Title'].value_counts() <  10)



for key, dataset in data_dict.items():



    # Replace titles with 'Misc' if they are rare (= value in title_names = True)

    ## Code from: https://www.kaggle.com/ldfreeman3/a-data-science-framework-to-achieve-99-accuracy

    dataset['Title'] = dataset['Title'].apply(lambda x: 'Misc' if title_names.loc[x] == True else x)



data_all = concat_data(train_data, test_data)



print(data_all['Title'].value_counts())
print('Maximum age in train & test data: ', data_all['Age'].max() )

print('Minimum age in train & test data: ', data_all['Age'].min() )
for key, dataset in data_dict.items():

    

    # Convert age data to integer

    dataset['Age'] = dataset['Age'].astype('int', copy=True)

    

    # Define labels for age group

    age_labels = [1,2,3,4,5]

    

    ## Split the data into 5 chunks of equals size within the range of Age 

    ## (entire dataset as basis!)

    

    # Intervals:

    dataset['Age interval'] = pd.cut(data_all['Age'].astype('int'),5)

    

    # Label intervals with number (integers):

    dataset['Age Code']     = pd.cut(data_all['Age'],5, labels=age_labels).astype('int64')

    

data_all = concat_data(train_data, test_data)

    

print('Number of cases within age intervals in datast:')

data_all['Age interval'].value_counts()
for key, dataset in data_dict.items():

    

    # Define labels for age group

    fare_labels = [1,2,3,4,5]

    

    ## Split the data into 5 chunks of equals number of values 

    ## (entire dataset as basis!):

    

    # Intervals:

    dataset['Fare interval'] = pd.qcut( data_all['Fare'], 5 )

    

    # Label intervals with number (integers):

    dataset['Fare Code'] = pd.qcut( data_all['Fare'], 5, labels=fare_labels).astype('int64')



data_all = concat_data(train_data, test_data)



print('Number of cases per fare category in dataset (approx. equal):')

data_all['Fare interval'].value_counts()
for key, dataset in data_dict.items():

    

    dataset['Family Size'] = dataset ['SibSp'] + dataset['Parch'] + 1                                                                



data_all = concat_data(train_data, test_data)
max_family = dataset['Family Size'].max()

min_family = dataset['Family Size'].min()



x = data_all['Family Size'].hist(bins=max_family-1, grid=False, figsize=(10,3),  zorder=2, rwidth=0.9);



ax = plt.suptitle('Distribution of Family Size onboard the Titanic');

data_all['Family Size'].value_counts()
# Define family size mapping via lambda function

family_mapping = (lambda s: 1 if s == 1 else (2 if s == 2 else (3 if 3 <= s <= 4 else (4 if s >= 5 else 0))))



for key, dataset in data_dict.items():

    

    dataset['Family Size Code'] = dataset['Family Size'].map(family_mapping)

    

# data_all['Family Size categories'] = data_all['Family Size'].map(family_mapping)

data_all = concat_data(train_data, test_data)

data_all['Family Size Code'].value_counts()
data_all[ ['Family Size Code', 'Survived'] ].groupby('Family Size Code')['Survived'].mean().plot(kind='bar', figsize=(10,3))

ax = plt.suptitle('Survival rates in relation to family size:');
for key, dataset in data_dict.items():

    

    dataset['Mother'] = np.where((dataset.Title == 'Mrs') & (dataset.Parch >0),1,0)    



data_all = concat_data(train_data, test_data)
for key, dataset in data_dict.items():



    dataset['Ticket Frequency'] = data_all.groupby('Ticket')['Ticket'].transform('count')

    

data_all = concat_data(train_data, test_data)
train_data.head()
# Collect features

features = ['Sex', 'Age Code', 'SibSp', 'Parch', 'Mother', 'Family Size Code',

            'Pclass', 'Fare Code', 'Embarked', 'Fare', 'Deck',

            'Title', 'Ticket Frequency']





print(data_all[features].info())

data_all[features].head(5)
# Get dummy code for nominal features

# (function 'get_dummies' just affets the variables with dtype 'object')

X_train = pd.get_dummies(train_data[features])

X_test  = pd.get_dummies(test_data[ features])



X_test.head()
# Get target variable 'Survived'

y = train_data['Survived']
# Define model

#model = RandomForestClassifier(n_estimators=60,max_depth=5,random_state=1)



model = RandomForestClassifier(n_estimators=1800,

                               max_depth=8,

                               min_samples_split=6,

                               min_samples_leaf=6,

                               max_features='auto',

                               oob_score=True,

                               random_state=42,

                               n_jobs=-1,

                               verbose=1)





# Fit model to train data "survived"

model.fit(X_train, y)



# Get predictions

predictions = model.predict(X_test)



#model.score(X,y)

acc_random_forest = round(model.score(X_train, y) * 100, 2)

print(acc_random_forest)





# Compile prediction with PassengerID into data frame

output = pd.DataFrame({'PassengerID': test_data.PassengerId, 'Survived': predictions})



output = output.convert_dtypes() # also convert output to int!



# Write data frame with predition to csv file

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")



output.head()