# Import the friendly libraries. 

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

import seaborn as sns
train = pd.read_csv('../input/train.csv')

train.info()
cm = train.corr()

print(cm['Survived'].sort_values(ascending= False))
def readAndCleanUpData(fileName):    

    df = pd.read_csv(fileName)



    # Drop ticket and cabin

    df.drop(['Ticket', 'Cabin'], axis = 1, inplace = True)



    # Replace male with 0, female with 1

    df.Sex = df.Sex.map( {'male': 0, 'female': 1} )

    

    # Fill fare with median value

    df['Fare'].fillna( df.Fare.median(), inplace = True)



    # In Embarked we have 3 null values. We will replace this with median values

    # First convert to 0 1 and 2

    df.replace( {'S': 0, 'C':1, 'Q':2}, inplace = True)

    df['Embarked'].fillna( df['Embarked'].median(), inplace = True)    



    # Get title of all passengers

    titles = df.Name.str.split(',').str.get(1).str.split('\.').str.get(0)

    titles.value_counts()

    

    # We have Dr, Sir, Don, Capt, major, Rev. Replace, Jonkheer with Mr    

    df.Name.replace(['Dr', 'Sir', 'Don', 'Capt', 'Major', 'Rev', 'Col', 'Jonkheer'], 'Mr', regex = True, inplace = True)        

    # Replace Ms, Mlle with Miss

    df.Name.replace(['Ms', 'Mlle'], 'Miss', regex = True, inplace = True)    



    # Replace Lady, Countess, Mme with Mrs

    df.Name.replace(['Lady', 'the Countess', 'Mme'], 'Mrs', regex = True, inplace = True)        



    # Now get mediam mr, mrs, master ages and replace na with mean

    idx = df['Name'].str.contains('Mr\.')

    median = df.Age[idx].mean()

    df.loc[ idx, 'Age'] = df.loc[ idx, 'Age'].fillna(median)



    idx = df['Name'].str.contains('Mrs\.')

    median = df.Age[idx].median()

    df.loc[ idx, 'Age'] = df.loc[ idx, 'Age'].fillna(median)



    idx = df['Name'].str.contains('Miss\.')

    median = df.Age[idx].median()

    df.loc[ idx, 'Age'] = df.loc[ idx, 'Age'].fillna(median)



    idx = df['Name'].str.contains('Master\.')

    median = df.Age[idx].median()

    df.loc[ idx, 'Age'] = df.loc[ idx, 'Age'].fillna(median)

    

    df.Age = df.Age.astype(int)

    

    # Create a new column on relatives

    df['Relatives'] = df['SibSp'] + df['Parch']

    df['isAlone'] = 0

    df.loc[ df.Relatives == 0, 'isAlone'] = 1



    # We can delete the name column now    

    pId= df.PassengerId

    df = df.drop(['Name', 'SibSp', 'Parch', 'Relatives', 'PassengerId'], axis = 1)

    

    # Normalize Fare    

    return df, pId
# Read in training data

df, pId = readAndCleanUpData('../input/train.csv')

# Read in test data

t_df, pId = readAndCleanUpData('../input/test.csv')

nTrain = df.shape[0] # This is number of training data we have



# Get surival data from train set

y     = df.Survived
allData = pd.concat( [df.drop(['Survived'], axis = 1), t_df] )

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

colNames = list( allData.columns.values )

allData_Scaled = pd.DataFrame( scaler.fit_transform(allData), columns = colNames )

# Split back now

X = allData_Scaled.iloc[:nTrain, :]

t_df_c = allData_Scaled.iloc[nTrain:, :]
# Correlation matrix

cm = X.corr()

% matplotlib inline

import matplotlib.pylab as plt

sns.set(font_scale=1.2)

fig, ax = plt.subplots(figsize = (10,10))

sns.heatmap(ax = ax, data = cm, vmax = 0.8, square = True, fmt = '.2g', annot = True)

plt.show()
from sklearn.ensemble import RandomForestClassifier

params = {'bootstrap': False, 'min_samples_leaf': 5, 'n_estimators': 80, 'min_samples_split': 10, 

              'max_features': 'auto', 'max_depth': 20}

model = RandomForestClassifier(**params)

model.fit(X, y)
# A subroutine to compute the accuracy. Maybe scikit has it

from sklearn.model_selection import cross_val_score

def compute_accuracy(model, X, y):

    return np.mean( cross_val_score(model, X, y, cv = 5, scoring='accuracy') )  

print( "Accuracy of the model ", compute_accuracy(model, X, y) )
# Predictions

y_pred = model.predict( t_df_c )



# Create predictions

predictions =  pd.DataFrame( {'PassengerId' : pId,

                             'Survived'    : y_pred} )



# Save the output

predictions.to_csv("my_predictions.csv", index = False)