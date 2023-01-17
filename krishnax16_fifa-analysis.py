# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import model_selection

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/fifa19/data.csv')
df.head()
df.describe()
df['Preferred Foot'].value_counts()
df.dtypes
df.columns
def value_to_int(df_value):

    try:

        value = float(df_value[1:-1])

        suffix = df_value[-1:]



        if suffix == 'M':

            value = value * 1000000

        elif suffix == 'K':

            value = value * 1000

    except ValueError:

        value = 0

    return value



df['Value'] = df['Value'].apply(value_to_int)

df['Wage'] = df['Wage'].apply(value_to_int)
df.drop(['Unnamed: 0'],axis=1,inplace=True)
df.drop(['Photo','Club Logo','Real Face'],axis=1,inplace=True)
print(df['Jersey Number'].nunique())

print(df['Jersey Number'].value_counts())

df['Jersey Number'].hist(bins=10,grid = False)

print(max(df['Jersey Number']))
missing_height = df[df['Height'].isnull()].index.tolist()

df.drop(df.index[missing_height],inplace=True)
df.drop(['Release Clause'],axis=1,inplace=True)
pot =df.loc[df['Potential'].idxmax()][1]

ovr =df.loc[df['Overall'].idxmax()][1]

inte =df.loc[df['International Reputation'].idxmax()][1]

shot =df.loc[df['ShotPower'].idxmax()][1]

skill =df.loc[df['Skill Moves'].idxmax()][1]

acc =df.loc[df['Acceleration'].idxmax()][1]

print("Highest Acceleration "+acc+" "+ str(max(df['Acceleration'])))

print("Highest Skill Moves "+skill+" "+ str(max(df['Skill Moves'])))

print("Highest Shot Power "+shot+" "+ str(max(df['ShotPower'])))

print("Highest Potential "+pot+" "+ str(max(df['Potential'])))

print("Highest International Reputation "+inte+" "+ str(max(df['International Reputation'])))

print("Highest Overall "+ovr+" "+str(max(df['Overall'])))
attr_cols=['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',

       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',

       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',

       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',

       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',

       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',

       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']

print('BEST IN DIFFERENT ASPECTS :')

print('_________________________\n\n')

for i in attr_cols:

    print('Best {0} : {1}'.format(i,df.loc[df[i].idxmax()][1]))
new_df = df[['Overall', 'Potential', 'Skill Moves', 'Position','Height', 'Weight', 'LS', 'ST', 'RS','LW','LF', 'CF', 'RF', 'RW', 'LAM','CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM','RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB','Crossing','Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys','Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl','Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance','ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots','Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties','Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving','GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']]
new_df.info()
new_df['Position'].fillna('Unknown',inplace= True)
new_df.isnull().sum()
sns.pairplot(new_df[["Skill Moves","Finishing","FKAccuracy","SprintSpeed","Acceleration","Volleys","Dribbling","Penalties"]], palette='deep')
plt.rcParams['figure.figsize'] = (15, 7)



sns.countplot(new_df['Position'],palette="deep")

plt.title("Player's positions distribution", fontsize = 20)

plt.tick_params(axis='x', rotation=70)

plt.show()
plt.rcParams['figure.figsize'] = (20, 7)

plt.style.use('seaborn-dark-palette')



sns.boxenplot(df['Overall'], df['Age'], hue = df['Preferred Foot'], palette = 'rocket')

plt.title('Comparison of Overall Scores and age wrt Preferred foot', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (20, 7)

plt.style.use('seaborn-dark-palette')



sns.boxenplot(df['Overall'], df['International Reputation'], hue = df['Preferred Foot'], palette = 'rocket')

plt.title('Comparison of Overall Scores and age wrt Preferred foot', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (20, 7)

plt.style.use('seaborn-dark-palette')



sns.boxenplot(df['Overall'], df['Value'], hue = df['Preferred Foot'], palette = 'rocket')

plt.title('Comparison of Overall Scores and age wrt Preferred foot', fontsize = 20)

plt.show()
plt.rcParams['figure.figsize'] = (20, 7)

plt.style.use('seaborn-dark-palette')



sns.boxenplot(df['Overall'], df['Potential'], hue = df['Preferred Foot'], palette = 'rocket')

plt.title('Comparison of Overall Scores and age wrt Preferred foot', fontsize = 20)

plt.show()
old =df.loc[df['Age'].idxmax()][1]

print("Oldest Player is " + old + " "+str(max(df['Age'])) )

#Oldest Player
print(df['Nationality'].nunique())

print(df['Nationality'].value_counts())
print(df['Club'].nunique())

print(df['Club'].value_counts())
df1 = new_df.copy()
df1.drop(['Height', 'Weight', 'LS', 'ST', 'RS','LW','LF', 'CF', 'RF', 'RW', 'LAM','CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM','RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB'],axis=1,inplace = True)
#Create a simplified position varaible to account for all player positions

def simple_position(df):

    if (df['Position'] == 'GK'):

        return 'GK'

    elif ((df['Position'] == 'RB') | (df['Position'] == 'LB') | (df['Position'] == 'CB') | (df['Position'] == 'LCB') | (df['Position'] == 'RCB') | (df['Position'] == 'RWB') | (df['Position'] == 'LWB') ):

        return 'DF'

    elif ((df['Position'] == 'LDM') | (df['Position'] == 'CDM') | (df['Position'] == 'RDM')):

        return 'DM'

    elif ((df['Position'] == 'LM') | (df['Position'] == 'LCM') | (df['Position'] == 'CM') | (df['Position'] == 'RCM') | (df['Position'] == 'RM')):

        return 'MF'

    elif ((df['Position'] == 'LAM') | (df['Position'] == 'CAM') | (df['Position'] == 'RAM') | (df['Position'] == 'LW') | (df['Position'] == 'RW')):

        return 'AM'

    elif ((df['Position'] == 'RS') | (df['Position'] == 'ST') | (df['Position'] == 'LS') | (df['Position'] == 'CF') | (df['Position'] == 'LF') | (df['Position'] == 'RF')):

        return 'ST'

    else:

        return df.Position

df1['Simple_Position'] = df1.apply(simple_position,axis = 1)
#Split ID as a Target value

target = df1.Overall

df2 = df1.drop(['Overall'], axis = 1)



#Splitting into test and train

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df2, target, test_size=0.2)



#One Hot Encoding

X_train = pd.get_dummies(X_train)

X_test = pd.get_dummies(X_test)

print(X_test.shape,X_train.shape)

print(y_test.shape,y_train.shape)
#Applying Linear Regression

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)



#Finding the r2 score and root mean squared error

from sklearn.metrics import r2_score, mean_squared_error

print('r2 score: '+str(r2_score(y_test, predictions)))

print('RMSE : '+str(np.sqrt(mean_squared_error(y_test, predictions))))

#Visualising the results

plt.figure(figsize=(18,10))

sns.regplot(predictions,y_test,scatter_kws={'color':'red','edgecolor':'blue','linewidth':'0.7'},line_kws={'color':'black','alpha':0.5})

plt.xlabel('Predictions')

plt.ylabel('Overall')

plt.title("Linear Prediction of Player Rating")

plt.show()
#Applying Logistic Regression

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)



#Finding the r2 score and root mean squared error

from sklearn.metrics import r2_score, mean_squared_error

print('r2 score: '+str(r2_score(y_test, predictions)))

print('RMSE : '+str(np.sqrt(mean_squared_error(y_test, predictions))))



from sklearn.metrics import accuracy_score

print("accuracy: "+ str(accuracy_score(y_test,predictions)))
#Visualising the results

plt.figure(figsize=(18,10))

sns.regplot(predictions,y_test,scatter_kws={'color':'red','edgecolor':'blue','linewidth':'0.7'},line_kws={'color':'black','alpha':0.5})

plt.xlabel('Predictions')

plt.ylabel('Overall')

plt.title("Logistic Regression Prediction of Player Rating")

plt.show()
#Applying Logistic RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train, y_train)

predictions = model.predict(X_test)



#Finding the r2 score and root mean squared error

from sklearn.metrics import r2_score, mean_squared_error

print('r2 score: '+str(r2_score(y_test, predictions)))

print('RMSE : '+str(np.sqrt(mean_squared_error(y_test, predictions))))



from sklearn.metrics import accuracy_score

print("accuracy: "+ str(accuracy_score(y_test,predictions)))
#Visualising the results

plt.figure(figsize=(18,10))

sns.regplot(predictions,y_test,scatter_kws={'color':'red','edgecolor':'blue','linewidth':'0.7'},line_kws={'color':'black','alpha':0.5})

plt.xlabel('Predictions')

plt.ylabel('Overall')

plt.title("RandomForestClassifier Prediction of Player Rating")

plt.show()
#Applying Logistic KNeighborsClassifier

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier()

model.fit(X_train, y_train)

predictions = model.predict(X_test)



#Finding the r2 score and root mean squared error

from sklearn.metrics import r2_score, mean_squared_error

print('r2 score: '+str(r2_score(y_test, predictions)))

print('RMSE : '+str(np.sqrt(mean_squared_error(y_test, predictions))))



from sklearn.metrics import accuracy_score

print("accuracy: "+ str(accuracy_score(y_test,predictions)))
#Visualising the results

plt.figure(figsize=(18,10))

sns.regplot(predictions,y_test,scatter_kws={'color':'red','edgecolor':'blue','linewidth':'0.7'},line_kws={'color':'black','alpha':0.5})

plt.xlabel('Predictions')

plt.ylabel('Overall')

plt.title("KNeighborsClassifier Prediction of Player Rating")

plt.show()