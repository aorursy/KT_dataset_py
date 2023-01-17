import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import eli5

from eli5.sklearn import PermutationImportance

from collections import Counter

import missingno as msno



import warnings

warnings.filterwarnings('ignore')

import plotly

sns.set_style('darkgrid')
df=pd.read_csv('../input/data.csv')
df.head().T
df.columns
df.info()
df.describe().T
df.drop(['Unnamed: 0','Photo','Flag','Club Logo'],axis=1,inplace=True)
msno.bar(df.sample( 18207 ),(28,10),color='red')
df.isnull().sum()
missing_height = df[df['Height'].isnull()].index.tolist()

missing_weight = df[df['Weight'].isnull()].index.tolist()

if missing_height == missing_weight:

    print('They are same')

else:

    print('They are different')
df.drop(df.index[missing_height],inplace =True)
df.isnull().sum()
df.drop(['Loaned From','Release Clause','Joined'],axis=1,inplace=True)
#Number of countries available and top 5 countries with highest number of players

print('Total number of countries : {0}'.format(df['Nationality'].nunique()))

print(df['Nationality'].value_counts().head(5))

print('--'*40)

print("\nEuropean Countries have most players")
#Total number of clubs present and top 5 clubs with highest number of players

print('Total number of clubs : {0}'.format(df['Club'].nunique()))

print(df['Club'].value_counts().head(5))
#Player with maximum Potential and Overall Performance

print('Maximum Potential : '+str(df.loc[df['Potential'].idxmax()][1]))

print('Maximum Overall Perforamnce : '+str(df.loc[df['Overall'].idxmax()][1]))
pr_cols=['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',

       'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl',

       'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',

       'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots',

       'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties',

       'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving',

       'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']

print('BEST IN DIFFERENT ASPECTS :')

print('_________________________\n\n')

i=0

while i < len(pr_cols):

    print('Best {0} : {1}'.format(pr_cols[i],df.loc[df[pr_cols[i]].idxmax()][1]))

    i += 1
#Cleaning some of values so that we can interpret them 

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
df.head().T
#Top earners

print('Most valued player : '+str(df.loc[df['Value'].idxmax()][1]))

print('Highest earner : '+str(df.loc[df['Wage'].idxmax()][1]))

print("--"*40)

print("\nTop Earners")
sns.jointplot(x=df['Age'],y=df['Potential'],

              joint_kws={'alpha':0.1,'s':5,'color':'red'},

              marginal_kws={'color':'red'})



player_features = (

    'Acceleration', 'Aggression', 'Agility', 

    'Balance', 'BallControl', 'Composure', 

    'Crossing', 'Dribbling', 'FKAccuracy', 

    'Finishing', 'GKDiving', 'GKHandling', 

    'GKKicking', 'GKPositioning', 'GKReflexes', 

    'HeadingAccuracy', 'Interceptions', 'Jumping', 

    'LongPassing', 'LongShots', 'Marking', 'Penalties'

)



from math import pi

idx = 1

plt.figure(figsize=(15,45))

for position_name, features in df.groupby(df['Position'])[player_features].mean().iterrows():

    top_features = dict(features.nlargest(5))

    

    # number of variable

    categories=top_features.keys()

    N = len(categories)



    # We are going to plot the first line of the data frame.

    # But we need to repeat the first value to close the circular graph:

    values = list(top_features.values())

    values += values[:1]



    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)

    angles = [n / float(N) * 2 * pi for n in range(N)]

    angles += angles[:1]



    # Initialise the spider plot

    ax = plt.subplot(10, 3, idx, polar=True)



    # Draw one axe per variable + add labels labels yet

    plt.xticks(angles[:-1], categories, color='grey', size=8)

 # Draw ylabels

    ax.set_rlabel_position(0)

    plt.yticks([25,50,75], ["25","50","75"], color="grey", size=7)

    plt.ylim(0,100)

    

    plt.subplots_adjust(hspace = 0.5)

    

    # Plot data

    ax.plot(angles, values, linewidth=1, linestyle='solid')



    # Fill area

    ax.fill(angles, values, 'b', alpha=0.1)

    

    plt.title(position_name, size=11, y=1.1)

    

    idx += 1
sns.lmplot(data = df, x = 'Age', y = 'SprintSpeed',lowess=True,scatter_kws={'alpha':0.01, 's':5,'color':'green'}, 

           line_kws={'color':'red'})
sns.lmplot(x = 'BallControl', y = 'Dribbling', data = df,col = 'Preferred Foot',scatter_kws = {'alpha':0.1,'color':'orange'},

           line_kws={'color':'red'})


sns.jointplot(x=df['Dribbling'], y=df['Crossing'], kind="hex", color="#4CB391");
value = df.Value

cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)



sns.relplot(x="Age", y="Potential", hue=value/100000, 

            sizes=(40, 400), alpha=.5,

            height=6, data=df);
corr = df.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(15, 15))

    ax = sns.heatmap(corr,mask=mask,square=True,linewidths=.8,cmap="YlGnBu")
plt.rcParams['figure.figsize'] = (20, 7)

plt.style.use('seaborn-dark-palette')



sns.boxenplot(df['Overall'], df['Age'], hue = df['Preferred Foot'], palette = 'rocket')

plt.title('Comparison of Overall Scores and age wrt Preferred foot', fontsize = 20)

plt.show()
cols = ['Age','Overall','Potential','Acceleration','SprintSpeed',"Agility","Stamina",'Strength','Preferred Foot']

df_small = df[cols]
df_small.head()
sns.pairplot(df_small, hue ='Preferred Foot',palette=["black", "red"],plot_kws=dict(s=50, alpha =0.8),markers=['^','v'])
df=pd.read_csv('../input/data.csv')
#DROP UNNECESSARY VALUES

drop_cols = df.columns[28:54]

df = df.drop(drop_cols, axis = 1)

df = df.drop(['Unnamed: 0','ID','Photo','Flag','Club Logo','Jersey Number','Joined','Special','Loaned From','Body Type', 'Release Clause',

               'Weight','Height','Contract Valid Until','Wage','Value','Name','Club'], axis = 1)

df = df.dropna()

df.head()
#Turn Real Face into a binary indicator variable

def face_to_num(df):

    if (df['Real Face'] == 'Yes'):

        return 1

    else:

        return 0

    

#Turn Preferred Foot into a binary indicator variable

def right_footed(df):

    if (df['Preferred Foot'] == 'Right'):

        return 1

    else:

        return 0



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



#Get a count of Nationalities in the Dataset, make of list of those with over 250 Players (our Major Nations)

nat_counts = df.Nationality.value_counts()

nat_list = nat_counts[nat_counts > 250].index.tolist()



#Replace Nationality with a binary indicator variable for 'Major Nation'

def major_nation(df):

    if (df.Nationality in nat_list):

        return 1

    else:

        return 0



#Create a copy of the original dataframe to avoid indexing errors

df1 = df.copy()



#Apply changes to dataset to create new column

df1['Real_Face'] = df1.apply(face_to_num, axis=1)

df1['Right_Foot'] = df1.apply(right_footed, axis=1)

df1['Simple_Position'] = df1.apply(simple_position,axis = 1)

df1['Major_Nation'] = df1.apply(major_nation,axis = 1)



#Split the Work Rate Column in two

tempwork = df1["Work Rate"].str.split("/ ", n = 1, expand = True) 

#Create new column for first work rate

df1["WorkRate1"]= tempwork[0]   

#Create new column for second work rate

df1["WorkRate2"]= tempwork[1]

#Drop original columns used

df1 = df1.drop(['Work Rate','Preferred Foot','Real Face', 'Position','Nationality'], axis = 1)

df1.head()

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
perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())

#Top 3 important features are Potential, Age & Reactions 
#Visualising the results

plt.figure(figsize=(18,10))

sns.regplot(predictions,y_test,scatter_kws={'color':'red','edgecolor':'blue','linewidth':'0.7'},line_kws={'color':'black','alpha':0.5})

plt.xlabel('Predictions')

plt.ylabel('Overall')

plt.title("Linear Prediction of Player Rating")

plt.show()