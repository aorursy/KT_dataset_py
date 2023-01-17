# Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

#for displaying 500 results in pandas dataframe
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Importing file
# File Name

import os
print(os.listdir("../input/fifa19"))

# OR

# File Path
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


        
df = pd.read_csv('/kaggle/input/fifa19/data.csv')
df.head()
#Shape of dataframe
print(" Shape of dataframe: ", df.shape)
# Drop duplicates
df.drop_duplicates()
print(" Shape of dataframe after dropping duplicates: ", df.shape)
#Variable inspection

print("Names of columns ", list(df.columns))
df= df.drop(columns= "Unnamed: 0")

#Null values

null= df.isnull().sum().sort_values(ascending=False)
total =df.shape[0]
percent_missing= (df.isnull().sum()/total).sort_values(ascending=False)

missing_data= pd.concat([null, percent_missing], axis=1, keys=['Total missing', 'Percent missing'])

missing_data.reset_index(inplace=True)
missing_data= missing_data.rename(columns= { "index": " column name"})
 
print ("Null Values in each column:\n", missing_data)

#See see how null values look in dataframe
#Missing data as white lines 
import missingno as msno
msno.matrix(df,color=(0,0.3,0.9))
#Filtering data with null values for position

df_isnull = pd.isnull(df["LB"])
pos_null= df[df_isnull]
print(pos_null.shape)
print(pos_null.isnull().sum().sort_values(ascending= False))
#Filtering relevant data & checking for club null values

df_notnull = pd.notnull(df["LB"])
df= df[df_notnull]

df_isnull = pd.isnull(df["Club"])
pos_null= df[df_isnull]
print(pos_null.shape)
print(pos_null.isnull().sum().sort_values(ascending= False))
df_notnull = pd.notnull(df["Club"])
df= df[df_notnull]
print(df.shape)
print(df.isnull().sum().sort_values(ascending= False))
print(df.info())

df[['Height_ft','Height_inch']] = df['Height'].str.split("'",expand=True)
df["Height"]= 2.54*(df["Height_inch"].astype(str).astype(int))+30.48 *(df["Height_ft"].astype(str).astype(int))
df= df.drop(columns= ['Height_ft','Height_inch'])

df.columns
df["Weight"]= df["Weight"].str.split("lbs", n = 1, expand = True)
df["Weight"].astype(str).astype(int)
df[['a','Release Clause']] = df['Release Clause'].str.split("€",expand=True)
#df[['Release Clause','b']] = df['b'].str.split("M",expand=True)
df= df.drop(columns= ['a'])
df["Release Clause"]= df["Release Clause"].replace(regex=['k'], value='000')
df["Release Clause"]= df["Release Clause"].replace(regex=['K'], value='000')
df["Release Clause"]= df["Release Clause"].replace(regex=['M'], value='000000')
df["Release Clause"]=df["Release Clause"].astype(str).astype(float)
df["Release Clause"]=df["Release Clause"]/1000000
df.head()

df[['a','Value']] = df['Value'].str.split("€",expand=True)
df= df.drop(columns= ['a'])
df["Value"]= df["Value"].replace(regex=['k'], value='000')
df["Value"]= df["Value"].replace(regex=['K'], value='000')
df["Value"]= df["Value"].replace(regex=['M'], value='000000')
df["Value"]=df["Value"].astype(str).astype(float)
df["Value"]=df["Value"]/1000000
df.head()

df["Wage"]= df["Wage"].replace(regex=['k'], value='000')
df["Wage"]= df["Wage"].replace(regex=['K'], value='000')
df["Wage"]= df["Wage"].replace(regex=['M'], value='000000')
df[["b",'Wage']] = df['Wage'].str.split("€",expand=True)
df= df.drop(columns= ['b'])


df["Wage"]=df["Wage"].astype(str).astype(float)
df.head()
#df["LS"]= df["LS"].str.split("+", n = 1, expand = True) 
#df["LS"]= df["LS"].astype(str).astype(int)
#df.shape
split= ["LS","ST","RS","LW","LF","CF","RF","RW","LAM","CAM","RAM","LM","LCM","CM","RCM","RM","LWB",
        "LDM","CDM","RDM","RWB","LB","LCB","CB","RCB","RB"]
df = df.apply(lambda x : x.str.split('+').str[0].astype(str).astype(int) if x.name in ["LS","ST","RS","LW","LF","CF","RF","RW","LAM","CAM","RAM","LM","LCM","CM","RCM","RM","LWB",
        "LDM","CDM","RDM","RWB","LB","LCB","CB","RCB","RB"] else x)
    
df.head()
df=df.fillna(0)
unique_position= df.Position.unique()
unique_position
# Creating subsets according to playing positions

attack = ["CF", "LF", "LS", "LW", "RF", "RS", "RW", "ST"]
# NOt in df_attack= df[~df.Position.isin(attack) ]
df_attack= df[df.Position.isin(attack) ]
print(df_attack.shape)

defense= ["RWB","RCB", "RB", "LWB","LCB", "LB","CB"]
df_defense= df[df.Position.isin(defense)]
print(df_defense.shape)

mid= ["RM","RDM","RCM","RAM","LM","LDM","LAM","LCM","CM","CDM","CAM"]
df_mid= df[df.Position.isin(mid)]
print(df_mid.shape)


gk= ["GK"]
df_gk= df[df.Position.isin(gk)]
print(df_gk.shape)

df_defense.info()
#Defender with maximum Potential and Overall
print('Maximum Potential : '+str(df_defense.loc[df_defense['Potential'].idxmax()][1]))
print('Maximum Overall Ranking : '+str(df_defense.loc[df_defense['Overall'].idxmax()][1]))
# Exploring dependent variable
print(df.Value.describe())
sns.distplot(df.Value);
#skewness and kurtosis
print("Skewness: %f" % df['Value'].skew())
print("Kurtosis: %f" % df['Value'].kurt())

#histogram and normal probability plot
sns.distplot(df['Value'], fit=norm);
fig = plt.figure()
res = stats.probplot(df['Value'], plot=plt)
boxplot = df.boxplot(column=['Value'])
#scatter plot 
#### Independent variables (Numerical): 'Overall', 'Potential',  'Wage', "International Reputation", "Weak Foot", "Skill Moves","Work Rate", "LS","ST","RS","LW","LF","CF","RF","RW","LAM","CAM","RAM","LM","LCM","CM","RCM","RM","LWB","LDM","CDM","RDM","RWB","LB","LCB","CB","RCB","RB",'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause'
#Using important variables

ind_var= ['Overall', 'Potential',  'Wage', "International Reputation"]
for x in ind_var:
    var = x
    data = pd.concat([df['Value'], df[var]], axis=1)
    data.plot.scatter(x=var, y='Value')
#df_defense_1= df_defense[df_defense.Overall>70] #N is equal in Overall>70
df_1= df[df.Value>10]
print(df_1.shape)
print(df.shape)
#scatter plot 
#### Independent variables (Numerical): 'Overall', 'Potential',  'Wage', "International Reputation", "Weak Foot", "Skill Moves","Work Rate", "LS","ST","RS","LW","LF","CF","RF","RW","LAM","CAM","RAM","LM","LCM","CM","RCM","RM","LWB","LDM","CDM","RDM","RWB","LB","LCB","CB","RCB","RB",'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause'
ind_var= ['Overall', 'Potential',  'Wage', "International Reputation"]
for x in ind_var:
    var = x
    data = pd.concat([df_1['Value'], df_1[var]], axis=1)
    data.plot.scatter(x=var, y='Value')
df_2= df[df["Overall"]>70]
df_2= df[df.Value>50]
print(df_2.shape)
print(df.shape)
print(df_2["Overall"].describe())

#scatter plot 
#### Independent variables (Numerical): 'Overall', 'Potential',  'Wage', "International Reputation", "Weak Foot", "Skill Moves","Work Rate", "LS","ST","RS","LW","LF","CF","RF","RW","LAM","CAM","RAM","LM","LCM","CM","RCM","RM","LWB","LDM","CDM","RDM","RWB","LB","LCB","CB","RCB","RB",'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause'
ind_var= ['Overall', 'Potential',  'Wage', "International Reputation"]
for x in ind_var:
    var = x
    data = pd.concat([df_2['Value'], df_2[var]], axis=1)
    data.plot.scatter(x=var, y='Value')
df_3= df[df.Value<10]
print(df_3.shape)
print(df.shape)
print(df_3["Overall"].describe())

#scatter plot 
#### Independent variables (Numerical): 'Overall', 'Potential',  'Wage', "International Reputation", "Weak Foot", "Skill Moves","Work Rate", "LS","ST","RS","LW","LF","CF","RF","RW","LAM","CAM","RAM","LM","LCM","CM","RCM","RM","LWB","LDM","CDM","RDM","RWB","LB","LCB","CB","RCB","RB",'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause'
ind_var= ['Overall', 'Potential',  'Wage', "International Reputation"]
for x in ind_var:
    var = x
    data = pd.concat([df_3['Value'], df_3[var]], axis=1)
    data.plot.scatter(x=var, y='Value')
#df_defense_4= df_defense[df_defense["Overall"]<70]
df_4= df[df.Value>1]
print(df_4.shape)
print(df.shape)
print(df_4["Overall"].describe())

#scatter plot 
#### Independent variables (Numerical): 'Overall', 'Potential',  'Wage', "International Reputation", "Weak Foot", "Skill Moves","Work Rate", "LS","ST","RS","LW","LF","CF","RF","RW","LAM","CAM","RAM","LM","LCM","CM","RCM","RM","LWB","LDM","CDM","RDM","RWB","LB","LCB","CB","RCB","RB",'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause'
ind_var= ['Overall', 'Potential',  'Wage', "International Reputation"]
for x in ind_var:
    var = x
    data = pd.concat([df_4['Value'], df_4[var]], axis=1)
    data.plot.scatter(x=var, y='Value')
df_5= df[df.Value<1]
print(df_5.shape)
print(df.shape)
print(df_5["Overall"].describe())

#scatter plot 
#### Independent variables (Numerical): 'Overall', 'Potential',  'Wage', "International Reputation", "Weak Foot", "Skill Moves","Work Rate", "LS","ST","RS","LW","LF","CF","RF","RW","LAM","CAM","RAM","LM","LCM","CM","RCM","RM","LWB","LDM","CDM","RDM","RWB","LB","LCB","CB","RCB","RB",'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause'
ind_var= ['Overall', 'Potential',  'Wage', "International Reputation"]
for x in ind_var:
    var = x
    data = pd.concat([df_5['Value'], df_5[var]], axis=1)
    data.plot.scatter(x=var, y='Value')
cut_labels=["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1"]
cut_bins= [0,0.2,0.4,0.6,0.8,1]
df_5["Value_cuts"]= pd.cut(df_5["Value"], bins= cut_bins, labels= cut_labels)

df_5["Value_cuts"].value_counts(normalize=True)
print(df_5.groupby("Value_cuts")["Wage"].mean())
print(df_5.groupby("Value_cuts")["Overall"].mean())
print(df_5.groupby("Value_cuts")["Potential"].mean())
print(df_5.groupby("Value_cuts")["International Reputation"].value_counts(normalize=True))
#histogram and normal probability plot
sns.distplot(df['Overall'], fit=norm);
fig = plt.figure()
res = stats.probplot(df['Overall'], plot=plt)

#skewness and kurtosis
print("Skewness: %f" % df['Overall'].skew())
print("Kurtosis: %f" % df['Overall'].kurt())

boxplot= df.boxplot(column=['Overall', 'Potential']);
#Relationship with independent variables
#scatter plot 
#### Independent variables (Numerical):  'Potential',  'Wage', "International Reputation", "Weak Foot", "Skill Moves","Work Rate", 
####"LS","ST","RS","LW","LF","CF","RF","RW","LAM","CAM","RAM","LM","LCM","CM","RCM","RM","LWB","LDM","CDM","RDM","RWB","LB","LCB",
####"CB","RCB","RB",'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes', 'Release Clause'
ind_var= ["LWB","LDM","CDM","RDM","RWB","LB","LCB","CB","RCB","RB",'ShortPassing',  'LongPassing', 
          'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance',  
          'Jumping','Stamina', 'Strength', 'Aggression', 'Interceptions', 'Positioning', 
          'Vision', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle' ]
for x in ind_var:
    var = x
    data = pd.concat([df['Overall'], df[var]], axis=1)
    data.plot.scatter(x=var, y='Overall')
sns.jointplot(x=df_defense['Age'],y=df_defense['Overall'],
              joint_kws={'alpha':0.1,'s':5,'color':'Green'},
              marginal_kws={'color':'Green'})

boxplot= df.boxplot(column=['Wage']);
#correlation matrix
corrmat = df_defense.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
sns.set()

cols = df_defense[["Overall","Age", "Potential","LWB","LDM","CDM","RDM","RWB","LB","LCB","CB","RCB","RB" ]]
corr = cols.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, annot=True, vmax=.8, square=True);
sns.set()

cols = df[["Overall","CB","Age",'ShortPassing',  'LongPassing', 'BallControl', 'Acceleration', 
                   'SprintSpeed', 'Agility', 'Reactions', 'Balance',  
          'Jumping','Stamina', 'Strength', 'Aggression', 'Interceptions', 'Positioning', 
          'Vision', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle'  ]]
corr = cols.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, annot=True, vmax=.8, square=True);
#Drop unnecessary columns

df_model= df_defense.drop(columns=['ID', 'Name', 'Photo', 'Nationality', 
                           'Flag','Club', 'Club Logo', 'Value','Wage', 
                           'Special', 'Preferred Foot','Weak Foot', 
                           'Skill Moves', 'Work Rate', 'Body Type',
                           'Real Face', 'Position', 'Jersey Number', 
                           'Joined', 'Loaned From', 'Contract Valid Until',
                           'Height', 'Weight', 'LS', 'ST', 'RS', 'LW', 'LF', 
                           'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM','LM', 
                           'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 
                           'RDM', 'RWB', 'LB', 'LCB',  'RCB', 'RB','Crossing', 
                           'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',  
                           'Dribbling', 'Curve', 'FKAccuracy', 'BallControl', 'Acceleration',
                           'Reactions','ShotPower', 'LongShots',  'Interceptions', 'Penalties', 
                           'Marking','StandingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 
                           'GKPositioning', 'GKReflexes', 'Release Clause'])
df_model.columns
#Split Overall as a Target value
target = df_model.Overall
df_model2 = df_model.drop(['Overall'], axis = 1)

#Splitting into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_model2, target, test_size=0.2)

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
#Using PermutationImportance to see important variables

import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())
plt.figure(figsize=(18,10))
sns.regplot(predictions,y_test,scatter_kws={'alpha':0.6,'color':'orange'},line_kws={'color':'black','alpha':0.6})
plt.xlabel("Predictors")
plt.ylabel('Overall')
plt.title("Linear Model for Overall Rating Prediction")
plt.show()
# However, we require a model for the master data set
df_model= df.drop(columns=['ID', 'Name', 'Photo', 'Nationality', 
                           'Flag','Club', 'Club Logo', 'Value','Wage', 
                           'Special', 'Preferred Foot','Weak Foot', 
                           'Skill Moves', 'Work Rate', 'Body Type',
                           'Real Face', 'Position', 'Jersey Number', 
                           'Joined', 'Loaned From', 'Contract Valid Until',
                           'Height', 'Weight', 'LS', 'ST', 'RS', 'LW', 'LF', 
                           'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM','LM', 
                           'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 
                           'RDM', 'RWB', 'LB', 'LCB',  'RCB', 'RB','Crossing', 
                           'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys',  
                           'Dribbling', 'Curve', 'FKAccuracy', 'BallControl', 'Acceleration',
                           'Reactions','ShotPower', 'LongShots',  'Interceptions', 'Penalties', 
                           'Marking','StandingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 
                           'GKPositioning', 'GKReflexes', 'Release Clause'])
#Split Overall as a Target value
target = df_model.Overall
df_model2 = df_model.drop(['Overall'], axis = 1)

#Splitting into test and train
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_model2, target, test_size=0.2)

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
#Using PermutationImportance to see important variables

import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)
eli5.show_weights(perm, feature_names = X_test.columns.tolist())
plt.figure(figsize=(18,10))
sns.regplot(predictions,y_test,scatter_kws={'alpha':0.6,'color':'orange'},line_kws={'color':'black','alpha':0.6})
plt.xlabel("Predictors")
plt.ylabel('Overall')
plt.title("Linear Model for Overall Rating Prediction")
plt.show()
def get_mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mape = get_mape(y_test, predictions)
mape