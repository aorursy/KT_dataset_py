#Import Libraries

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
fifa = pd.read_csv("/kaggle/input/fifa19/data.csv")

fifa.head()
fifa.shape
fifa.info()
# dropping ALL duplicte values 

fifa.drop_duplicates(subset ="Name",  keep = False, inplace = True)
fifa.shape
fifa.drop(['ID', 'Unnamed: 0', 'Photo','Flag','Club Logo', 'Jersey Number', 'Loaned From', 'Real Face', 

           'Release Clause', 'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM',

          'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB'], axis=1,inplace=True)
def value_to_int(fifa_value):

    try:

        value = int(float(fifa_value[1:-1]))

        suffix = fifa_value[-1:]



        if suffix == 'M':

            value = value * 1000000

        elif suffix == 'K':

            value = value * 1000

    except ValueError:

        value = 0

    return value



fifa['Value'] = fifa['Value'].apply(value_to_int)

fifa['Wage'] = fifa['Wage'].apply(value_to_int)
fifa['Height'].unique()
def replace_height(height):

    try: 

        return height.replace("'", ".")

    except AttributeError:

        # first we return zero value to calculate mean height.

        #return 0

        # to fill missing values after calculate mean height we return 5.8 in inch as a value return

        return 5.8



fifa['Height'] = fifa['Height'].map(replace_height)
def feet_to_cm(height_feet):

    height_cm=round(float(height_feet)/int(1)*int(12)*float(2.54))

    return height_cm

fifa['Height'] = fifa['Height'].apply(feet_to_cm) 
fifa['Height'].unique()
def replace_weight(weight):

    try: 

        return weight.replace('lbs', '')

    except AttributeError:

        # first we return zero value to calculate mean weight in Kg.

        #return 0

        # to fill missing values after calculate mean weight we return 165 in lbs as a value return

        return 165

fifa['Weight'] = fifa['Weight'].apply(replace_weight)
def lbs_to_kg(weight_lbs):

    weight_kg=round(int(weight_lbs)/int(1)*float(0.453592))

    return weight_kg

fifa['Weight'] = fifa['Weight'].apply(lbs_to_kg)
def year_joined(joined_date):

    try:

        return joined_date.replace(joined_date, joined_date[-4:])

    except AttributeError:

        return 2018



fifa['Joined'] = fifa['Joined'].apply(year_joined)
def year_joined(joined_date):

    try:

        return joined_date.replace(joined_date, joined_date[-4:])

    except AttributeError:

        return 2020



fifa['Contract Valid Until'] = fifa['Contract Valid Until'].apply(year_joined)
fifa['Club'].fillna('No Club', inplace = True)

fifa['Preferred Foot'].fillna('Right', inplace = True)

fifa['International Reputation'].fillna(fifa['International Reputation'].mean(), inplace= True)

fifa['Weak Foot'].fillna(fifa['Weak Foot'].mean(), inplace = True)

fifa['Skill Moves'].fillna(fifa['Skill Moves'].mean(), inplace = True)

fifa['Work Rate'].fillna('Medium/ Medium', inplace = True)

fifa['Body Type'].fillna('Normal', inplace = True)

fifa['Position'].fillna('ST', inplace = True)

fifa['Crossing'].fillna(fifa['Crossing'].mean(), inplace = True)

fifa['Finishing'].fillna(fifa['Finishing'].mean(), inplace = True)

fifa['HeadingAccuracy'].fillna(fifa['HeadingAccuracy'].mean(), inplace = True)

fifa['ShortPassing'].fillna(fifa['ShortPassing'].mean(), inplace = True)

fifa['Volleys'].fillna(fifa['Volleys'].mean(), inplace = True)

fifa['Dribbling'].fillna(fifa['Dribbling'].mean(), inplace = True)

fifa['Curve'].fillna(fifa['Curve'].mean(), inplace = True)

fifa['FKAccuracy'].fillna(fifa['FKAccuracy'].mean(), inplace = True)

fifa['LongPassing'].fillna(fifa['LongPassing'].mean(), inplace = True)

fifa['BallControl'].fillna(fifa['BallControl'].mean(), inplace = True)

fifa['Acceleration'].fillna(fifa['Acceleration'].mean(), inplace = True)

fifa['SprintSpeed'].fillna(fifa['SprintSpeed'].mean(), inplace = True)

fifa['Agility'].fillna(fifa['Agility'].mean(), inplace = True)

fifa['Reactions'].fillna(fifa['Reactions'].mean(), inplace = True)

fifa['Balance'].fillna(fifa['Balance'].mean(), inplace = True)

fifa['ShotPower'].fillna(fifa['ShotPower'].mean(), inplace = True)

fifa['Jumping'].fillna(fifa['Jumping'].mean(), inplace = True)

fifa['Stamina'].fillna(fifa['Stamina'].mean(), inplace = True)

fifa['Strength'].fillna(fifa['Strength'].mean(), inplace = True)

fifa['LongShots'].fillna(fifa['LongShots'].mean(), inplace = True)

fifa['Aggression'].fillna(fifa['Aggression'].mean(), inplace = True)

fifa['Interceptions'].fillna(fifa['Interceptions'].mean(), inplace = True)

fifa['Positioning'].fillna(fifa['Positioning'].mean(), inplace = True)

fifa['Vision'].fillna(fifa['Vision'].mean(), inplace = True)

fifa['Penalties'].fillna(fifa['Penalties'].mean(), inplace = True)

fifa['Composure'].fillna(fifa['Composure'].mean(), inplace = True)

fifa['Marking'].fillna(fifa['Marking'].mean(), inplace = True)

fifa['StandingTackle'].fillna(fifa['StandingTackle'].mean(), inplace = True)

fifa['SlidingTackle'].fillna(fifa['SlidingTackle'].mean(), inplace = True)

fifa['GKDiving'].fillna(fifa['GKDiving'].mean(), inplace = True)

fifa['GKHandling'].fillna(fifa['GKHandling'].mean(), inplace = True)

fifa['GKKicking'].fillna(fifa['GKKicking'].mean(), inplace = True)

fifa['GKPositioning'].fillna(fifa['GKPositioning'].mean(), inplace = True)

fifa['GKReflexes'].fillna(fifa['GKReflexes'].mean(), inplace = True)
fifa['International Reputation']= fifa['International Reputation'].astype('int64')

fifa['Weak Foot']=fifa['Weak Foot'].astype('int64')

fifa['Skill Moves']=fifa['Skill Moves'].astype('int64')

fifa['Crossing']=fifa['Crossing'].astype('int64')

fifa['Finishing']=fifa['Finishing'].astype('int64')

fifa['HeadingAccuracy']=fifa['HeadingAccuracy'].astype('int64')

fifa['ShortPassing']=fifa['ShortPassing'].astype('int64')

fifa['Volleys']=fifa['Volleys'].astype('int64')

fifa['Dribbling']=fifa['Dribbling'].astype('int64')

fifa['Curve']=fifa['Curve'].astype('int64')

fifa['FKAccuracy']=fifa['FKAccuracy'].astype('int64')

fifa['LongPassing']=fifa['LongPassing'].astype('int64')

fifa['BallControl']=fifa['BallControl'].astype('int64')

fifa['Acceleration']=fifa['Acceleration'].astype('int64')

fifa['SprintSpeed']=fifa['SprintSpeed'].astype('int64')

fifa['Agility']=fifa['Agility'].astype('int64')

fifa['Reactions']=fifa['Reactions'].astype('int64')

fifa['Balance']=fifa['Balance'].astype('int64')

fifa['ShotPower']=fifa['ShotPower'].astype('int64')

fifa['Jumping']=fifa['Jumping'].astype('int64')

fifa['Stamina']=fifa['Stamina'].astype('int64')

fifa['Strength']=fifa['Strength'].astype('int64')

fifa['LongShots']=fifa['LongShots'].astype('int64')

fifa['Aggression']=fifa['Aggression'].astype('int64')

fifa['Interceptions']=fifa['Interceptions'].astype('int64')

fifa['Positioning']=fifa['Positioning'].astype('int64')

fifa['Vision']=fifa['Vision'].astype('int64')

fifa['Penalties']=fifa['Penalties'].astype('int64')

fifa['Composure']=fifa['Composure'].astype('int64')

fifa['Marking']=fifa['Marking'].astype('int64')

fifa['StandingTackle']=fifa['StandingTackle'].astype('int64')

fifa['SlidingTackle']=fifa['SlidingTackle'].astype('int64')

fifa['GKDiving']=fifa['GKDiving'].astype('int64')

fifa['GKHandling']=fifa['GKHandling'].astype('int64')

fifa['GKKicking']=fifa['GKKicking'].astype('int64')

fifa['GKPositioning']=fifa['GKPositioning'].astype('int64')

fifa['GKReflexes']=fifa['GKReflexes'].astype('int64')
order = fifa['Nationality'].value_counts(ascending=False).head(50).index

plt.figure(figsize=(14,16))



ax = sns.countplot(y="Nationality", data=fifa, order=order, palette='Paired') 

ax.set_xlabel('Count',fontsize=20)

ax.set_ylabel('Nationality', fontsize=20)

ax.set_title('Distribution of Nationality',fontsize=20, weight='bold')





total = len(fifa['Nationality'])

for p in ax.patches:

        percentage ='{:.2f}%'.format(100 * p.get_width()/total)

        width, height =p.get_width(),p.get_height()

        x=p.get_x()+width+3

        y=p.get_y()+height-0.1

        ax.annotate(percentage,(x,y))

plt.xticks(fontsize =13)

plt.yticks(fontsize =13)

plt.show()
plt.figure(figsize=(18,9))

ax = sns.countplot(x="Age", data=fifa)

ax.set_xlabel('Age',fontsize=14)

ax.set_ylabel('Count', fontsize=14)

ax.set_title('Distribution of Age',fontsize=20, weight='bold')



total = len(fifa['Age'])

for p in ax.patches:

        percentage ='{:.2f}%'.format(100 * p.get_height()/total)

        width, height =p.get_width(),p.get_height()

        x=p.get_x()+width-0.8

        y=p.get_y()+height

        ax.annotate(percentage,(x,y))

        

plt.xticks(fontsize =13)

plt.yticks(fontsize =13)

plt.show()
plt.figure(figsize=(18,9))

ax = sns.boxplot(x="Age", y="Overall", data=fifa)

ax.set_xlabel('Age',fontsize=16)

ax.set_ylabel('Overall', fontsize=16)

ax.set_title('Player Age and Overall',fontsize=20, weight='bold')       

plt.xticks(fontsize =13)

plt.yticks(fontsize =13)

plt.show()
plt.figure(figsize=(18,9))

ax = sns.boxplot(x="Age", y="Potential", data=fifa)

ax.set_xlabel('Age',fontsize=16)

ax.set_ylabel('Potential', fontsize=16)

ax.set_title('Player Age and Potential',fontsize=20, weight='bold')       

plt.xticks(fontsize =13)

plt.yticks(fontsize =13)

plt.show()
plt.figure(figsize=(20,5))

ax = sns.countplot(x="Position", data=fifa)

ax.set_xlabel('Position',fontsize=14)

ax.set_ylabel('Count', fontsize=14)

ax.set_title('Distribution of Position',fontsize=20, weight='bold')



total = len(fifa['Position'])

for p in ax.patches:

        percentage ='{:.2f}%'.format(100 * p.get_height()/total)

        width, height =p.get_width(),p.get_height()

        x=p.get_x()+width-0.8

        y=p.get_y()+height

        ax.annotate(percentage,(x,y))

        

plt.xticks(fontsize =13)

plt.yticks(fontsize =13)

plt.show()
plt.hist('Overall', data=fifa, color='blue') 

plt.style.use('ggplot')

plt.title("Distribution of Overall") 

plt.xlabel("Overall")  

plt.ylabel("Count") 

plt.show()
plt.hist('Potential', data=fifa, color='pink') 

plt.style.use('ggplot')

plt.title("Distribution of Potential")  

plt.xlabel("Potential")  

plt.ylabel("Count") 

plt.show()
plt.figure(figsize=(15,5))

ax = sns.countplot(x="Work Rate", data=fifa)

ax.set_xlabel('Work Rate',fontsize=14)

ax.set_ylabel('Count', fontsize=14)

ax.set_title('Distribution of Work Rate',fontsize=20, weight='bold')



total = len(fifa['Work Rate'])

for p in ax.patches:

        percentage ='{:.2f}%'.format(100 * p.get_height()/total)

        width, height =p.get_width(),p.get_height()

        x=p.get_x()+width-0.6

        y=p.get_y()+height

        ax.annotate(percentage,(x,y))

        

plt.xticks(fontsize =13)

plt.yticks(fontsize =13)

plt.show()
plt.figure(figsize=(5,5))

ax = sns.countplot(x="Preferred Foot", data=fifa)

ax.set_xlabel('Preferred Foot',fontsize=14)

ax.set_ylabel('Count', fontsize=14)

ax.set_title('Distribution of Prefferd Foot',fontsize=20, weight='bold')



total = len(fifa['Preferred Foot'])

for p in ax.patches:

        percentage ='{:.2f}%'.format(100 * p.get_height()/total)

        width, height =p.get_width(),p.get_height()

        x=p.get_x()+width-0.6

        y=p.get_y()+height

        ax.annotate(percentage,(x,y))

        

plt.xticks(fontsize =13)

plt.yticks(fontsize =13)

plt.show()
plt.figure(figsize=(20,5))

ax = sns.countplot(x="Body Type", data=fifa)

ax.set_xlabel('Body Type',fontsize=14)

ax.set_ylabel('Count', fontsize=14)

ax.set_title('Distribution of Body Type',fontsize=20, weight='bold')



total = len(fifa['Body Type'])

for p in ax.patches:

        percentage ='{:.2f}%'.format(100 * p.get_height()/total)

        width, height =p.get_width(),p.get_height()

        x=p.get_x()+width-0.6

        y=p.get_y()+height

        ax.annotate(percentage,(x,y))

        

plt.xticks(fontsize =13)

plt.yticks(fontsize =13)

plt.show()
order = fifa['Weight'].value_counts(ascending=False).index

plt.figure(figsize=(14,16))



ax = sns.countplot(y="Weight", data=fifa, order=order, palette='Paired') 

ax.set_xlabel('Count',fontsize=20)

ax.set_ylabel('Weight', fontsize=20)

ax.set_title('Distribution of Weight',fontsize=20, weight='bold')





total = len(fifa['Weight'])

for p in ax.patches:

        percentage ='{:.2f}%'.format(100 * p.get_width()/total)

        width, height =p.get_width(),p.get_height()

        x=p.get_x()+width+3

        y=p.get_y()+height-0.1

        ax.annotate(percentage,(x,y))

plt.xticks(fontsize =13)

plt.yticks(fontsize =13)

plt.show()
order = fifa['Height'].value_counts(ascending=False).index

plt.figure(figsize=(14,7))



ax = sns.countplot(y="Height", data=fifa, order=order, palette='Paired') 

ax.set_xlabel('Count',fontsize=20)

ax.set_ylabel('Height', fontsize=20)

ax.set_title('Distribution of Height',fontsize=20, weight='bold')





total = len(fifa['Height'])

for p in ax.patches:

        percentage ='{:.2f}%'.format(100 * p.get_width()/total)

        width, height =p.get_width(),p.get_height()

        x=p.get_x()+width+3

        y=p.get_y()+height-0.1

        ax.annotate(percentage,(x,y))

plt.xticks(fontsize =13)

plt.yticks(fontsize =13)

plt.show()
plt.figure(figsize = (9, 5)) 

fifa['Special'].plot(kind ="hist")
plt.figure(figsize=(20,5))

ax = sns.countplot(x="International Reputation", data=fifa)

ax.set_xlabel('International Reputation',fontsize=14)

ax.set_ylabel('Count', fontsize=14)

ax.set_title('Distribution of International Reputation',fontsize=20, weight='bold')



total = len(fifa['International Reputation'])

for p in ax.patches:

        percentage ='{:.2f}%'.format(100 * p.get_height()/total)

        width, height =p.get_width(),p.get_height()

        x=p.get_x()+width-0.6

        y=p.get_y()+height

        ax.annotate(percentage,(x,y))

        

plt.xticks(fontsize =13)

plt.yticks(fontsize =13)

plt.show()
plt.figure(figsize=(20,5))

ax = sns.countplot(x="Weak Foot", data=fifa)

ax.set_xlabel('Weak Foot',fontsize=14)

ax.set_ylabel('Count', fontsize=14)

ax.set_title('Distribution of Weak Foot',fontsize=20, weight='bold')



total = len(fifa['Weak Foot'])

for p in ax.patches:

        percentage ='{:.2f}%'.format(100 * p.get_height()/total)

        width, height =p.get_width(),p.get_height()

        x=p.get_x()+width-0.6

        y=p.get_y()+height

        ax.annotate(percentage,(x,y))

        

plt.xticks(fontsize =13)

plt.yticks(fontsize =13)

plt.show()
plt.figure(figsize=(20,5))

ax = sns.countplot(x="Skill Moves", data=fifa)

ax.set_xlabel('Skill Moves',fontsize=14)

ax.set_ylabel('Count', fontsize=14)

ax.set_title('Distribution of Skill Moves',fontsize=20, weight='bold')



total = len(fifa['Skill Moves'])

for p in ax.patches:

        percentage ='{:.2f}%'.format(100 * p.get_height()/total)

        width, height =p.get_width(),p.get_height()

        x=p.get_x()+width-0.6

        y=p.get_y()+height

        ax.annotate(percentage,(x,y))

        

plt.xticks(fontsize =13)

plt.yticks(fontsize =13)

plt.show()
plt.figure(figsize=(18,9))

ax = sns.boxenplot(x="International Reputation", y="Overall", scale="linear", data=fifa)

ax.set_xlabel('International Reputation',fontsize=16)

ax.set_ylabel('Overall', fontsize=16)

ax.set_title('Player International Reputation and Overall',fontsize=20, weight='bold')       

plt.xticks(fontsize =13)

plt.yticks(fontsize =13)

plt.show()
plt.figure(figsize=(18,9))

ax = sns.boxenplot(x="International Reputation", y="Potential", scale="linear", data=fifa)

ax.set_xlabel('International Reputation',fontsize=16)

ax.set_ylabel('Overall', fontsize=16)

ax.set_title('Player International Reputation and Potential',fontsize=20, weight='bold')       

plt.xticks(fontsize =13)

plt.yticks(fontsize =13)

plt.show()
fifa.groupby('Club')['Overall'].mean().sort_values(ascending=False).head(10).plot(

    kind='bar', figsize=(10,5), color='chocolate')

plt.xlabel('Club', fontsize=18)

plt.ylabel('Overall', fontsize=18)

plt.title('Top 10 Clubs with the highest Overall', fontsize=18)
fifa.groupby('Club')['Potential'].mean().sort_values(ascending=False).head(10).plot(

    kind='bar', figsize=(10,5), color='sandybrown')

plt.xlabel('Club', fontsize=18)

plt.ylabel('Potential', fontsize=18)

plt.title('Top 10 Clubs with the highest Potential', fontsize=18)
fifa.groupby('Club')['Value'].sum().sort_values(ascending=False).head(10).plot(

    kind='bar', figsize=(10,5), color='mediumturquoise')

plt.xlabel('Club', fontsize=18)

plt.ylabel('Value', fontsize=18)

plt.title('Top 10 Clubs with the highest total Value', fontsize=18)
fifa.groupby('Club')['Wage'].mean().sort_values(ascending=False).head(10).plot(

    kind='bar', figsize=(10,5), color='orangered')

plt.xlabel('Club', fontsize=18)

plt.ylabel('Wage', fontsize=18)

plt.title('Top 10 Clubs with the highest average Wage', fontsize=18)
fifa.groupby('Club')['International Reputation'].mean().sort_values(ascending=False).head(10).plot(

    kind='bar', figsize=(10,5), color='chocolate')

plt.xlabel('Club', fontsize=18)

plt.ylabel('International Reputation', fontsize=18)

plt.title('Top 10 Club with the highest average International Reputation', fontsize=18)
fifa.loc[fifa.groupby(['Position', 'Preferred Foot'])['Value'].idxmax()][['Name', 'Age', 'Position', 'Value', 'Wage',

                                                                'Overall', 'Club', 'Nationality', 'Preferred Foot']]
fifa.loc[fifa.groupby(['Position', 'Preferred Foot'])['Overall'].idxmax()][['Name', 'Age', 'Position', 'Value', 'Wage'

                                                            , 'Overall', 'Club', 'Nationality', 'Preferred Foot']]
fifa.loc[fifa.groupby(['Position', 'Preferred Foot'])['Special'].idxmax()][['Name', 'Age', 'Position', 'Value', 'Wage'

                                                     , 'Overall', 'Club', 'Nationality', 'Special', 'Preferred Foot']]
corrmat = fifa.corr()

  

f, ax = plt.subplots(figsize =(26, 16)) 

sns.heatmap(corrmat, ax = ax,  linewidths = 0.1, annot=True)
corrmat = fifa.corr()   

cg = sns.clustermap(corrmat, linewidths = 0.1, annot=True, figsize=(28, 18)) 

plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation = 0) 

  

cg
k = 19

  

cols = corrmat.nlargest(k, 'Special')['Special'].index 

  

cm = np.corrcoef(fifa[cols].values.T) 

f, ax = plt.subplots(figsize =(18, 10)) 

  

sns.heatmap(cm, ax = ax, 

            linewidths = 0.1,annot=True, yticklabels = cols.values,  

                              xticklabels = cols.values)
import statsmodels.api as sm

fifa['intercept']=1

lm_mlr=sm.OLS(fifa['Special'], fifa[['intercept', 'BallControl', 'ShortPassing', 'Dribbling', 'Crossing', 'Curve',

                                            'LongPassing', 'LongShots', 'ShotPower', 'Positioning', 'FKAccuracy', 'Stamina',

                                            'Volleys', 'Skill Moves', 'Vision', 'Composure', 'Penalties', 'Finishing', 

                                             'Agility']])

results_mlr=lm_mlr.fit()

results_mlr.summary()
from sklearn.model_selection import train_test_split

X = fifa[['Potential']]

y = fifa[['Overall']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=44, shuffle =True)
# Import Libraries

from sklearn.linear_model import LinearRegression



#Applying Linear Regression Model 



LinearRegressionModel = LinearRegression(fit_intercept=True, normalize=True,copy_X=True,n_jobs=-1)

LinearRegressionModel.fit(X_train, y_train)



#Calculating Details

print('Linear Regression Train Score is : ' , LinearRegressionModel.score(X_train, y_train))

print('Linear Regression Test Score is : ' , LinearRegressionModel.score(X_test, y_test))

print('Linear Regression Coef is : ' , LinearRegressionModel.coef_)

print('Linear Regression intercept is : ' , LinearRegressionModel.intercept_)



#Calculating Prediction

y_LRM = LinearRegressionModel.predict(X_test)
plt.scatter(X_test, y_test,  color='black')

plt.plot(X_test, y_LRM, color='blue', linewidth=3)

plt.xlabel("Potential")

plt.ylabel("Overall")

plt.show()
from sklearn.metrics import mean_squared_error, r2_score #common metris to evaluate regression models

# The mean squared error

print("Mean squared error is : %.2f" % mean_squared_error(y_test, y_LRM))

 

# Explained variance score: 1 is perfect prediction

print('Variance score is : %.2f' % r2_score(y_test, y_LRM))
plt.hist('Height', data=fifa, color='blue') 

plt.style.use('ggplot')

plt.title("Distribution of Height")  

plt.xlabel("Height")  

plt.ylabel("Count") 

plt.show()
plt.hist('Weight', data=fifa, color='pink') 

plt.style.use('ggplot')

plt.title("Distribution of Weight") #Assign title 

plt.xlabel("Weight") #Assign x label 

plt.ylabel("Count") #Assign y label

plt.show()
X = fifa[['Height']]

y = fifa[['Weight']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=44, shuffle =True)
#Applying Linear Regression Model 



LinearRegressionModel = LinearRegression(fit_intercept=True, normalize=True,copy_X=True,n_jobs=-1)

LinearRegressionModel.fit(X_train, y_train)



#Calculating Details

print('Linear Regression Train Score is : ' , LinearRegressionModel.score(X_train, y_train))

print('Linear Regression Test Score is : ' , LinearRegressionModel.score(X_test, y_test))

print('Linear Regression Coef is : ' , LinearRegressionModel.coef_)

print('Linear Regression intercept is : ' , LinearRegressionModel.intercept_)



#Calculating Prediction

y_LRM = LinearRegressionModel.predict(X_test)
plt.scatter(X_test, y_test,  color='black')

plt.plot(X_test, y_LRM, color='blue', linewidth=3)

plt.xlabel("Height")

plt.ylabel("Weight")

plt.show()
# The mean squared error

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_LRM))

 

# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % r2_score(y_test, y_LRM))
from sklearn.model_selection import train_test_split

X = fifa[['Value']]

y = fifa[['Overall']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=44, shuffle =True)
#Applying Linear Regression Model 



LinearRegressionModel = LinearRegression(fit_intercept=True, normalize=True,copy_X=True,n_jobs=-1)

LinearRegressionModel.fit(X_train, y_train)



#Calculating Details

print('Linear Regression Train Score is : ' , LinearRegressionModel.score(X_train, y_train))

print('Linear Regression Test Score is : ' , LinearRegressionModel.score(X_test, y_test))

print('Linear Regression Coef is : ' , LinearRegressionModel.coef_)

print('Linear Regression intercept is : ' , LinearRegressionModel.intercept_)



#Calculating Prediction

y_LRM = LinearRegressionModel.predict(X_test)
plt.scatter(X_test, y_test,  color='black')

plt.plot(X_test, y_LRM, color='blue', linewidth=3)

plt.xlabel("Value")

plt.ylabel("Overall")

plt.show()
# The mean squared error

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_LRM))

 

# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % r2_score(y_test, y_LRM))
from sklearn.preprocessing import PolynomialFeatures

from sklearn import linear_model



# Create linear regression object

poly = PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)



X_train = poly.fit_transform(X_train)

X_test = poly.fit_transform(X_test)



model = linear_model.LinearRegression(fit_intercept=True, normalize=True,copy_X=True,n_jobs=-1)

model.fit(X_train, y_train)



print('Linear Regression Train Score is : ', model.score(X_train, y_train))

print('Linear Regression Test Score is : ' , model.score(X_test, y_test))

print('Linear Regression Coef is : ' , LinearRegressionModel.coef_)

print('Linear Regression intercept is : ' , LinearRegressionModel.intercept_)

y_poly_LRM = model.predict(X_test)
# The mean squared error

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_poly_LRM))

 

# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % r2_score(y_test, y_poly_LRM))
#Import Libraries

from sklearn.svm import SVR



SVRModel = SVR(C = 1.0 ,epsilon=0.1,kernel = 'rbf', gamma='auto') # it also can be : linear, poly, rbf, sigmoid, precomputed

SVRModel.fit(X_train, y_train)



#Calculating Details

print('SVRModel Train Score is : ' , SVRModel.score(X_train, y_train))

print('SVRModel Test Score is : ' , SVRModel.score(X_test, y_test))



y_SVR = SVRModel.predict(X_test)
# The mean squared error

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_SVR))

 

# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % r2_score(y_test, y_SVR))
#Import Libraries

from sklearn.ensemble import GradientBoostingRegressor



#Applying Gradient Boosting Regressor Model 



GBRModel = GradientBoostingRegressor(n_estimators=1000,max_depth=8,learning_rate = 0.1 ,random_state=99)

GBRModel.fit(X_train, y_train)



#Calculating Details

print('GBRModel Train Score is : ' , GBRModel.score(X_train, y_train))

print('GBRModel Test Score is : ' , GBRModel.score(X_test, y_test))



#Calculating Prediction

y_GBR = GBRModel.predict(X_test)
# The mean squared error

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_GBR))

 

# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % r2_score(y_test, y_GBR))
X=fifa[['BallControl', 'ShortPassing', 'Dribbling', 'Crossing', 'Curve',

                                            'LongPassing', 'LongShots', 'ShotPower', 'Positioning', 'FKAccuracy', 'Stamina',

                                            'Volleys', 'Skill Moves', 'Vision', 'Composure', 'Penalties', 'Finishing', 

                                             'Agility']]

y = fifa[['Special']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=44, shuffle =True)
#Applying Linear Regression Model 



LinearRegressionModel = LinearRegression(fit_intercept=True, normalize=True,copy_X=True,n_jobs=-1)

LinearRegressionModel.fit(X_train, y_train)



#Calculating Details

print('Linear Regression Train Score is : ' , LinearRegressionModel.score(X_train, y_train))

print('Linear Regression Test Score is : ' , LinearRegressionModel.score(X_test, y_test))

print('Linear Regression Coef is : ' , LinearRegressionModel.coef_)

print('Linear Regression intercept is : ' , LinearRegressionModel.intercept_)



#Calculating Prediction

y_LRM = LinearRegressionModel.predict(X_test)
# The mean squared error

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_LRM))

 

# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % r2_score(y_test, y_LRM))
# Create linear regression object

poly = PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)



X_train = poly.fit_transform(X_train, y_train)

X_test = poly.fit_transform(X_test, y_test)



model = linear_model.LinearRegression(fit_intercept=True, normalize=True,copy_X=True,n_jobs=-1)

model.fit(X_train, y_train)



print('Linear Regression Train Score is : ', model.score(X_train, y_train))

print('Linear Regression Test Score is : ' , model.score(X_test, y_test))

print('Linear Regression Coef is : ' , LinearRegressionModel.coef_)

print('Linear Regression intercept is : ' , LinearRegressionModel.intercept_)

y_pred = model.predict(X_test)
# The mean squared error

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

 

# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % r2_score(y_test, y_pred))
#Applying Gradient Boosting Regressor Model

GBRModel = GradientBoostingRegressor(n_estimators=1000,max_depth=8,learning_rate = 0.1 ,random_state=99)

GBRModel.fit(X_train, y_train)



#Calculating Details

print('GBRModel Train Score is : ' , GBRModel.score(X_train, y_train))

print('GBRModel Test Score is : ' , GBRModel.score(X_test, y_test))



#Calculating Prediction

y_pred = GBRModel.predict(X_test)
# The mean squared error

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

 

# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % r2_score(y_test, y_pred))
X=fifa[['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 

        'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',

        'Jumping', 'Stamina','Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 

        'Penalties', 'Composure', 'Marking','StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 

        'GKPositioning', 'GKReflexes']]

y=fifa[['Position']]
from sklearn.ensemble import RandomForestClassifier

my_model = RandomForestClassifier(n_estimators=100, random_state=0).fit(X, y)
from eli5.sklearn import PermutationImportance

import eli5

perm = PermutationImportance(my_model,n_iter=2).fit(X, y)

eli5.show_weights(perm, feature_names = X.columns.tolist())
from lightgbm import LGBMClassifier

lgbc=LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,

            reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)

lgbc.fit(X,y)
from sklearn.metrics import accuracy_score

#define a score function. In this case I use accuracy

def score(X, y):

    y_pred = lgbc.predict(X)

    return accuracy_score(y, y_pred)
from eli5.permutation_importance import get_score_importances

# This function takes only numpy arrays as inputs

base_score, score_decreases = get_score_importances(score, np.array(X), y)

feature_importances = np.mean(score_decreases, axis=0)
feature_importance_dict = {}

for i, feature_name in enumerate(X.columns):

    feature_importance_dict[feature_name]=feature_importances[i]

print(dict(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)[:5]))