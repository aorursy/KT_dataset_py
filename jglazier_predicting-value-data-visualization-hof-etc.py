#Import libraries



import numpy as np 

import pandas as pd 



#Read data, count rows

data = pd.read_csv('../input/nfl_draft.csv')

rows = len(data.index)

#Lets view the top of the data to see the format



data.head(1)
#Now lets view the bottom to ensure nothing bizarre happens in the data



data.tail(1)
#We see sacks are turned into Colleges at some point



#let's move sacks over to colleges where applicable with a function





#Our function will switch sacks with University, where applicable

def switch_sacks(df): 

    #Copy the dataframe

    df = df.copy(deep=True)

    #Get the number of rows

    nrows = len(df.index) 

    

    #Create a test of if a string is a number

    #We'll use this to make sure that the value for sacks is not a number

    def is_number(s):

        try:

            float(s)

            return True

        except ValueError:

            return False

    # Now we make sure that both the College isn't labeled and he has no sacks

    # A couple players didnt go to college, so its important to have both

    for i in range(nrows):

        if isinstance(df.loc[i]["College/Univ"], str) == False and is_number(df.loc[i]["Sk"]) == False: 

            df.set_value(i, "College/Univ", df.loc[i]["Sk"])

            df.set_value(i, "Sk", np.nan)

            

    return df
#Lets make this data clean and check all of the colleges



newdata = switch_sacks(data)

#Lets add a column for conference

#We'll only use the Power 5



def add_cfb_conf(df): 

    conf =[]

    nrows = len(df.index) 

    

    pac12 = ("Stanford", "California", "Arizona St.", "Arizona", "Washington", 

          "Washington St.", "Oregon", "Oregon St.", "USC", "UCLA", "Utah", "Colorado")

    

    big12 = ("Oklahoma", "Oklahoma St.", "TCU", "Baylor", "Iowa St.", "Texas", "Kansas", 

            "Kansas St.", "West Virginia", "Texas Tech")

    

    b1g = ("Northwestern", "Michigan", "Michigan St.", "Iowa", "Ohio St.", "Purdue", 

          "Indiana", "Rutgers", "Illinois", "Minnesota", "Penn St.", "Nebraska", "Maryland", 

          "Wisconsin")

    

    acc = ("Florida St.", "Syracuse", "Miami", "North Carolina", "North Carolina St.", 

          "Duke", "Virginia", "Virginia Tech", "Boston College", "Clemson", "Wake Forest",

          "Pittsburgh", "Louisville", "Louisville", "Georgia Tech")

    

    sec = ("Alabama", "Georgia", "Vanderbilt", "Kentucky", "Florida", "Missouri", 

          "Mississippi", "Mississippi St.", "Texas A&M", "Louisiana St.", "Arkansas", 

          "Auburn", "South Carolina", "Tennessee")

    

    for i in range(nrows): 

        if df.loc[i]["College/Univ"] in pac12: 

            conf.append("Pac 12")

            

        elif df.loc[i]["College/Univ"] in big12:

            conf.append("Big 12")

            

        elif df.loc[i]["College/Univ"] in b1g:

            conf.append("Big 10")

            

        elif df.loc[i]["College/Univ"] in acc:

            conf.append("ACC")

            

        elif df.loc[i]["College/Univ"] in sec: 

            conf.append("SEC")

        

        else: 

            conf.append("Not Power 5")

    return conf
conf = add_cfb_conf(newdata)
newdata['CFB_Conference'] = conf
newdata
import seaborn as sns

sns.set(style="whitegrid", color_codes=True)
sns.countplot(x = "CFB_Conference", data = newdata, palette = "Greens_d")
sns.violinplot(x = "CFB_Conference", y="Rnd", data = newdata, palette = "husl")
newdata.loc[newdata['Rnd'] == 8].head(1)
sns.violinplot(x = "CFB_Conference", y="Rnd", data = newdata.ix[:5957][:], palette = "husl")
Rd1 = newdata.loc[newdata['Rnd'] == 1]
Rd1
import matplotlib.pyplot as plt



Positions = Rd1.groupby(['Pos']).size()



Positions.index
labels = Positions.index

sizes = Positions

explode = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0)  # only "explode" the QBs



fig1, ax1 = plt.subplots(figsize=(20,10))

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()
Top10 = Rd1.loc[Rd1['Pick'] < 11]



Positions = Top10.groupby(['Pos']).size()



labels = Positions.index

sizes = Positions

labels



explode = (0, 0, 0, 0, 0, 0, 0.2, 0, 0, 0, 0)  # only "explode" the QBs



fig1, ax1 = plt.subplots(figsize=(20,10))

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.show()



newdata.groupby(['Tm']).size()

def add_div_conf(df):

    div = []

    conf = []

    nrows = len(df.index)

    

    afcsouth = ['IND', 'JAX', 'TEN', 'HOU']

    afcnorth = ['PIT', 'BAL', 'CIN', 'CLE']

    # Must account for the moves of the raiders

    afcwest = ['RAI', 'OAK', 'SDG', 'DEN', 'KAN']

    afceast = ['BUF', 'MIA', 'NWE', 'NYJ']

    

    nfcsouth = ['CAR', 'NOR', 'TAM', 'ATL']

    nfceast = ['WAS', 'PHI', 'NYG', 'DAL']

    # Must account for moves of cardinals and rams

    nfcwest = ['SFO', 'STL', 'RAM', 'PHO', 'ARI', 'SEA']

    nfcnorth = ['CHI', 'GNB', 'MIN', 'DET']

    

    

    for i in range(nrows):

        

        

        if df.loc[i]['Tm'] in afcsouth: 

            div.append('South')

            conf.append('AFC')

        

        elif df.loc[i]['Tm'] in nfcsouth: 

            div.append('South')

            conf.append('NFC')

            

        elif df.loc[i]['Tm'] in afcwest: 

            div.append('West')

            conf.append('AFC')

        

        elif df.loc[i]['Tm'] in nfcwest: 

            div.append('West')

            conf.append('NFC')

            

        elif df.loc[i]['Tm'] in afceast: 

            div.append('East')

            conf.append('AFC')

            

        elif df.loc[i]['Tm'] in nfceast: 

            div.append('East')

            conf.append('NFC')

        

        elif df.loc[i]['Tm'] in afcnorth: 

            div.append('North')

            conf.append('AFC')

        

        elif df.loc[i]['Tm'] in nfcnorth: 

            div.append('North')

            conf.append('NFC')

            

        else: 

            div.append(float('nan'))

            conf.append(float('nan'))

        

    return div, conf



div, conf = add_div_conf(newdata)
newdata['Conf'] = conf

newdata['Div'] = div
newdata
from matplotlib import pyplot



fig, ax = pyplot.subplots(figsize=(20,10))

sns.countplot(y="Div", hue="CFB_Conference", data=newdata.loc[newdata['CFB_Conference']!= "Not Power 5"], palette="hls");
#Let's fix up a missing year of rounds



newdata['Rnd'].isnull().sum().sum()

def fix_rds(df): 

    nrow = len(df.index)

    

    for i in range(nrow): 

        if df.loc[i]['Year'] == 1993: 

            if df.loc[i]['Pick'] >= 1 and df.loc[i]['Pick'] <= 28: 

                df.set_value(i, 'Rnd', 1)

            elif df.loc[i]['Pick'] >= 29 and df.loc[i]['Pick'] <= 56: 

                df.set_value(i, 'Rnd', 2)

            elif df.loc[i]['Pick'] >= 57 and df.loc[i]['Pick'] <= 84: 

                df.set_value(i, 'Rnd', 3)

                

            elif df.loc[i]['Pick'] >= 85 and df.loc[i]['Pick'] <= 112: 

                df.set_value(i, 'Rnd', 4)

                

            elif df.loc[i]['Pick'] >= 113 and df.loc[i]['Pick'] <= 140: 

                df.set_value(i, 'Rnd', 5)

            

            elif df.loc[i]['Pick'] >= 141 and df.loc[i]['Pick'] <= 168: 

                df.set_value(i, 'Rnd', 6)

            

            elif df.loc[i]['Pick'] >= 169 and df.loc[i]['Pick'] <= 196: 

                df.set_value(i, 'Rnd', 7)

            

            elif df.loc[i]['Pick'] >= 197 and df.loc[i]['Pick'] <= 224: 

                df.set_value(i, 'Rnd', 8)

    return df



newdata = fix_rds(newdata)
# Lets check in all of our pertinent categories for missing values and replace them



newdata['Rnd'].isnull().sum().sum()
newdata['Pick'].isnull().sum().sum()
newdata['Tm'].isnull().sum().sum()
newdata['Pos'].isnull().sum().sum()
newdata['Position Standard'].isnull().sum().sum()
newdata['Age'].isnull().sum().sum()



#Way too many missing values, let's just avoid this section entirely. Realistically, the age 

#distribution won't be of utmost concern anyways. 
newdata['College/Univ'].isnull().sum().sum()



#Lets go ahead and replace all of these values with None. Perhaps these students didn't attend 

# college, perhaps it was overseas. 
newdata['College/Univ'].fillna("Other", inplace=True)



newdata['College/Univ'].isnull().sum().sum()
newdata['CFB_Conference'].isnull().sum().sum()
newdata['Conf'].isnull().sum().sum()
newdata['Div'].isnull().sum().sum()
from sklearn import preprocessing

from sklearn.model_selection import train_test_split



train, test = train_test_split(newdata, test_size=0.2)





def encode_features(df_train, df_test):

    features = ['Year', 'Pick', 'Rnd', 'College/Univ', 'Tm', 'Div', 'Conf', 'Pos', 

                'Position Standard', 'CFB_Conference']

    df_combined = pd.concat([df_train[features], df_test[features]])

    

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(df_combined[feature])

        df_train[feature] = le.transform(df_train[feature])

        df_test[feature] = le.transform(df_test[feature])

    return df_train, df_test

    

data_train, data_test = encode_features(train, test)

data_train.head()



data_train['G'].isnull().sum()
data_train['G'].fillna(0, inplace=True)

data_train['CarAV'].fillna(0, inplace = True)



data_test['G'].fillna(0, inplace=True)

data_test['CarAV'].fillna(0, inplace = True)
from sklearn.model_selection import train_test_split



X_all = data_train.drop(['Player_Id', 'Player', 'First4AV', 'Age', 'To', 'AP1',

                        'PB', 'St', 'CarAV', 'DrAV', 'G', 'Cmp', 'Pass_Att', 

                        'Pass_TD', 'Pass_Int', 'Rush_Att', 'Rush_Yds', 'Rush_TDs',

                        'Rec', 'Rec_Yds', 'Rec_Tds', 'Tkl', 'Def_Int', 'Sk',

                        'Pass_Yds', 'Unnamed: 32'], axis=1)

y_all = data_train['CarAV']



num_test = 0.20

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)
y_train.isnull().sum()
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV



# Choose the type of classifier. 

mdl = RandomForestRegressor(n_estimators = 50, criterion = 'mae', max_features = "log2")



# Fit the best algorithm to the data. 

mdl.fit(X_train, y_train)
predictions = mdl.predict(X_test)
from sklearn.metrics import r2_score



r2_score(y_test, predictions)
residuals = predictions - y_test

plt.hist(residuals, bins = 30, range = (-50, 50))



# The distribution is left-skewed, as we'll have a handful of players that perform very well

# This would follow the idea of power law distributions, where many have little success

# And few have much success
from sklearn.cross_validation import KFold

from sklearn.metrics import r2_score



def run_kfold(mdl):

    kf = KFold(6748, n_folds=10)

    outcomes = []

    fold = 0

    for train_index, test_index in kf:

        fold += 1

        X_train, X_test = X_all.values[train_index], X_all.values[test_index]

        y_train, y_test = y_all.values[train_index], y_all.values[test_index]

        mdl.fit(X_train, y_train)

        predictions = mdl.predict(X_test)

        accuracy = r2_score(y_test, predictions)

        outcomes.append(accuracy)

        print("Fold {0} accuracy: {1}".format(fold, accuracy))     

    mean_outcome = np.mean(outcomes)

    print("Mean Accuracy: {0}".format(mean_outcome)) 



run_kfold(mdl)
importance = mdl.feature_importances_
importance
X_train.columns


objects = X_train.columns

y_pos = np.arange(len(objects))
plt.barh(y_pos, importance, align='center', alpha=0.5)



plt.yticks(y_pos, objects)



plt.xlabel('Importance')

plt.title('Feature Importance')

 

plt.show()
X_all = data_train.drop(['Player_Id', 'Player', 'First4AV', 'Age', 'To', 'AP1',

                        'PB', 'St', 'CarAV', 'DrAV', 'G', 'Cmp', 'Pass_Att', 

                        'Pass_TD', 'Pass_Int', 'Rush_Att', 'Rush_Yds', 'Rush_TDs',

                        'Rec', 'Rec_Yds', 'Rec_Tds', 'Tkl', 'Def_Int', 'Sk',

                        'Pass_Yds', 'Unnamed: 32'], axis=1)

y_all = data_train['G']



num_test = 0.20

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=22)
mdl.fit(X_train, y_train)
predictions = mdl.predict(X_test)

r2_score(y_test, predictions)
#You'll see the same left-skew, which follows the power law idea



residuals = predictions - y_test

plt.hist(residuals, bins = 30, range = (-150, 150))

plt.title('Residuals: Games Played')
run_kfold(mdl)
importance1 = mdl.feature_importances_

importance1 - importance
plt.barh(y_pos, importance1, align='center', alpha=0.5)



plt.yticks(y_pos, objects)



plt.xlabel('Importance')

plt.title('Feature Importance: Games Played')

 

plt.show()
newdata.sort_values(['G'], ascending=[False]).head(10)
newdata.sort_values(['CarAV'], ascending=[False]).head(10)

def hof(df): 

    nrow = len(df.index)

    hall = []

    for i in range(nrow):

        if (df['Player'].str[-3:][i] == "HOF"): 

            hall.append(1)

        else: 

            hall.append(0)

    

    return hall

newdata['HOF'] = hof(newdata)
thehall = newdata.loc[newdata['HOF'] == 1]

thehall.head(3)
Rounds = thehall.groupby(['Rnd']).size()



labels = Rounds.index

sizes = Rounds

explode = (0.3, 0, 0, 0, 0, 0)  # only "explode" the QBs



fig1, ax1 = plt.subplots(figsize=(20,10))

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.title('Hall of Famers by round drafted')



plt.show()



Con = thehall.groupby(['CFB_Conference']).size()

labels = Con.index

sizes = Con

explode = (0, 0, 0, 0, 0, 0.3)  # only "explode" the QBs



fig1, ax1 = plt.subplots(figsize=(20,10))

ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',

        shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.title('Hall of Famers by Conference')



plt.show()