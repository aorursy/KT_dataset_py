import pandas as pd
#Column definitions:

#R_ and B_ prefix signifies red and blue corner fighter stats respectively

#_opp_ containing columns is the average of damage done by the opponent on the fighter

#KD is number of knockdowns

#SIG_STR is no. of significant strikes 'landed of attempted'

#SIG_STR_pct is significant strikes percentage

#TOTAL_STR is total strikes 'landed of attempted'

#TD is no. of takedowns

#TD_pct is takedown percentages

#SUB_ATT is no. of submission attempts

#PASS is no. times the guard was passed?

#REV is the no. of Reversals landed

#HEAD is no. of significant strinks to the head 'landed of attempted'

#BODY is no. of significant strikes to the body 'landed of attempted'

#CLINCH is no. of significant strikes in the clinch 'landed of attempted'

#GROUND is no. of significant strikes on the ground 'landed of attempted'

#win_by is method of win

#last_round is last round of the fight (ex. if it was a KO in 1st, then this will be 1)

#last_round_time is when the fight ended in the last round

#Format is the format of the fight (3 rounds, 5 rounds etc.)

#Referee is the name of the Ref

#date is the date of the fight

#location is the location in which the event took place

#Fight_type is which weight class and whether it's a title bout or not

#Winner is the winner of the fight

#Stance is the stance of the fighter (orthodox, southpaw, etc.)

#Height_cms is the height in centimeter

#Reach_cms is the reach of the fighter (arm span) in centimeter

#Weight_lbs is the weight of the fighter in pounds (lbs)

#age is the age of the fighter

#title_bout Boolean value of whether it is title fight or not

#weight_class is which weight class the fight is in (Bantamweight, heavyweight, Women's flyweight, etc.)

#no_of_rounds is the number of rounds the fight was scheduled for

#current_lose_streak is the count of current concurrent losses of the fighter

#current_win_streak is the count of current concurrent wins of the fighter

#draw is the number of draws in the fighter's ufc career

#wins is the number of wins in the fighter's ufc career

#losses is the number of losses in the fighter's ufc career

#total_rounds_fought is the average of total rounds fought by the fighter

#total_time_fought(seconds) is the count of total time spent fighting in seconds

#total_title_bouts is the total number of title bouts taken part in by the fighter

#win_by_Decision_Majority is the number of wins by majority judges decision in the fighter's ufc career

#win_by_Decision_Split is the number of wins by split judges decision in the fighter's ufc career

#win_by_Decision_Unanimous is the number of wins by unanimous judges decision in the fighter's ufc career

#win_by_KO/TKO is the number of wins by knockout in the fighter's ufc career

#win_by_Submission is the number of wins by submission in the fighter's ufc career

#win_by_TKO_Doctor_Stoppage is the number of wins by doctor stoppage in the fighter's ufc career'''
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv("../input/ufcdata/raw_total_fight_data.csv",sep=";")

data
data.shape
data.head()
data.info()
data.columns
data.iloc[0]
columns = ['R_SIG_STR.', 'B_SIG_STR.', 'R_TOTAL_STR.', 'B_TOTAL_STR.',

       'R_TD', 'B_TD', 'R_HEAD', 'B_HEAD', 'R_BODY','B_BODY', 'R_LEG', 'B_LEG', 

        'R_DISTANCE', 'B_DISTANCE', 'R_CLINCH','B_CLINCH', 'R_GROUND', 'B_GROUND']
attemp = '_att'

landed = '_landed'



for column in columns:

    data[columns].isnull().sum()

    data[column+attemp] = data[column].apply(lambda X: int(X.split('of')[1]))

    data[column+landed] = data[column].apply(lambda X: int(X.split('of')[0]))

    

data.drop(columns, axis=1, inplace=True)
data.iloc[0]
data.shape
pct_columns = ['R_SIG_STR_pct','B_SIG_STR_pct', 'R_TD_pct', 'B_TD_pct']



for column in pct_columns:

    data[column] = data[column].apply(lambda X: float(X.replace('%', ''))/100)
data['Winner']
data['Winner'].isnull().sum()
data['Winner'].fillna('Draw', inplace=True)
def get_renamed_winner(row):

    if row['R_fighter'] == row['Winner']:

        return '1'

    elif row['B_fighter'] == row['Winner']:

        return '2'

    elif row['Winner'] == 'Draw':

        return '0'



data['Winner'] = data[['R_fighter', 'B_fighter', 'Winner']].apply(get_renamed_winner, axis=1)
data['Winner'].value_counts()
data.columns
import matplotlib.pyplot as plt
data[pct_columns].hist()

data['Winner'].value_counts().plot.pie(explode=[0.05,0.05,0.02],shadow=True,autopct='%1.1f%%')

plt.show()

dataset= data.drop(['location','date','Referee','Format','last_round_time','Fight_type','win_by'], axis=1)
dataset
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

encode=dataset[['R_fighter','B_fighter']].apply(enc.fit_transform)

dataset[['R_fighter','B_fighter']] = encode[['R_fighter','B_fighter']] 
from sklearn.model_selection import train_test_split
trainlabel= dataset["Winner"]

traindata = dataset.drop("Winner",axis=1)
X_train,X_test,y_train,y_test = train_test_split(traindata,trainlabel,test_size = 0.25, random_state = 42)
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()
model.fit(X_train,y_train)
Score = model.score(X_test,y_test)

print("Score: %.2f%%" % (Score * 100.0))