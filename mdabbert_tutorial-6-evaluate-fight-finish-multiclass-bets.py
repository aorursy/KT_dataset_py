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
pd.options.mode.chained_assignment = None
#Load the matches that have already occurred 

df = pd.read_csv("/kaggle/input/ultimate-ufc-dataset/ufc-master.csv")



#df["Winner"] = df["Winner"].astype('category')

df = df[(df['Winner'] == 'Blue') | (df['Winner'] == 'Red')]



#Let's fix the date

df['date'] = pd.to_datetime(df['date'])
#Load the complementary-dataset



odds_df = pd.read_csv("/kaggle/input/ufc-fight-night-whittaker-vs-till-exotic-bet-odds/exotic_bet_worksheet.csv")



display(odds_df)
#This function is to be used with lambda

#DQs seem to generally count in the KO/TKO category.  So that's where I will put them



def return_finish_type(winner, finish):

    #print(winner, finish)

    #Why overcomplicate things?  We can just use a few if statements

    if winner == 'Red':

        #print("HI")

        if finish in ['U-DEC', 'S-DEC', 'M-DEC']:

            return ('Red - DEC')

        if finish in ['SUB']:

            return('Red - SUB')

        if finish in ['KO/TKO', 'DQ']:

            return('Red - KO/TKO')

    if winner == 'Blue':

        if finish in ['U-DEC', 'S-DEC', 'M-DEC']:

            return ('Blue - DEC')

        if finish in ['SUB']:

            return('Blue - SUB')

        if finish in ['KO/TKO', 'DQ']:

            return('Blue - KO/TKO')

        

    #Test for NaN

    if finish != finish:

        return('')

    

    if finish == 'Overturned':

        return('')

    

    

    return ('error')

    #
#This calls for the power of lambda!

df['finish_type'] = df.apply(lambda x: return_finish_type(x['Winner'], x['finish']), axis=1)



mask = df['finish_type'] == 'error'



display(df[['Winner', 'finish', 'finish_type']][mask])



#Let's remove the blank finish_types

mask = df['finish_type'] != ''

df = df[mask]

display(df[['Winner', 'finish', 'finish_type']])
from sklearn import preprocessing #Used for LabelEncoder



le = preprocessing.LabelEncoder()



le.fit(df['finish_type'])



display(le.classes_)



df['label'] = le.transform(df['finish_type'])



display(df)



#OK looks good!
#Create a label df:

label_df = df['label']





df_train = df[12:]

label_train = label_df[12:]



df_test = df[:12]

label_test = label_df[:12]



print(len(df_test))

print(len(label_test))



print(len(df_train))

print(len(label_train))
display(df_test)
from sklearn.naive_bayes import GaussianNB



my_model = GaussianNB()



my_features = ['R_Reach_cms', 'total_round_dif', 'R_Height_cms', 'R_avg_SIG_STR_pct', 'B_age', 'R_longest_win_streak', 'lose_streak_dif', 'ko_dif', 'R_win_by_Decision_Majority', 'longest_win_streak_dif', 'avg_sub_att_dif', 'R_Weight_lbs', 'sig_str_dif', 'B_Height_cms', 'B_avg_SUB_ATT', 'R_win_by_TKO_Doctor_Stoppage', 'B_draw', 'avg_td_dif', 'R_win_by_Decision_Split', 'age_dif', 'B_current_win_streak', 'R_odds']



#Categorize strings.  Remove nulls... 

df_train = df_train[my_features].copy()

df_test = df_test[my_features].copy()

df_train = df_train.dropna()

df_test = df_test.dropna()



df_train = pd.get_dummies(df_train)

df_test = pd.get_dummies(df_test)

df_train, df_test = df_train.align(df_test, join='left', axis=1)    #Ensures both sets are dummified the same

df_test = df_test.fillna(0)



label_train = label_train [label_train.index.isin(df_train.index)]

label_test = label_test[label_test.index.isin(df_test.index)]



my_model.fit(df_train, label_train) 

pred = my_model.predict(df_test) 
display(pred)

display(le.classes_)
from sklearn.metrics import plot_confusion_matrix

import matplotlib.pyplot as plt



disp = plot_confusion_matrix(my_model, df_test, label_test,

                                 labels=[0,1,2,3,4,5],

                                 display_labels=le.classes_,

                                 cmap=plt.cm.Blues)

plt.xticks(rotation=60)

probs = my_model.predict_proba(df_test)
np.set_printoptions(suppress=True)

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

print(probs)

#These are the different columns above

for c in le.classes_:

    print(c)
odds_df_filtered = odds_df[le.classes_]
probs
odds = odds_df_filtered.values #Let's change the df into an array to make it easier to work with
display(odds) #Here are the odds of the different finishes


#A helper function to determine if a bet is good.  If this number is positive we have a profitable bet.

def get_bet_ev(odds, prob):

    if odds>0:

        return ((odds * prob) - (100 * (1-prob)) )

    else:

        return ((100 / abs(odds))*100*prob - (100 * (1-prob)))




make_bet_array = []



for i in range(len(probs)):

    bet_row = []

    for j in range(len(probs[0])):

        temp_ev = get_bet_ev(odds[i][j], probs[i][j])

        if temp_ev > 0:

            bet_row.append(1)

            print(f"RED {odds_df.iloc[i,0]} vs. BLUE {odds_df.iloc[i,1]}",

                f"\nBET EV of {temp_ev} on  {le.classes_[j]}\n",

                  f"Odds: {odds[i][j]}\n"

                  )

        else:

            bet_row.append(0)

    make_bet_array.append(bet_row)
make_bet_array #These are our bets.  We just need to translate them...
#Go through the bet array and determine winners and losers.  Additionally we will keep track of how much we are winning and losing



bet_total = 0

bet_count = 0

for i in range(len(make_bet_array)):

    for j in range(len(make_bet_array[0])):

        if(make_bet_array[i][j] == 1):

            #We have a bet...

            print(f"RED {odds_df.iloc[i,0]} vs. BLUE {odds_df.iloc[i,1]}",

                f"\n{le.classes_[j]}\n",

                  f"Odds: {odds[i][j]}"

                  )

            if label_test[i] == j: #We have a winner:

                print('WINNER\n')

                bet_total = bet_total + odds[i][j]

                bet_count = bet_count + 1

            else:

                print('LOSER\n')

                bet_total = bet_total - 100

                bet_count = bet_count + 1
print(f"We have a profit of ${bet_total} on {bet_count} $100 bets")