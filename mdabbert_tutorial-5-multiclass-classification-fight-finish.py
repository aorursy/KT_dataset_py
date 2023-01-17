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

import math
#Load the matches that have already occurred 

df = pd.read_csv("/kaggle/input/ultimate-ufc-dataset/ufc-master.csv")



#df["Winner"] = df["Winner"].astype('category')

df = df[(df['Winner'] == 'Blue') | (df['Winner'] == 'Red')]



#Let's fix the date

df['date'] = pd.to_datetime(df['date'])

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





df_train = df[200:]

label_train = label_df[200:]



df_test = df[:200]

label_test = label_df[:200]



print(len(df_test))

print(len(label_test))



print(len(df_train))

print(len(label_train))
#Pick some features and a model

from sklearn.tree import DecisionTreeClassifier

#Pick a model

my_model = DecisionTreeClassifier(max_depth=10)



#Pick some features

#There isn't really much logic to this collection.  Don't use this in real life to make bets

my_features = ['R_odds', 'B_odds','weight_class', 'gender',

       'B_avg_SIG_STR_landed', 'B_avg_SIG_STR_pct', 'B_avg_SUB_ATT',

       'B_avg_TD_landed', 'B_avg_TD_pct', 'B_longest_win_streak', 'B_losses',

       'B_total_rounds_fought', 'B_total_title_bouts',

       'B_win_by_Decision_Majority', 'B_win_by_Decision_Split',

       'B_win_by_Decision_Unanimous', 'B_win_by_KO/TKO', 'B_win_by_Submission',

       'B_win_by_TKO_Doctor_Stoppage', 'B_wins', 'B_Stance', 'B_Height_cms',

       'B_Reach_cms', 'B_Weight_lbs', 'R_current_lose_streak',

       'R_current_win_streak', 'R_draw', 'R_avg_SIG_STR_landed',

       'R_avg_SIG_STR_pct', 'R_avg_SUB_ATT', 'R_avg_TD_landed', 'R_avg_TD_pct',

       'R_longest_win_streak', 'R_losses', 'R_total_rounds_fought',

       'R_total_title_bouts', 'R_win_by_Decision_Majority',

       'R_win_by_Decision_Split', 'R_win_by_Decision_Unanimous',

       'R_win_by_KO/TKO', 'R_win_by_Submission',

       'R_win_by_TKO_Doctor_Stoppage', 'R_wins', 'R_Stance', 'R_Height_cms',

       'R_Reach_cms', 'R_Weight_lbs', 'R_age', 'B_age', 'lose_streak_dif',

       'win_streak_dif', 'longest_win_streak_dif', 'win_dif', 'loss_dif',

       'total_round_dif', 'total_title_bout_dif', 'ko_dif', 'sub_dif',

       'height_dif', 'reach_dif', 'age_dif', 'sig_str_dif', 'avg_sub_att_dif',

       'avg_td_dif', 'empty_arena']





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

  

 
from sklearn.metrics import plot_confusion_matrix

import matplotlib.pyplot as plt



disp = plot_confusion_matrix(my_model, df_test, label_test,

                                 display_labels=le.classes_,

                                 cmap=plt.cm.Blues,

                                 normalize=None)

plt.xticks(rotation=60)
