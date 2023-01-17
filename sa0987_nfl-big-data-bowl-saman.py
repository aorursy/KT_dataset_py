import os

# __print__ = print

# def print(string):

#     os.system(f'echo \"{string}\"')

#     __print__(string)



fname = "/kaggle/input/nfl-big-data-bowl-2020/train.csv"



# original public Score .01422
#from kaggle.competitions import nflrush

import pandas as pd

import datetime

# Load scikit's random forest classifier library

from sklearn.ensemble import RandomForestClassifier

import re



# Load pandas

import pandas as pd



# Load numpy

import numpy as np







from dateutil.relativedelta import relativedelta







def lookup(s):

    """

    This is an extremely fast approach to datetime parsing.

    For large data, the same dates are often repeated. Rather than

    re-parse these, we store all unique dates, parse them, and

    use a lookup to convert all dates.

    """

    dates = {date:pd.to_datetime(date) for date in s.unique()}

    return s.map(dates)



def ageNow(d):

       

    now = datetime.datetime.now()

    difference = relativedelta(now, d)

    return(difference.years)

    


# Load scikit's random forest classifier library

from sklearn.ensemble import RandomForestClassifier



# Load pandas

import pandas as pd



# Load numpy

import numpy as np





    

clf = []



def randomForest(dfIn,cols):

    global clf

    # RANDOM FOREST



    # Load the library with the iris dataset

   

    # Set random seed

    np.random.seed(0)



    print("Starting 1")

    # Create a random forest Classifier. By convention, clf means 'Classifier'

    clf = RandomForestClassifier(n_jobs=1, random_state=0,n_estimators=10)



    # Train the Classifier to take the training features and learn how they relate

    # to the training y (the species)





    training_df = dfIn[cols][::22]



    y=training_df['Yards']

    X=training_df[[i for i in training_df.columns if "Yards" not in i]]







    clf.fit(X, y)

    print("Trained")









def modifyXY(dfx):

    dfIn = dfx

    print('Doing ModifyXY')

    # add features for us later 

    dfIn['XM'] = dfIn['X'] - 10 # normalize this to absolute yards

    dfIn['XS'] = dfIn['AbsoluteYardLine'] # this is to use for later and it is easier to understand

    T = 0.5 # X Y in .5 seconds

    print('Doing ModifyXY ',T)

    dfIn['S05'] = dfIn['S'] +  dfIn['A'] * T 

    dfIn['X05'] = dfIn[['X','S','A','Orientation']].apply(lambda x: x['X'] + np.sin(x['Orientation'] * (np.pi/180)) + (x['S'] * T +  x['A']  * x['A'] * T / 2.0),axis=1)

    dfIn['X05'] = dfIn[['Y','S','A','Orientation']].apply(lambda x: x['Y'] + np.cos(x['Orientation'] * (np.pi/180)) + (x['S'] * T +  x['A']  * x['A'] * T / 2.0),axis=1)



    T = 1 # X Y in 1.0 seconds

    print('Doing ModifyXY',T)

    dfIn['S10'] = dfIn['S'] +  dfIn['A'] * T 

    dfIn['Y10'] = dfIn[['X','S','A','Orientation']].apply(lambda x: x['X'] + np.sin(x['Orientation'] * (np.pi/180)) + (x['S'] * T +  x['A']  * x['A'] * T / 2.0),axis=1)

    dfIn['Y10'] = dfIn[['Y','S','A','Orientation']].apply(lambda x: x['Y'] + np.cos(x['Orientation'] * (np.pi/180)) + (x['S'] * T +  x['A']  * x['A'] * T / 2.0),axis=1)

    print('Done ModifyXY')

    

    print(dfIn.columns)

    dfOut = dfIn

    return dfOut



def isRequired(strIn):

    global flat_list,isTraining

    str2 = ".*"+strIn+".*"

    if (isTraining):

        print("--> isRequired? ",str2,strIn,flat_list)



    r = re.compile(str2)

    l = list(filter(r.match, flat_list))

    #print(l,len(l))



    rc = len(l) 

    if rc > 0: 

        if (isTraining):

            print("Field Required = ",strIn)

    return rc 

        

    

def addFeatures(dfy):

    global flatlist,isTraining



    

    dfIn = dfy

    # correct some typos in the team names 

    map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}

    for abb in dfIn['PossessionTeam'].unique():

        map_abbr[abb] = abb



    dfIn['PossessionTeam'] = dfIn['PossessionTeam'].map(map_abbr)

    dfIn['HomeTeamAbbr'] = dfIn['HomeTeamAbbr'].map(map_abbr)

    dfIn['VisitorTeamAbbr'] = dfIn['VisitorTeamAbbr'].map(map_abbr)

    

   # print('Adding features')

    # calculate the correct Absolute Yards to go to TD 

    dfIn['AbsoluteYardLine'] = dfIn[['YardLine','PlayDirection','HomeTeamAbbr','PossessionTeam','FieldPosition']].apply(lambda x: int((100-x['YardLine'])) if x['PossessionTeam'] == x['FieldPosition'] else int(int(x['YardLine'])),axis=1)

    dfIn['PlayerIsRusher']  = dfIn['NflIdRusher'] == dfIn['NflId']

    

    # ADD MORE FEATURE CALS HERE

    if isRequired("XS"):

        dfIn = modifyXY(dfIn)

    

    if isRequired("PlayerWeight"):

       # print('Adding features PlayerWeight')

        # calculate the players weights 

        dfIn['PlayerWeightSumDefense'] = dfIn['PlayerWeight']

        dfIn['PlayerWeightSumOffense'] = dfIn['PlayerWeight']



        dfIn.loc[(dfIn['PossessionTeam'] == dfIn['VisitorTeamAbbr']) & ( dfIn['Team'] == 'away'), 'PlayerWeightSumDefense'] = 0

        dfIn.loc[(dfIn['PossessionTeam'] == dfIn['HomeTeamAbbr']) & ( dfIn['Team'] == 'home'), 'PlayerWeightSumDefense'] = 0

        dfIn.loc[(dfIn['PossessionTeam'] != dfIn['VisitorTeamAbbr']) & ( dfIn['Team'] == 'away'), 'PlayerWeightSumOffense'] = 0

        dfIn.loc[(dfIn['PossessionTeam'] != dfIn['HomeTeamAbbr']) & ( dfIn['Team'] == 'home'), 'PlayerWeightSumOffense'] = 0





        w = dfIn.groupby('PlayId')['PlayerWeightSumDefense','PlayerWeightSumOffense'].sum()

        w.rename(columns={'PlayerWeightSumDefense' : 'PlayerWeightSumDefenseSum', 'PlayerWeightSumOffense' : 'PlayerWeightSumOffenseSum'},inplace=True)

        #print("1",w.columns)

        w = w.reset_index()

        #print("2",w.columns)

        dfIn.drop(['PlayerWeightSumDefense','PlayerWeightSumOffense'],axis=1,inplace=True)

        dfIn = pd.merge(dfIn,w,how='left',left_on='PlayId',right_on='PlayId')

        dfIn['PlayerWeightSumDiff'] = dfIn['PlayerWeightSumDefenseSum'] - dfIn['PlayerWeightSumOffenseSum']



    dfIn2 = dfIn

    # Calculate Player Age 

    # ensure the Player birth day is date

    # calculate Age 

    if isRequired("PlayerBirthDate") : dfIn2['PlayerBirthDate'] =  lookup(dfIn2['PlayerBirthDate'])

    if isRequired("PlayerAge") : dfIn2['PlayerAge'] = dfIn2['PlayerBirthDate'].apply(lambda x: ageNow(x))

    if isRequired("PlayerYearsInNFL") : dfIn2['PlayerYearsInNFL'] = dfIn2['PlayerAge'] - 22 # from college as 22 year old

 



    ## MORE STUFF 

    dfIn2['GameClock2'] = dfIn2['GameClock'].apply(lambda x: int(x[0:2])+int(x[3:5])/60)

    dfIn2['AbsoluteGameClock'] = dfIn2[['GameClock','Quarter']].apply(lambda x: int(x['GameClock'][0:2])+float(x['GameClock'][3:5])/60+((4-x['Quarter'])*15),axis=1)

    dfIn2['AbsoluteGameClockRemaining'] = 60 - dfIn2['AbsoluteGameClock'] 

    

    

    # get the ouptut ready to go 

    dfOut = dfIn2

    return dfOut
def predict_using_my_model_RandomForest(dfIn,sample_prediction_df):

    global clfs # all the CLFs

    global dfME

    global colsSet # array of columns, one for each CLF



    test_dfIn = dfIn[dfIn['NflIdRusher'] == dfIn['NflId']].copy()

    test_dfIn = addFeatures(test_dfIn)

    df3 = pd.DataFrame()

    c = 0 

    #print("colsSet = ",colsSet)

#     dfABS = pd.DataFrame([[0] * 199])

#     dfABS.columns = ["Yards"+str(x) for x in range (-99,100)]

    AbsoluteYardLine = test_dfIn['AbsoluteYardLine'].values[0]

    minYards =  - AbsoluteYardLine 

    maxYards = 100 - AbsoluteYardLine 

    #dfABS.iloc[ 0 , minYards:maxYards] = 1

    #print(minYards,maxYards,AbsoluteYardLine)

    for cols in colsSet:

        clf = clfs[c]

        c = c + 1

        y_df = pd.DataFrame(list(zip(clf.classes_,[x for x in clf.predict_proba(test_dfIn[cols])[0]])))

        i = range(-99,100)

        missingyards = set(i).difference(sorted(y))

        #print( missingyards)

        y_df = y_df.append(pd.DataFrame(list(zip(missingyards,[0]*len(missingyards)))))

        y_df.columns = ['Yards','probablity']

        y_df = y_df.sort_values('Yards')



        df1 =  pd.DataFrame(y_df.set_index('Yards').reset_index()['probablity']).T

        df1.columns = ["Yards"+str(x) for x in range (-99,100)]

#         for i in range (-99,100): 

#            # print(type(i),type(minYards),type(maxYards))

#             if i < minYards: df1["Yards"+str(i)] = 0.0

#             if i > maxYards: df1["Yards"+str(i)] = 0.0

        df2 = df1.cumsum(axis=1)

        df3 = pd.concat([df3,df2])

        

   #print(df3.T) 

    df4 = pd.DataFrame(df3.sum()).T

   # print(df4.T)

    sample_prediction_df = df4.div(df4.max(axis=1), axis=0)

   # print(sample_prediction_df.T)

   # print(sample_prediction_df['Yards99'].values[0],sample_prediction_df['Yards99'].values[0] == 1.0)

    assert sample_prediction_df['Yards99'].values[0] == 1.0, "Values is not 1.0"



    return sample_prediction_df





isTraining = True 

os.system(f'echo Training')

clf = 0 





# colsSet = [

#             #['Down','A','S','PlayerWeightSumDiff','JerseyNumber','AbsoluteYardLine','Quarter']

# #             ['A','S']

# #             ,['A','S','AbsoluteYardLine']

# #             ,['A','S','JerseyNumber']

# #             ,['Down','AbsoluteYardLine']

#             ['Down','NflId','AbsoluteGameClockRemaining']

#             ]





colsSet = [

            #['Down','A','S','PlayerWeightSumDiff','JerseyNumber','AbsoluteYardLine','Quarter']

#             ['A','S']

#             ,['A','S','AbsoluteYardLine']

#             ,['A','S','JerseyNumber']

#             ,['Down','AbsoluteYardLine']

#           ['Down','NflId','AbsoluteGameClockRemaining']

            ['Down','JerseyNumber'] # attempt #2 

           ]





i = 0 





flat_list = list(set([item for sublist in colsSet for item in sublist]))



# RANDOM FOREST





# Set random seed

np.random.seed(0)



print("Starting 1")

train = True

if train == True:

    #filename = '/Users/sa0987/Downloads/nfl-big-data-bowl-2020/train.csv'

    filename = fname

    dfIn = pd.read_csv(filename, low_memory=False)

    train_df  = dfIn

    test_df = dfIn



    map_abbr = {'ARI': 'ARZ', 'BAL': 'BLT', 'CLE': 'CLV', 'HOU': 'HST'}

    for abb in train_df['PossessionTeam'].unique():

        map_abbr[abb] = abb



    train_df['PossessionTeam'] = train_df['PossessionTeam'].map(map_abbr)

    train_df['HomeTeamAbbr'] = train_df['HomeTeamAbbr'].map(map_abbr)

    train_df['VisitorTeamAbbr'] = train_df['VisitorTeamAbbr'].map(map_abbr)

   # train_df['AbsoluteYardLine'] = train_df[['YardLine','PlayDirection','HomeTeamAbbr','PossessionTeam','FieldPosition']].apply(lambda x: int((100-x['YardLine'])) if x['PossessionTeam'] == x['FieldPosition'] else int(int(x['YardLine'])),axis=1)



#train_df= pd.read_csv('train.csv')

train_df = addFeatures(train_df)

isTraining = False

print('Done adding to train_df')







i = 0 

clfs = []

for cols in colsSet:

    clf = RandomForestClassifier(n_jobs=1, random_state=0,n_estimators=10)



    Y = ['Yards']

    training_df = train_df[train_df['PlayerIsRusher'] == True][cols+Y].copy()

    #training_df = train_df[['JerseyNumber','Down','Yards']][::22]



    y=training_df['Yards']

    X=training_df[[i for i in training_df.columns if "Yards" not in i]]





    # Create a random forest Classifier. By convention, clf means 'Classifier'

    clf = RandomForestClassifier(n_jobs=2, random_state=0,n_estimators=100)



    print("Training")

    os.system(f'echo training')

    clf.fit(X, y)

    print("Trained")

    os.system(f'echo trained')

    clfs.append(clf)







from kaggle.competitions import nflrush



# # You can only call make_env() once, so don't lose it!

env = nflrush.make_env()

i = 0 

# # You can only iterate through a result from `env.iter_test()` once

# # so be careful not to lose it once you start iterating.



for (test_df, sample_prediction_df) in env.iter_test():

   # predictions_df = make_my_predictions(test_df, sample_prediction_df)

    #print(test_df[['JerseyNumber','Down']][0:2])



    #dfPredict = predict_using_my_model_Jersey_Down(test_df,sample_prediction_df)

   # test_df['AbsoluteYardLine'] = test_df[['YardLine','PlayDirection','HomeTeamAbbr','PossessionTeam','FieldPosition']].apply(lambda x: int((100-x['YardLine'])/10) if x['PossessionTeam'] != x['FieldPosition'] else int(int(x['YardLine'])/10),axis=1)



    dfPredict = predict_using_my_model_RandomForest(test_df,sample_prediction_df)

    

    #Together 

    #dfPredict = predict_using_my_model_FieldPostionDown(test_df,sample_prediction_df)

    #dfPredict = predict_using_my_model_FieldPostionBox(test_df,sample_prediction_df)



    os.system(f'echo {i} ')

    i = i + 1 







    



    



    

    env.predict(dfPredict)



env.write_submission_file()



env.write_submission_file()
# We've got a submission file!

import os

print([filename for filename in os.listdir('/kaggle/working') if '.csv' in filename])