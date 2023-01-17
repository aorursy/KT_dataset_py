# Data analysis

import pandas as pd

import random

import scipy.stats as stats



# Plotting

import numpy as np

import matplotlib.pyplot as plt



# Regression

import statsmodels.api as sm

import statsmodels.formula.api as smf



# Drawing

from PIL import Image, ImageDraw

import sys

from IPython.display import display



"""

Import datasets -- these datasets are quite big.

Takes about ~1min to load

"""

KAGGLE = True

if not KAGGLE:

    PT_data = pd.read_csv('PlayerTrackData.csv')

    IR_data = pd.read_csv('InjuryRecord.csv')

    PL_data = pd.read_csv('PlayList.csv')

else:

    df = pd.read_csv('/kaggle/input/q2data/Q2-Data.csv')



pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)
if not KAGGLE:

    WR_data = PL_data[PL_data['PositionGroup'] == 'WR']

    df = pd.merge(PT_data, WR_data, on = 'PlayKey', how='right')

    df = pd.merge(df, IR_data, on = 'PlayKey', how='left')

    df.head()
def labelPlayStartEnd(df):

    START_CRIT = ['ball_snap','snap_direct','onside_kick','kickoff','free_kick','drop_kick',]

    END_CRIT = ['pass_outcome_incomplete','out_of_bounds','tackle','touchdown','qb_kneel','qb_sack','pass_outcome_touchdown','two_point_conversion','qb_spike','touchback','safety','field_goal','fair_catch','punt_downed','extra_point','field_goal_missed']

    END_PASS_CRIT = ['pass_outcome_incomplete', 'pass_outcome_caught', 'pass_outcome_interception'] # unused

     

    df.assign(playActive=False)

    playArray = []

    currentFlag = False

    previousTime = 1000.0



    print('Starting dataframe length', len(df))

    

    # Iter rows

    for i, row in df.iterrows():

    

        event = row['event']

        currTime = row['time']



        # If we hit a start event, set flag to true

        if pd.isnull(event) == False:

            if event in START_CRIT:

                currentFlag = True

            elif event in END_CRIT:

                currentFlag = False

            elif currTime < previousTime: # Just to make sure we ended the play

                currentFlag = False 



        previousTime = currTime

        playArray.append(currentFlag)

        

    df['playActive'] = playArray

    return df
if not KAGGLE:

    df = labelPlayStartEnd(df)

    df = df[df['playActive'] == True]

    df = df.reset_index(drop=True)
# This function calculates the number of games missed and appends it to our dataframe

def countGamesMissed(df):

    

    print('Starting Length', len(df))

    

    GM_array = []

    for i, row in df.iterrows():



        DM1  = row['DM_M1']

        DM7  = row['DM_M7']

        DM28 = row['DM_M28']

        DM42 = row['DM_M42']



        if DM42 == 1:

            GM_array.append(6.5)

        elif DM28 == 1:

            GM_array.append(4.5)

        elif DM7 == 1:

            GM_array.append(1.5)

        elif DM1 == 1:

            GM_array.append(.5)

        else:

            GM_array.append(0)



    df['GamesMissed'] = GM_array

    return df



if not KAGGLE:

    df = countGamesMissed(df)
def drawPlay(pks, scaler=1, displayBool=False, saveBool=False, returnImg=False, CV=False, TRANSPOSE_OPTION=False, injuryColorCode=False):

    

    # Format the plays if just one

    multiplePlays = True

    if not isinstance(pks, list):

        multiplePlays = False

        pks = [pks]

    

    # Set up image dimensions

    fieldWidth = 54 * scaler 

    fieldHeight = 120 * scaler

    LOS = 20 * scaler

    

    # CV or Presentation settings

    if CV == False:

        fieldColor = 'rgb(50, 168, 82)'

        LOSColor = 'blue'

        lineWidth = 2 * scaler       

        hashWidth = 1 * scaler

        endZoneColor = 'rgb(11, 117, 40)'

        

        # Init

        im = Image.new("RGBA", (fieldWidth, fieldHeight), color=fieldColor)

        draw = ImageDraw.Draw(im)

        

        # Endzones

        draw.rectangle(((0, 0), (54*scaler, 10*scaler)), fill = endZoneColor)

        draw.rectangle(((0, (120-10)*scaler), (54*scaler, 120*scaler)), fill = endZoneColor)

        

        # Yard lines

        for i in range(20, 110, 10):

            YL = i * scaler

            draw.line((0,YL) + (fieldHeight,YL), fill='white', width=hashWidth)

            

        # Adjustments for multiple plays

        if multiplePlays == True:

            lineWidth = scaler # Thinner lines

        

    else:

        fieldColor = 'rgb(0,0,0)'

        LOSColor = 'black'

        lineWidth = 2 * scaler

        

        # Init

        im = Image.new("RGBA", (fieldWidth, fieldHeight), color=fieldColor)

        draw = ImageDraw.Draw(im)

        

    # For each play, draw play

    for pk in pks:

        

        # Get data on that play

        playData = df[df['PlayKey'] == pk]

        

        # Figure out if we need to flip the play (we always go up)

        FLIP = False

        startY, endY = list(playData['x'])[0], list(playData['x'])[-1]

        if startY < endY:

            FLIP = True



        # Draw route

        firstRow = True

        lastCoordinates = (0,0)

        startX = None

        LOS_offset = None

        for i, row in playData.iterrows():



            # Flip the play if needed

            if FLIP:

                yardLine = fieldHeight - (row['x'] * scaler)

                yardHash = fieldWidth - (row['y'] * scaler)

            else:

                yardLine = row['x'] * scaler

                yardHash = row['y'] * scaler

            

            coordinates = (yardHash, yardLine)

            if firstRow:



                firstRow = False

                startY = yardLine

                startX = yardHash

                LOS_offset = startY - LOS

                lastCoordinates = coordinates

                

                # Draw line of scimage if just one play

                if multiplePlays == False:

                    draw.line((0,startY) + (fieldHeight,startY), fill=LOSColor, width=lineWidth)

                    

            else:

                if injuryColorCode:

                    # If multiple plays, color code based on injury

                    if (row['GamesMissed'] > 0):

                        routeColor =  'rgb(255, 0, 0)'

                    else:

                        routeColor =  'rgb(0, 0, 255)'

                    

                elif CV:

                    # White for CV model

                    routeColor = 'white'

                else:

                    

                    # Color code route based on speed

                    playerSpeed = row['s']

                    speedScale = int(min(playerSpeed, 12.5) * 20)

                    routeColor =  'rgb(255,' + str(251 - speedScale) + ', 0)'



                draw.line(lastCoordinates + coordinates, fill=routeColor, width=lineWidth)

                endY = yardLine

                lastCoordinates = coordinates



    # Transpose so that we always start left of QB

    if TRANSPOSE_OPTION:

        if startY < endY:        

            if startX < fieldWidth / 2:

                im = im.transpose(Image.FLIP_LEFT_RIGHT)



        elif startX > fieldWidth / 2:

            im = im.transpose(Image.FLIP_LEFT_RIGHT)



    del draw



    # write to stdout

    if displayBool:

        display(im)

    if saveBool:

        im.save('wr/routes/' + str(pk) + '.jpg')

    if returnImg:

        return im
drawPlay('38876-29-14', displayBool = True, CV=False, scaler=3)
drawPlay(['38876-29-12', '38876-29-13', '38876-29-14'], displayBool = True, CV=False, scaler=3)
drawPlay(['38876-29-14'], displayBool = True, CV=True, scaler=3)
WR_INJURY_PLAYS = list(set(df[df['GamesMissed'] > 0]['PlayKey']))



def drawManyRoutes(PLAY_LIST = [], labelRoute = False, routeLabels = []):

    plays = []

    for pk in PLAY_LIST:

        plays.append(drawPlay(pk, returnImg=True))



    

    TOTAL_ROWS = int(len(PLAY_LIST) / 8) + 1

    plt.figure(figsize=(14,5 * (TOTAL_ROWS)))

    

    for n in range(len(PLAY_LIST)):

        ax = plt.subplot(TOTAL_ROWS,8,n+1)

        plt.xticks([])

        plt.yticks([])

        plt.grid(False)

        plt.imshow(plays[n], cmap=plt.cm.binary)

        

        if labelRoute:

            plt.xlabel(routeLabels[n] + '\n' + str(PLAY_LIST[n]))

        else:

            plt.xlabel(str(df[df['PlayKey'] == PLAY_LIST[n]]['GamesMissed'].max()) + \

                       ' GM \n' + str(PLAY_LIST[n]))

        

drawManyRoutes(WR_INJURY_PLAYS)
# Draw a random sample of plays

ALL_PLAYS = list(df.drop_duplicates(["PlayKey"])['PlayKey'])

drawPlay(random.choices(ALL_PLAYS, k=100) + random.choices(WR_INJURY_PLAYS, k=12), scaler=6, displayBool = True, injuryColorCode=True)

print('Red=Injury; Blue=Non-injury')
# Draw a random sample of plays

drawPlay(random.choices(ALL_PLAYS, k=50), scaler=6, displayBool = True, CV=False)
from IPython.display import Image as DisplayInlineImage

DisplayInlineImage("../input/routespng/routes.png")
injuryRoutes = [

    'Curl',    'Go',    'Go',    'NA',      'Slant',  'Go',     'Out',  'Go',

    'NA',      'Out',   'Curl',   'Slant',   'Go',    'Slant',  'Go',   'Out',

]

drawManyRoutes(WR_INJURY_PLAYS, labelRoute = True, routeLabels = injuryRoutes)
ROUTES200 = ['41086-24-31', '42405-8-55', '38364-19-22', '40335-15-34', '44449-1-41', '42405-12-23', '44203-25-18', '38192-20-36', '42359-32-46', '41086-19-60', '30068-6-54', '47285-2-37', '45962-30-45', '38192-24-21', '42405-9-34', '44423-9-10', '44424-17-63', '47376-13-37', '34347-3-11', '46021-24-6', '45962-23-39', '42359-15-49', '44449-27-17', '44203-18-17', '34243-9-20', '45061-1-2', '40405-15-1', '42398-5-31', '42432-11-2', '44203-13-11', '38252-22-25', '43532-6-25', '42432-8-21', '30068-25-49', '38252-17-36', '44424-7-38', '44203-26-14', '42432-7-40', '38192-6-40', '30068-15-32', '42346-24-42', '34230-7-45', '42346-11-45', '42432-21-27', '45987-30-63', '40335-27-43', '42346-5-17', '42346-15-45', '38192-20-46', '44449-19-47', '38876-28-21', '42432-3-12', '34347-12-1', '45962-11-26', '42359-1-10', '44203-17-53', '45987-29-58', '45987-27-24', '45962-25-40', '42405-9-57', '45962-1-61', '45962-6-9', '46021-19-45', '45987-15-44', '42405-12-17', '40335-23-41', '38192-22-26', '42346-12-5', '35648-14-18', '38192-23-10', '42405-5-42', '40405-23-9', '45987-22-53', '42432-12-9', '44424-20-1', '36591-3-5', '36630-19-21', '44423-7-50', '34243-24-38', '47376-9-11', '34243-17-3', '35648-5-20', '46067-11-11', '42359-4-39', '45987-2-31', '47273-8-5', '42432-10-37', '42405-2-37', '38876-17-22', '44203-25-7', '44449-4-39', '41086-20-75', '43532-4-36', '45962-18-42', '42359-23-2', '46021-9-7', '39656-19-13', '43532-18-47', '42432-11-34', '42346-4-1', '44203-23-3', '46021-20-56', '30068-28-27', '38364-11-16', '36630-7-7', '35648-6-6', '36591-2-2', '39809-24-7', '42432-22-20', '30068-2-52', '42405-8-4', '45962-17-2', '46021-8-35', '44203-32-33', '34243-15-3', '43532-12-7', '45962-15-40', '47273-15-31', '38192-24-9', '42346-18-24', '44449-28-22', '34230-4-9', '42432-9-35', '47784-13-48', '38876-16-11', '42448-15-15', '34347-23-7', '47285-2-23', '42346-22-38', '34243-8-1', '42405-27-31', '42405-18-6', '34230-19-50', '46021-11-28', '42405-6-10', '38876-8-35', '47784-7-1', '42432-6-21', '44203-13-34', '46021-20-48', '44424-14-43', '44424-8-49', '42448-4-38', '44203-9-6', '45987-15-49', '47287-3-11', '42448-11-3', '47784-13-19', '42448-5-20', '38252-9-9', '45987-29-13', '34230-3-55', '44449-15-8', '38252-12-20', '46021-26-51', '42398-12-51', '41086-24-35', '42398-7-2', '34230-20-14', '47376-12-12', '38252-2-50', '45962-17-36', '44449-9-15', '42346-11-44', '38876-5-39', '30068-31-30', '42405-23-44', '30068-29-42', '34347-7-7', '42346-26-27', '42405-12-40', '42346-12-6', '44424-24-28', '47784-10-7', '30068-16-31', '38192-7-37', '30068-29-31', '44424-9-58', '39656-15-11', '40335-18-22', '44203-32-21', '38252-1-12', '47273-7-8', '47287-10-2', '35648-12-6', '39656-12-1', '43532-9-65', '45187-19-11', '46098-18-26', '44424-23-29', '47287-10-39', '42359-29-41', '42346-26-44', '46067-10-14', '41086-17-46', '42346-5-30', '42448-6-7', '47287-8-45', '47287-3-8', '44203-25-31']
ROUTES200Labels = [

    'Slant', 'NA',    'Slant',  'Go',     'NA',     'Slant',  'NA',  'NA',  

    'Slant', 'Go',    'Go',      'Slant',  'Go',     'Go',     'Go',     'Slant',

    'Out',   'Out',   'Out',    'Curl',   'NA',     'NA',     'Go',     'Slant',

    'Go',    'Go',    'Slant',  'Go',     'Slant',  'Slant',  'Out',    'Slant',

    'Out',   'Slant', 'Out',    'Out',    'Slant',  'NA',     'NA',     'NA',

    'Curl',  'Out',   'Go',     'Slant',  'Slant',  'Go',     'NA',     'Go',

    'NA',    'NA',    'Go',     'Slant',  'Out',    'NA',     'Out',    'Go',

    'NA',    'Go',    'Slant',  'NA',     'NA',     'Out',    'Go',     'NA',

    'NA',    'Go',    'Out',    'Go',     'Go',     'Slant',  'Out',    'NA', 

    'NA',    'Slant', 'NA',     'NA',     'Slant',  'Slant',  'Slant',  'Go',

    'NA',    'Slant', 'Go',     'Out',    'Go',     'Out',    'Go',     'Slant',

    'Go',    'Slant', 'NA',     'NA',     'NA',     'Slant',  'Slant',  'Out',

    'Out',   'Out',   'NA',    'NA',     'Slant',  'Slant',  'NA',     'Slant',

    'Go',    'NA',    'Go',     'Go',     'Curl',   'Slant',  'Out',    'NA',

    'Slant', 'NA',    'NA',     'NA',     'NA',     'Go',     'Slant',  'NA',

    'Curl',  'NA',    'Slant',  'NA',     'NA',     'Out',    'Slant',  'Slant',

    'Out',   'Out',   'Slant',  'Slant',  'Go',     'Go',     'NA',     'Slant',

    'Go',    'Slant', 'Go',     'Out',    'Out',    'Curl',   'NA',     'Go',

    'NA',    'NA',    'Slant',  'NA',     'Slant',  'NA',     'NA',     'NA',

    'Go',    'NA',    'Slant',  'NA',     'Curl',   'Slant',  'NA',     'Go',

    'Slant', 'Go',    'NA',     'Out',    'Slant',  'NA',     'NA',     'Slant',

    'Go',    'Go',    'Curl',   'NA',     'NA',     'NA',     'NA',     'NA',

    'NA',    'Curl',  'Slant',  'Slant',  'NA',     'NA',     'Go',     'Slant',

    'NA',    'Slant', 'NA',     'NA',     'Slant',  'Out',    'Go',     'Slant',

    'Out',   'Go',    'Out',    'NA',     'Go',     'NA',     'NA',     'Out',

]
drawManyRoutes(ROUTES200[0:200], labelRoute = True, routeLabels = ROUTES200Labels)
routeDF = pd.DataFrame(index = list(set(ROUTES200Labels)))



routeDF['Injury'] = [injuryRoutes.count(i) for i in list(routeDF.index)]

routeDF['Injury_Average'] = routeDF['Injury'] / sum(routeDF['Injury'])



routeDF['All'] = [ROUTES200Labels.count(i) for i in list(routeDF.index)]

routeDF['All_Average'] = routeDF['All'] / sum(routeDF['All'])



routeDF['FrequencyDifference'] = routeDF['Injury_Average'] / routeDF['All_Average']





def getPValue(x_True, x_Total, y_True, y_Total):

    x_sample = [1] * x_True + [0] * (x_Total - x_True)

    y_sample = [1] * y_True + [0] * (y_Total - y_True)

    t_stat, p_val = stats.ttest_ind(x_sample, y_sample, equal_var=False)

    return p_val



routeDF['P-Stat'] = [getPValue(int(routeDF.loc[i, 'Injury']), 12, 

                               int(routeDF.loc[i, 'All']), 200) for i in list(routeDF.index)]



routeDF
playKey = []

playKeyRows = []

playKeyRowsBuilder = []

previousRowPlay = '' 



# Add to the dataframe

print('Starting Rows', len(df))

for i, row in df.iterrows():

    

    # Get current play key

    thisPlayKey = row['PlayKey']

    if previousRowPlay != thisPlayKey:

        if previousRowPlay != '':

            playKeyRows.append(playKeyRowsBuilder)

        playKeyRowsBuilder = []

    previousRowPlay = thisPlayKey

    playKeyRowsBuilder.append(i)



playKeyRows.append(playKeyRowsBuilder)

playKeyRowsBuilder = []
ALL_PLAYS = list(df.drop_duplicates(["PlayKey"])['PlayKey'])

play_df = pd.DataFrame()

play_df['PlayKey'] = ALL_PLAYS

play_df['Keys'] = playKeyRows

play_df = pd.merge(play_df, df, on = 'PlayKey', how='left').drop_duplicates(["PlayKey"])

play_df = play_df.drop(columns=['PlayerKey_y', 'GameID_y', 'playActive', 'time','event','x','y','dir','dis','o','s'])
def getAcceleration(play):

    speeds = list(play['s'])

    accelerations = []

    if len(speeds) > 5:

        for i in range(5, len(speeds)):

            accelerations.append(speeds[i] - speeds[i-5])       

        return max(accelerations) * 2, min(accelerations) * 2 # we do 0.5 second intervals, mulitply by 2

    else:

        return 0, 0 



# Get the total degrees the player turned

def getTurnDegrees(play):



    turns = list(play['o'])

    degrees = 0

    for i in range(len(turns)):

        if i != 0:

            turn = abs(turns[i] - turns[i-1])

            if turn > 200:

                turn = 0

            degrees += turn

    return degrees



distanceTraveled = []

yMovement = []

xMovement = []



degreesTurned = []



maxSpeed = []

maxAcceleration = []

maxDeceleration = []



print('Starting Rows', len(play_df))

for i, row in play_df.iterrows():

    play = df.iloc[row['Keys']] 

        

    # Distance

    distanceTraveled.append(sum(play['s']) * .1)

    yMovement.append(play['y'].max() - play['y'].min())

    xMovement.append(play['x'].max() - play['x'].min())

    

    # Turning

    degreesTurned.append(getTurnDegrees(play))

    

    # Speed and Acceleration

    maxSpeed.append(play['s'].max())

    acceleration, deceleration = getAcceleration(play)

    maxDeceleration.append(deceleration)

    maxAcceleration.append(acceleration)

    

    

play_df['MaxSpeed'] =  maxSpeed

play_df['distanceTraveled'] = distanceTraveled

play_df['degreesTurned'] = degreesTurned

play_df['yMovement'] =  yMovement

play_df['xMovement'] =  xMovement

play_df['maxAcceleration'] =  maxAcceleration

play_df['maxDeceleration'] =  maxDeceleration



play_df.head(2)
def convertVariablesForRegression(tmp):



    def isIn(item, word):

        return item in str(word).lower()



    # Convert some fields to boolean

    tmp['Is_Synthetic'] = tmp['FieldType'] == 'Synthetic'

    tmp['Is_Rain'] = [1 if isIn('rain', i) or isIn('shower', i) else 0 for i in tmp['Weather']]

    tmp['Is_Snow'] = [1 if isIn('snow', i) else 0 for i in tmp['Weather']]



    # Clean some variables

    tmp['Temperature_Adj'] = [i if i != -999 else 72 for i in tmp['Temperature']]

    

    return tmp

    

play_df = convertVariablesForRegression(play_df)
X = 'GamesMissed'

Y = ['MaxSpeed', 'distanceTraveled', 'degreesTurned', 'yMovement', 'xMovement', 'Is_Synthetic', \

     'PlayerDay', 'PlayerGamePlay', 'Temperature_Adj', 'Is_Rain', 'Is_Snow', \

     'maxAcceleration', 'maxDeceleration']

Y_a = ' + '.join(Y)



model = sm.GLM.from_formula(X + ' ~ ' + Y_a, data=play_df, family=sm.families.Poisson()).fit()

print(model.summary())
play_df['Temperature_Adj'].corr(play_df['MaxSpeed'])
# Movement differences

movementDF = pd.DataFrame(columns = ['All', 'Injury', 'P-Val'])

for metric in Y:

    allPlays = play_df[metric]

    injuryPlays = play_df[play_df['GamesMissed'] > 0][metric]

    t_stat, p_val = stats.ttest_ind(allPlays, injuryPlays, equal_var=False)

    movementDF.loc[metric] = [allPlays.mean(), injuryPlays.mean(), round(p_val,3)]

    

movementDF['Ratio'] = movementDF['Injury'] / movementDF['All']

movementDF