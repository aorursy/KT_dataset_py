#code review welcome.  I'm pretty sure this is totally not pythonic.  I'm new.

#also any ideas for more pleasing graphical formatting are welcome.

#

#Feel free to use the code if its of use.

#

#Code for 10-fold cross validation set below, commented out.

#I don't think it will work on kaggle b/c of incremental saves..
import pandas as pd # data processing, CSV file I/O (e.g. pd. read_csv)

import numpy as np # linear algebra



import datetime

from datetime import timedelta

from dateutil.relativedelta import relativedelta



#imports for saving files

#from pathlib import Path

#import os.path



#import for alternate, random msnos

#from random import randint



import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

sns.set_style("whitegrid")



pd.options.mode.chained_assignment = None # default='warn'
train = pd.read_csv('../input/kkbox-churn-prediction-challenge/train.csv', nrows=20000)

#cross_validation_set = 0

#train = pd.read_csv('../input/kkbox-churn-prediction-challenge/train.csv', nrows=99000, skiprows=range(1, (cross_validation_set * 99000 - 1)))



members = pd.read_csv('../input/kkbox-churn-prediction-challenge/members.csv')

members = members.loc[members['msno'].isin(train['msno'])]



transactions = pd.read_csv('../input/kkbox-churn-prediction-challenge/transactions.csv')

transactions = transactions.loc[transactions['msno'].isin(train['msno'])]



userLog = pd.read_csv('../input/small-userlog-sample/UserLog_train0.csv')

#if you have the cross validation sets locally, load with this...

#userLog = pd.read_csv('F:kaggle/UserLog_train' + str(cross_validation_set) + '.csv')
exUser = pd.merge(train, userLog, how='inner', on=['msno']).msno.unique()
## This code produces a 10-fold cross validation training set.

## I'm not sure if it will work on kaggle, so this notebook uses

## a smaller set (first 20,000 users in train)



#user_log_length = pd.read_csv('E:kaggle csvs/first kaggle stay or go/user_logs.csv', usecols=['msno'])

#number_of_records_to_go_through = max(user_log_length.shape)

#del user_log_length

##above code returns this          vvvvvvvvv

#number_of_records_to_go_through = 392106543



####WARNING: THIS WILL TAKE QUITE A LONG TIME TO DO ITS MAGIC.  MAKE SURE YOU CHANGE THE DIRECTORIES TO SUIT YOUR NEEDS.



#num_records impacts speed substantially,- if anything crashes, try lowering this to 20000000 to 40000000

#but the lower the number, the slower, it has to read from the first line of the file, even skipping rows.

#num_records = 70000000

#

#for cv_fold in range (0, 10):

#    num_records_left = number_of_records_to_go_through

#    chunk = 0

#

#    train = pd.read_csv('E:kaggle csvs/first kaggle stay or go/train.csv', nrows=99000, skiprows=range(1, (cv_fold * 99000 - 1)))

#

#    while (num_records_left > num_records):

#        print('starting on chunk number ' + str(chunk))

#

#        userLog = pd.read_csv('E:kaggle csvs/first kaggle stay or go/user_logs.csv', skiprows = (num_records*chunk), nrows=(num_records), header=None)

#        userLog.rename(columns ={0: 'msno', 1:'date', 2:'num_25', 3:'num_50', 4:'num_75:', 5:'num_985', 6:'num_100', 7:'num_unq', 8:'total_secs'}, inplace=True)

#        userLog_train = userLog.loc[userLog['msno'].isin(train['msno'])]

#

#        chunk += 1

#        num_records_left = num_records_left - num_records

#        

#        persistent_save = Path("F:kaggle/userLog_train" + str(cv_fold) + ".csv")

#        if persistent_save.is_file():

#            with open('F:kaggle/userLog_train' + str(cv_fold) + '.csv', 'a') as f:

#                userLog_train.to_csv(f, header=False)

#                f.close()

#                print('csv appended')

#        else:

#            userLog_train.to_csv('F:kaggle/UserLog_train' + str(cv_fold) + '.csv')

#            print('csv created')

#    

#        del userLog

#        del userLog_train

#

#    print('made it to the final group of records!')

#

#    userLog = pd.read_csv('E:kaggle csvs/first kaggle stay or go/user_logs.csv', skiprows = (num_records*(chunk)), nrows=(num_records_left - 1), header=None)

#    userLog.rename(columns ={0: 'msno', 1:'date', 2:'num_25', 3:'num_50', 4:'num_75:', 5:'num_985', 6:'num_100', 7:'num_unq', 8:'total_secs'}, inplace=True)

#    userLog_train = userLog.loc[userLog['msno'].isin(train['msno'])]

#

#    persistent_save = Path("F:kaggle/userLog_train" + str(cv_fold) + ".csv")

#    if persistent_save.is_file():

#        with open('F:kaggle/userLog_train' + str(cv_fold) + '.csv', 'a') as f:

#            userLog_train.to_csv(f, header=False)

#            f.close()

#            print('csv appended')

#    else:

#        userLog_train.to_csv('F:kaggle/UserLog_train' + str(cv_fold) + '.csv')

#        print('csv created')
#different values for xyz give graphs of different users' timeseries



# PLAY HERE

#    |

#   \/

xyz=1000

numberOfUsers=100



for user in range(0, numberOfUsers):

    ###################################

    ###  USER SUBSCRIPTION HISTORY  ###

    ###################################

    #xyz = randint(0,40000)

    trans = transactions.loc[transactions['msno'] == exUser[xyz]]

    trans['startDate'] = trans['transaction_date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d %H:%M:%S'))

    trans['endDate'] = trans['membership_expire_date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d %H:%M:%S'))

    trans['linewidth']= (trans['actual_amount_paid'] / trans['payment_plan_days'])

    trans = trans[['startDate','endDate','linewidth']]

    trans = trans.sort_values('startDate')

    trans.index = pd.RangeIndex(len(trans.index))

    trans['type']='transaction'

    trans['color']='slateblue'

    

#some transactions are in reverse chronological order, this flips the two dates.

    for x in range(0, (len(trans.index))):

        if (trans.endDate[x] < trans.startDate[x]):

            temp = trans.endDate[x]

            trans.endDate[x] = trans.startDate[x]

            trans.startDate[x] = temp

            

    trans = trans.sort_values('startDate')

    trans.index = pd.RangeIndex(len(trans.index))

    

#merges single-day overlaps

#    indexDeletions = []

#    for x in range(0, (len(trans.index) - 1)):

#        if (trans.endDate[x] == trans.startDate[x + 1]):

#            trans.endDate[x] = trans.endDate[x+1]

#            indexDeletions.append(x+1)

#    for index in range(len(indexDeletions), 0):

#        trans.drop(indexDeletions[(index)], inplace=True)

#    trans.index = pd.RangeIndex(len(trans.index))    



    #find lapses, churns, and redundant subscription in transactions

    beginLapse = []

    endLapse = []

    colorLapse = []

    

    beginChurn = []

    endChurn = []

    

    for x in range(0, (len(trans.index) - 1)):

        #lapse

        if (trans.endDate[x] < (trans.startDate[x+1] - timedelta(days=1))):

            beginLapse.append((trans.endDate[x] + timedelta(days=1)))

            endLapse.append(trans.startDate[x+1] - timedelta(days=1))

            colorLapse.append('darkgrey')

        #redundant subscription

        if (trans.endDate[x] >= (trans.startDate[x+1])):

            beginLapse.append(trans.endDate[x])

            endLapse.append(trans.startDate[x+1])

            colorLapse.append('greenyellow')

        #churn

        if ((trans.endDate[x] + relativedelta(months=1)) < trans.startDate.iloc[x+1]):

            beginChurn.append(trans.endDate[x] + relativedelta(months=1))

            endChurn.append(trans.startDate[x+1])



    lapses = pd.DataFrame({'beginLapse': beginLapse, 'endLapse' : endLapse, 'color': colorLapse})

    churn = pd.DataFrame({'beginChurn' : beginChurn, 'endChurn' : endChurn})  

    

    ################################

    ###  USER LISTENING HISTORY  ###

    ################################

    cleanedUserLog = userLog.loc[userLog['msno'] == exUser[xyz]]

    cleanedUserLog['endDate'] = cleanedUserLog['date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d %H:%M:%S'))

    cleanedUserLog['startDate'] = cleanedUserLog['endDate'] - timedelta(days=1)

    cleanedUserLog['linewidth'] = cleanedUserLog['total_secs'] / 1200

    cleanedUserLog['color'] = 'darkgoldenrod'

    cleanedUserLog['type'] = 'userLog'

    

    ##############

    ###  plot  ###

    ##############

    matplotlib.rcParams['figure.figsize'] = (16, 1)

    frames = [trans, cleanedUserLog]

    ex = pd.concat(frames)

    ex = ex.sort_values('startDate')

    ex = ex.reset_index(drop=True)



    for x in range (0, len(ex.index)):

            plt.hlines(0, ex.startDate.iloc[x], ex.endDate.iloc[x], ex.color.iloc[x], linewidth=ex.linewidth.iloc[x], alpha = .70)



    for y in range (0, (len(lapses))):

        plt.axvspan(lapses.endLapse.iloc[y], lapses.beginLapse.iloc[y], ymin = .3, ymax = .7, facecolor = lapses.color.iloc[y], alpha=.4)

        

    for z in range (0, (len(churn))):

        plt.axvspan(churn.endChurn.iloc[z], churn.beginChurn.iloc[z], facecolor = 'k', alpha=.25)



    plt.ylabel('logged event')

    plt.xlabel('date')

    ischurn= train.iloc[xyz].is_churn

    

    plt.title('User History for: xyz = ' + str(xyz) + ',  ' + exUser[xyz] + ',  is_churn=' + str(ischurn))

    plt.show()



    xyz += 1

    

    #Logs are in brown, their height proportional to time the user listened to music.

    #The purple line indicates an active subscription, height proportional to cost.

    #grey boxes indicate lapses in subscription

    #green indicates double subscription

    #black indicates the user is in churn  