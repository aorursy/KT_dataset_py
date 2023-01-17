import pandas as pd



train_data = pd.read_csv("../input/zeemee-micro-competition-data/zeemee_train.csv")

test_data = pd.read_csv("../input/zeemee-micro-competition-data/zeemee_test.csv")
train_data.columns
train_data.describe()
test_data.describe()
train_data.head()
train_data.shape
## This section is so that I can view the types of data being used for each catagory and look for missing values

## This was also useful for figuring out which values were catagorical and which were ordinal

##some sections have been commented out as the output is lengthy and not frequently useful to view





print("college: ")

display(train_data.college.unique())

print("\n\npublic profile enabled: ")

display(train_data.public_profile_enabled.unique())

print("\n\ngoing: ")

display(train_data.going.unique())

print("\n\ninterested: ")

display(train_data.interested.unique())

print("\n\nstart term: ")

display(train_data.start_term.unique())

print("\n\ncohort year: ")

display(train_data.cohort_year.unique())

print("\n\ncreated by csv: ")

display(train_data.created_by_csv.unique())

#print("\n\nlast login: ") 

#display(train_data.last_login.unique()) ##found Nan here

print("\n\nschools followed: ")

display(train_data.schools_followed.unique())

print("\n\nhigh school: ")

display(train_data.high_school.unique())

print("\n\ntransfer status: ")

display(train_data.transfer_status.unique())

print("\n\nroommate match quiz: ")

display(train_data.roommate_match_quiz.unique())

#print("\n\nchat messages sent: ")

#display(train_data.chat_messages_sent.unique())

#print("\n\nchat viewed: ")

#display(train_data.chat_viewed.unique())

#print("\n\nvideos liked: ")

#display(train_data.videos_liked.unique())

#print("\n\nvideos viewed: ")

#display(train_data.videos_viewed.unique())

#print("\n\nvideos veiwed unique: ")

#display(train_data.videos_viewed_unique.unique())

#print("\n\ntotal official videos: ")

#display(train_data.total_official_videos.unique())

print("\n\nengaged: ")

display(train_data.engaged.unique())

print("\n\nfinal funnel stage: ")

display(train_data.final_funnel_stage.unique())
display(test_data.cohort_year.unique())

display(test_data.college.unique())

display(test_data.start_term.unique())
## train_data = train_data[train_data.cohort_year != 2017]

## this line lost accuracy in the model, interesting hypothesis though
print('is null in training data: ')

display(train_data.isnull().sum())

print('\n\nis null in test data: ')

display(test_data.isnull().sum())
train_data['last_login'] = train_data['last_login'].fillna(0)

## I tried running the model by filling the null entry with 0 and then again with 1500

## the assumption was that the null entry was either from the time being too long 

## to record or the entry being null because the user was loging on at the time of capture

test_data['last_login'] = test_data['last_login'].fillna(0)
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split



print('Setup Complete')
def final_funnel_num(val):

    if val == 'Inquired':

        return 0

    

    if val == 'Applied':

        return 0

    

    if val == 'Accepted':

        return 0

    

    if val == 'Deposited':

        return 1

    

    if val == 'Application_Complete':

        return 0

    

    if val == 'Enrolled':

        return 1

    

    else:

        return 'error'

    

## at this point in the process you may be wondering why I'm not using one of the in built encoding methods in sklearn over making my own

## When dealing with ordinal values (values that can be converted to integers and still hold meaning) it is benificial to hand encode values to ensure your model is not loosing some of the meaning in the data



## Also worth pointing out here, initially I had thought i needed to predict all 6 outcomes for the model, when I double checked the rules I discovered that the model was supposed to predict a binary outcome of either or for a subset of the outcomes (Enrolled and Deposited)

## which is why this function seems a bit silly, about an hour and a half before the deadline i discovered this error and went for the simplest fix.

## interestingly enough when I was predicting 6 outcomes the model was 54% accurate in testing (17% is the accuracy that dice would have had)

## I'm also curious how it would have effected accuracy if i had the algorithm predict all six outcomes and then translated that result to the desired binary output
def num_bool(val):

    if val == True:

        return 1

    elif val == False:

        return 0

    else:

        return 'error'

## The return code 'error' would actually be an error when the data is fed into the model

## as a random forrest cannot handle values that are non numeric
def goin_num(val):

    if val == 'undecided':

        return 0

    if val == 'going':

        return 1

    if val == 'notgoing':

        return -1

    else:

        return 'error'
def col_num(val):

    if val == 'college1':

        return 1

    if val == 'college2':

        return 2

    if val == 'college4':

        return 4

    if val == 'college3':

        return 3

    if val == 'college5':

        return 5

    if val == 'college6':

        return 6

    if val == 'college7':

        return 7

    if val == 'college8':

        return 8

    else:

        return 'error'
def term_num(val):

    if val == 'fall':

        return 1

    if val == 'spring':

        return 2

    if val == 'summer':

        return 3

    else:

        return 'error'
## since I'm using a random forest model for this project I will be converting all values I wish to use to numbers

## most likely i will have to drop some of these values for my final project, but its nice to have options

train_data['funnel_num'] = pd.Series([final_funnel_num(x) for x in train_data.final_funnel_stage], index=train_data.index)

train_data['transfer_status_num'] = pd.Series([num_bool(x) for x in train_data.transfer_status], index=train_data.index)

train_data['public_profile_enabled_num'] = pd.Series([num_bool(x) for x in train_data.public_profile_enabled], index=train_data.index)

train_data['interested_num'] = pd.Series([num_bool(x) for x in train_data.interested], index=train_data.index)

train_data['created_by_csv_num'] = pd.Series([num_bool(x) for x in train_data.created_by_csv], index=train_data.index)

train_data['roommate_match_quiz_num'] = pd.Series([num_bool(x) for x in train_data.roommate_match_quiz], index=train_data.index)

train_data['going_num'] = pd.Series([goin_num(x) for x in train_data.going], index=train_data.index)

train_data['college_num'] = pd.Series([col_num(x) for x in train_data.college], index=train_data.index)

train_data['start_term_num'] = pd.Series([term_num(x) for x in train_data.start_term], index=train_data.index)

## Also worth mentioning: isn't it wonderful not having to scale features for a random forest model
train_data.describe()
y = train_data.funnel_num

rf_features = ['cohort_year',  'going_num', 'chat_messages_sent', 'schools_followed', 'videos_liked',  'chat_viewed', 'total_official_videos',  'videos_viewed','transfer_status_num',  'videos_viewed_unique', 

               'public_profile_enabled_num', 'created_by_csv_num', 'interested_num', 'roommate_match_quiz_num', 'college_num']

X = train_data[rf_features]



## what isn't pictured here is the trial and error as I remove various components, and reran the next 2 cells to compare accuracy.

## This process also taught me something new, the order the features are in affects the model created



## As a worthwhile point to mention, I had initially guessed that using label encoding for the college would have confused a random forrest and that i was going to need to retry using one hot encoding or a sparse vector instead, 

## however using label encoding did have a positive end result (assuming I didn't overfit the model)



## start_term_num and last_login was removed as it did not improve the accuracy of the model



## Other notes:  roommate_match_quiz_num and public_profile_enabled_num had an almost insignifigant impact on accuracy, I chose to leave them in because the impact was positive

## however if this were a production enviorment i might drop it for computational efficiency





train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
forest_model = RandomForestRegressor(random_state=0)

forest_model.fit(train_X, train_y)

td_preds = forest_model.predict(val_X)

print(mean_squared_error(val_y, td_preds))
td_preds = td_preds.round()
print(td_preds.max(), td_preds.min())

## Checking to make sure there are no nonsensicle outputs
print(mean_squared_error(val_y, td_preds))
## this cell I added after the competition. Since the data sample has a split of about 1 positive case in 10

## I wanted to compare other evaluation methods

from sklearn.metrics import roc_auc_score



print(roc_auc_score(val_y, td_preds))
full_preds = forest_model.predict(train_data[rf_features])



train_data['preds'] = full_preds.round()
train_data.head(20)
y = 0

x = 0

for index, row in train_data.iterrows():

#    print(row['funnel_num'])

    if row['funnel_num'] == row['preds']:

        x = x + 1

    else:

        y = y + 1

print("Percent accuracy of model is: ", (x/(x+y))*100)
y = 0

x = 0

n = 0

q = 0

for index, row in train_data.iterrows():

#    print(row['funnel_num'])

    if row['funnel_num'] == 1:

        x = x + 1

    else:

        y = y + 1

    if row['preds'] == 1:

        n = n + 1

    else:

        q = q + 1

        

print('Percentage of positives in sample: ', (x/(x+y))*100, '\nPercentage of predicted positives in sample: ', (n/(n+q))*100)
test_data['transfer_status_num'] = pd.Series([num_bool(x) for x in test_data.transfer_status], index=test_data.index)

test_data['public_profile_enabled_num'] = pd.Series([num_bool(x) for x in test_data.public_profile_enabled], index=test_data.index)

test_data['interested_num'] = pd.Series([num_bool(x) for x in test_data.interested], index=test_data.index)

test_data['created_by_csv_num'] = pd.Series([num_bool(x) for x in test_data.created_by_csv], index=test_data.index)

test_data['roommate_match_quiz_num'] = pd.Series([num_bool(x) for x in test_data.roommate_match_quiz], index=test_data.index)

test_data['going_num'] = pd.Series([goin_num(x) for x in test_data.going], index=test_data.index)

test_data['college_num'] = pd.Series([col_num(x) for x in test_data.college], index=test_data.index)
test_preds = forest_model.predict(test_data[rf_features])

print('values should be 0  and 1.0: ', test_preds.min(), test_preds.max())

test_data['preds'] = test_preds.round()

test_data.head(20)
#def funnel_final(val):    

#    if val == 1:

#        return 'Inquired'

#    

#    if val == 2:

#        return 'Applied'

#    

#    if val == 3:

#        return 'Accepted'

#    

#    if val == 4:

#        return 'Deposited'

#    

#    if val == 5:

#        return 'Application_Complete'

#    

#    if val == 6:

#        return 'Enrolled'
## Now to bring this hot mess full circle

test_data['final_funnel_stage'] = pd.Series([x for x in test_data.preds], index=test_data.index)
test_data.head()
test_data = test_data[['college', 'public_profile_enabled', 'going', 'interested',

       'start_term', 'cohort_year', 'created_by_csv', 'last_login',

       'schools_followed', 'high_school', 'transfer_status',

       'roommate_match_quiz', 'chat_messages_sent', 'chat_viewed',

       'videos_liked', 'videos_viewed', 'videos_viewed_unique',

       'total_official_videos', 'engaged', 'final_funnel_stage']]

print(test_data.shape) ## checking to make sure we're back to 20 columns and that i didn't screw something up
#test_data.to_csv("zeemee_test_output.csv")

#output file for the copetition