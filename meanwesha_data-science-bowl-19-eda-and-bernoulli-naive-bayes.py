# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')

train_df.shape
train_df.head(3)
train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

train_labels.shape
train_labels.head(3)
#world is the section of the application the game or video belongs to. Helpful to identify the educational curriculum goals of the media. 

#Possible values are: 

#'NONE' (at the app's start screen), TREETOPCITY' (Length/Height), 'MAGMAPEAK' (Capacity/Displacement), 'CRYSTALCAVES' (Weight).

train_df['world'].unique()
train_df['installation_id'].unique()[:3]
# game_session is a randomly generated unique identifier grouping events within a single game or video play session of any world.

train_df[ train_df['installation_id']=='0001e90f'].groupby('game_session')['world'].value_counts()
train_df[ train_df['installation_id']=='000447c4'].groupby('game_session')['world'].value_counts()
#This means every time you start a game or a activity, it will be assigned a new game session. It does not depend on installation_id.

train_df.loc[ train_df['game_session']=='a1ec58f109218255', ['title','timestamp']].head(10)
train_df.loc[ train_df['game_session']=='f11eb823348bfa23', ['title','timestamp'] ].head(10)
#event_code - Identifier of the event 'class'. Unique per game, but may be duplicated across games. E.g. event code '2000' always 

#identifies the 'Start Game' event for all games. Extracted from event_data.

train_df.loc[ train_df['game_session']=='f11eb823348bfa23', 'event_code'].value_counts()
train_df.loc[ train_df['game_session']=='a1ec58f109218255', 'event_code'].value_counts()
#event_id is a randomly generated unique identifier for the event type.

train_df.loc[ (train_df['game_session']=='a1ec58f109218255') & (train_df['event_code']==4020), 'event_id'].unique()
train_df.loc[ (train_df['game_session']=='a1ec58f109218255') & (train_df['event_code']==4020)].head(4)
train_df.loc[ (train_df['installation_id']=='0001e90f') & (train_df['event_code']==4020) & (train_df['title']=='Scrub-A-Dub') ].head(4)
train_df.loc[ (train_df['game_session']=='ca8b415f34d12873') & (train_df['event_code']==4020)].head(4)
#So, this means as long as you play the same game or video or activity anytime, the event_id corresponding to an event_code will remain 

#same, even though the game_session will get changed. Once the title changes the event_id corresponding to an event_code will also change.

#Now let's merge the train_df dataframe with train_labels dataframe.

merged = pd.merge( left = train_df, right = train_labels, on = ['installation_id', 'game_session', 'title'] )

merged.shape
merged.head(3)
set1 = train_df[train_df['type']=='Assessment']['installation_id'].unique()

set2 = merged['installation_id'].unique()

diffset = set(set1) - set(set2)

len(diffset)
#It seems like there are some installation_id which took part in some assessment activity, but they were not recorded in train_labels table,

#so I am assuming that may be these assessments are part of some demo activities shown to the children to demonstrate the working of the app.

merged.groupby('world')['title'].unique()
#Again, TREETOPCITY' (Length/Height), 'MAGMAPEAK' (Capacity/Displacement), 'CRYSTALCAVES' (Weight).

# If you notice the outcomes in this competition are grouped into 4 'accuracy_group' in the data labeled as:

# 3: the assessment was solved on the first attempt. means accuracy = 1/(1+0) = 1 i.e. 1 correct after 0 incorrect attempt.

# 2: the assessment was solved on the second attempt. means accuracy = 1/(1+1) = 0.5 i.e. 1 correct after 1 incorrect attempt.

# 1: the assessment was solved after 3 or more attempts

# 1/(1+2) i.e. 1 correct after 2 incorrect attempts or, 1/(1+3) or, 1/(1+4) ...means accuracy <= 0.333

# 0: the assessment was never solved. means accuracy = 0/(0 + attempted any number of times)

merged.groupby(['accuracy_group', 'accuracy'])['num_incorrect'].unique()
merged.groupby(['world', 'title'])['accuracy_group'].value_counts()
#Well it is quite intuitive that  'Chest Sorter (Assessment)' in CRYSTALCAVES require more effort than 'Cart Balancer (Assessment)'. 

#Since, the former one seems to have highest count of accuracy_group=0 while the later one is having highest count of accuracy_group=3 in

#that world. Infact 'Chest Sorter (Assessment)' is having the highest count of accuracy_group=0 among all the assessments. Similarly, the

#'Bird Measurer (Assessment)' in TREETOPCITY seems to have required more effort than 'Mushroom Sorter (Assessment)' which is actually

#having the highest count of accuracy_group=3 among all other assessments. 'Cauldron Filler (Assessment)' on its own seems like a pretty 

#easy task since it is having a good number of accuracy_group=3 counts.
#Let's load the test dataset.

test_df = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')

test_df.shape
#Let's create the template for submission dataframe. We will take each unique combination of 'installation_id' and 'title' from test dataframe

#and put them into the 'combined' column of submission_df to be splitted upon later on.

temp = test_df.loc[ test_df['title'].str.contains('Assessment'), ['installation_id', 'title'] ]

temp['combined'] = temp['installation_id'].astype(str)+'_'+temp['title'].astype(str)

submission_df = pd.DataFrame( columns = ['combined', 'accuracy_group'] )

submission_df['combined'] = temp['combined'].unique()

del(temp)

submission_df[['installation_id', 'title']] = submission_df['combined'].str.split('_', expand=True)

submission_df = submission_df.drop('combined', axis=1)

submission_df = submission_df[['installation_id', 'title', 'accuracy_group']]

submission_df.head()
submission_df.shape
#The training set contains many installation_ids which never took assessments, whereas every installation_id in the test set made 

#an attempt on at least one assessment.

test_df.loc[ (test_df['installation_id']=='00abaee7') & (test_df['world']=='TREETOPCITY'), 'title' ].unique()
#The assessment information of this installation_id for TREETOPCITY is not captured in the test dataset.

test_df.loc[ (test_df['installation_id']=='00abaee7') & (test_df['world']=='MAGMAPEAK'), 'title' ].unique()
test_df.loc[ (test_df['installation_id']=='00abaee7') & (test_df['world']=='CRYSTALCAVES'), 'title' ].unique()
#Finally let's get started with the main part of the analysis. According to me the factor that should most influence the performance of a

#student in assessments is whether he has properly understood the videos or lessons provided prior to that. So, we will basically determine

#the maximum accuracy_group of a child corresponding to an assessment based on whether he has attended the previous lessons or not.

#For that we will create a dataframe for each world with all the titles under the world as the columns, installation_id as the index.

#The columns will be binary as to whether he has attended that lesson or not, except the Assessment columns which will contain the maximum

#accuracy_group of that id for that assessment.
#Get the maximum accuracy group of each id for each assessment in group1.

group1 = merged.groupby( by=['title', 'installation_id'] )['accuracy_group'].max()

group1
#Create the temporary dataframe.

columns = list( train_df.loc[ train_df['world']=='TREETOPCITY', 'title'].unique() )

ids = merged.loc[ merged['world']=='TREETOPCITY','installation_id'].unique()

treetop_df = pd.DataFrame( columns = columns, index=list( ids ) )

treetop_df.head(2)
#Get the titles of a world attended by each id in group2.

group2 = train_df.loc[ train_df['world']=='TREETOPCITY', ['installation_id', 'title']].groupby( by='installation_id' )

group2    
#Fill in the temporary dataframe with values from group1 and group2.

for i in ids:

    gg = group2.get_group(i)

    titles = gg['title'].unique()

    for t in titles:

        treetop_df.loc[i, t]=1

    assessments = list( merged.loc[(merged['installation_id']==i) & (merged['world']=='TREETOPCITY'), 'title'].unique() )

    for a in assessments:

        treetop_df.loc[i, a] = group1[(a, i)]

treetop_df.head()            
#Fill the null values with the minimum in the assessment columns and 0 elsewhere.

treetop_df.loc[ treetop_df['Bird Measurer (Assessment)'].isnull(), 'Bird Measurer (Assessment)' ] = min( treetop_df['Bird Measurer (Assessment)'] )

treetop_df.loc[ treetop_df['Mushroom Sorter (Assessment)'].isnull(), 'Mushroom Sorter (Assessment)' ] = min( treetop_df['Mushroom Sorter (Assessment)'] )

treetop_df = treetop_df.fillna(0)         

treetop_df.head()  
treetop_df['Bird Measurer (Assessment)'].value_counts()
#For modelling I will be using Bernoulli Naive Bayes as it deals with independent binary variables.
#Train the models on the temporary dataframe.

from sklearn.naive_bayes import BernoulliNB

bnb_tt1 = BernoulliNB()

Y_cols = ['Bird Measurer (Assessment)', 'Mushroom Sorter (Assessment)']

X_cols = list( set(treetop_df.columns) - set(Y_cols) )

X_train = treetop_df[ X_cols ]

Y_train = treetop_df['Mushroom Sorter (Assessment)']

bnb_tt1.fit(X_train, Y_train)

bnb_tt2 = BernoulliNB()

Y_train = treetop_df['Bird Measurer (Assessment)']

bnb_tt2.fit(X_train, Y_train)
#Delete the temporary dataframe and build another similar one from the test dataset on which the model will be tested.

del(treetop_df)

ids = test_df.loc[ (test_df['title']=='Bird Measurer (Assessment)') | (test_df['title']=='Mushroom Sorter (Assessment)'),'installation_id'].unique()

filtered_test = pd.DataFrame( columns = test_df.columns )

for i in ids:

    temp = test_df[ test_df['installation_id']==i ]

    filtered_test = pd.concat( [filtered_test, temp], axis=0, ignore_index=True )

treetop_test = pd.DataFrame( columns = columns, index=list( ids ) )

group2 = filtered_test.loc[ filtered_test['world']=='TREETOPCITY', ['installation_id', 'title']].groupby( by='installation_id' )

for i in ids:

    gg = group2.get_group(i)

    titles = gg['title'].unique()

    for t in titles:

        treetop_test.loc[i, t]=1    

treetop_test = treetop_test.fillna(0)

treetop_test.head(3)        
#Predict the results and store them into the temporary dataframe build from test dataset.

X_test = treetop_test[ X_cols ]

treetop_test['Mushroom Sorter (Assessment)'] = bnb_tt1.predict(X_test)

treetop_test['Bird Measurer (Assessment)'] = bnb_tt2.predict(X_test)

treetop_test[['Mushroom Sorter (Assessment)']].head(10)
#We can accept these assumptions, the reason being suppose a child takes an assesssment and pass it after 4 or 5 attempts. But since he has 

#gone through the necessary videos or activities before taking the assessment, he feels like he is able to understand his mistakes and 

#will be able to do better the next time. So, the next time he tries he succeds at the 2nd attempt and eventually he will be able to pass

#at the very first attempt. To make it easier, we have considered the maximum accuracy_group that a child can end up with, depending on

#whether he has understood the lessons provided prior to that in terms of videos or activity. On the other hand, if a child has not gone

#through the previous lessons or feels like an assessment is too tough for him to attempt any further then he will try no longer and end up

#in the lowest accuracy_group mostly 0 or 1.
#Copy the predictions from temporary dataframe to the submission_df dataframe.

for a in ['Mushroom Sorter (Assessment)', 'Bird Measurer (Assessment)']:

    ids = submission_df.loc[ submission_df['title']==a, 'installation_id' ]

    for i in ids:

        submission_df.loc[ (submission_df['installation_id']==i) & (submission_df['title']==a), 'accuracy_group'] = treetop_test.loc[ i,a ]
#We will do same for the other 2 worlds.

#Create the temporary dataframe.

columns = list( train_df.loc[ train_df['world']=='CRYSTALCAVES', 'title'].unique() )

ids = merged.loc[ merged['world']=='CRYSTALCAVES','installation_id'].unique()

crystal_df = pd.DataFrame( columns = columns, index=list( ids ) )

#Get the titles of a world attended by each id in group2.

group2 = train_df.loc[ train_df['world']=='CRYSTALCAVES', ['installation_id', 'title']].groupby( by='installation_id' )

#Fill in the temporary dataframe with values from group1 and group2.

for i in ids:

    gg = group2.get_group(i)

    titles = gg['title'].unique()

    for t in titles:

        crystal_df.loc[i, t]=1

    assessments = list( merged.loc[(merged['installation_id']==i) & (merged['world']=='CRYSTALCAVES'), 'title'].unique() )

    for a in assessments:

        crystal_df.loc[i, a] = group1[(a, i)]

#Fill the null values with the minimum in the assessment columns and 0 elsewhere.        

crystal_df.loc[ crystal_df['Chest Sorter (Assessment)'].isnull(), 'Chest Sorter (Assessment)' ] = min( crystal_df['Chest Sorter (Assessment)'] )

crystal_df.loc[ crystal_df['Cart Balancer (Assessment)'].isnull(), 'Cart Balancer (Assessment)' ] = min( crystal_df['Cart Balancer (Assessment)'] )

crystal_df = crystal_df.fillna(0)

#Train the models on the temporary dataframe.

bnb_tt1 = BernoulliNB()

Y_cols = ['Chest Sorter (Assessment)', 'Cart Balancer (Assessment)']

X_cols = list( set(crystal_df.columns) - set(Y_cols) )

X_train = crystal_df[ X_cols ]

Y_train = crystal_df['Cart Balancer (Assessment)']

bnb_tt1.fit(X_train, Y_train)

bnb_tt2 = BernoulliNB()

Y_train = crystal_df['Chest Sorter (Assessment)']

bnb_tt2.fit(X_train, Y_train)

#Delete the temporary dataframe and build another similar one from the test dataset on which the model will be tested.

del(crystal_df)

ids = test_df.loc[ (test_df['title']=='Chest Sorter (Assessment)') | (test_df['title']=='Cart Balancer (Assessment)'),'installation_id'].unique()

filtered_test = pd.DataFrame( columns = test_df.columns )

for i in ids:

    temp = test_df[ test_df['installation_id']==i ]

    filtered_test = pd.concat( [filtered_test, temp], axis=0, ignore_index=True )

crystal_test = pd.DataFrame( columns = columns, index=list( ids ) ) 

group2 = filtered_test.loc[ filtered_test['world']=='CRYSTALCAVES', ['installation_id', 'title']].groupby( by='installation_id' )

for i in ids:

    gg = group2.get_group(i)

    titles = gg['title'].unique()

    for t in titles:

        crystal_test.loc[i, t]=1  

crystal_test = crystal_test.fillna(0)

crystal_test.head(3) 
#Predict the results and store them into the temporary dataframe build from test dataset.

X_test = crystal_test[ X_cols ]

crystal_test['Cart Balancer (Assessment)'] = bnb_tt1.predict(X_test)

crystal_test['Chest Sorter (Assessment)'] = bnb_tt2.predict(X_test)

crystal_test[['Cart Balancer (Assessment)']].head(10)
#Copy the predictions from temporary dataframe to the submission_df dataframe.

for a in ['Cart Balancer (Assessment)', 'Chest Sorter (Assessment)']:

    ids = submission_df.loc[ submission_df['title']==a, 'installation_id' ]

    for i in ids:

        submission_df.loc[ (submission_df['installation_id']==i) & (submission_df['title']==a), 'accuracy_group'] = crystal_test.loc[ i,a ]
#We will do same for 'MAGMAPEAK' world.

#Create the temporary dataframe.

columns = list( train_df.loc[ train_df['world']=='MAGMAPEAK', 'title'].unique() )

ids = merged.loc[ merged['world']=='MAGMAPEAK','installation_id'].unique()

magma_df = pd.DataFrame( columns = columns, index=list( ids ) )

#Get the titles of a world attended by each id in group2.

group2 = train_df.loc[ train_df['world']=='MAGMAPEAK', ['installation_id', 'title']].groupby( by='installation_id' )

#Fill in the temporary dataframe with values from group1 and group2.

for i in ids:

    gg = group2.get_group(i)

    titles = gg['title'].unique()

    for t in titles:

        magma_df.loc[i, t]=1

    assessments = list( merged.loc[(merged['installation_id']==i) & (merged['world']=='MAGMAPEAK'), 'title'].unique() )

    for a in assessments:

        magma_df.loc[i, a] = group1[(a, i)]

#Fill the null values with the minimum in the assessment columns and 0 elsewhere.        

magma_df.loc[ magma_df['Cauldron Filler (Assessment)'].isnull(), 'Cauldron Filler (Assessment)' ] = min( magma_df['Cauldron Filler (Assessment)'] )

magma_df = magma_df.fillna(0)

#Train the models on the temporary dataframe.

bnb_tt = BernoulliNB()

Y_cols = ['Cauldron Filler (Assessment)']

X_cols = list( set(magma_df.columns) - set(Y_cols) )

X_train = magma_df[ X_cols ]

Y_train = magma_df['Cauldron Filler (Assessment)']

bnb_tt.fit(X_train, Y_train)

#Delete the temporary dataframe and build another similar one from the test dataset on which the model will be tested.

del(magma_df)

ids = test_df.loc[ test_df['title']=='Cauldron Filler (Assessment)','installation_id'].unique()

filtered_test = pd.DataFrame( columns = test_df.columns )

for i in ids:

    temp = test_df[ test_df['installation_id']==i ]

    filtered_test = pd.concat( [filtered_test, temp], axis=0, ignore_index=True )

magma_test = pd.DataFrame( columns = columns, index=list( ids ) )

group2 = filtered_test.loc[ filtered_test['world']=='MAGMAPEAK', ['installation_id', 'title']].groupby( by='installation_id' )

for i in ids:

    gg = group2.get_group(i)

    titles = gg['title'].unique()

    for t in titles:

        magma_test.loc[i, t]=1  

magma_test = magma_test.fillna(0)

magma_test.head(3)        
#Predict the results and store them into the temporary dataframe build from test dataset.

X_test = magma_test[ X_cols ]

magma_test['Cauldron Filler (Assessment)'] = bnb_tt.predict(X_test)

magma_test[['Cauldron Filler (Assessment)']].head(10)
#Copy the predictions from temporary dataframe to the submission_df dataframe.

a = 'Cauldron Filler (Assessment)'

ids = submission_df.loc[ submission_df['title']==a, 'installation_id' ]

for i in ids:

    submission_df.loc[ (submission_df['installation_id']==i) & (submission_df['title']==a), 'accuracy_group'] = magma_test.loc[ i,a ]                                                           
#If all the predictions are made none of columns should contain null values.

submission_df.isnull().sum()
#Store the submission dataframe to a csv file.

submission_df.to_csv('submission.csv', index=False)
#Please don't forget to upvote if you atleast appreciate my efforts, leave alone the scores or results. It helps a lot! 

#Also feel free to comment. :)