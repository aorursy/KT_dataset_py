# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

print("Load Packages")

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.tabular import *  #fast.ai tabular models

import os, gc, pickle, copy, datetime, warnings





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

print("Print Directories")

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.

#read in training data, outcomes and testing  

print("load train, test and submission")

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv') 

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')

print(train.shape)

print(test.shape)

print(submission.shape)
#read in population data 

pop = pd.read_csv('/kaggle/input/world-population-by-age-group-2020/WorldPopulationByAge2020.csv')

pop.head()
#spread the age group column 

pop = pop.pivot(index='Location',columns='AgeGrp',values=['PopMale', 'PopFemale','PopTotal'])

pop.head()

#pop_test.shape
#flatten the multi - index 

pop.columns = pop.columns.get_level_values(0)+pop.columns.get_level_values(1)

#pop_test.head()

pop['Location_column'] = pop.index

pop.head()

#rename united states or other variables to improve match 

pop=pop.replace("United States of America", "US")

pop[pop["Location_column"]=="US"]
#combine the population file with the train and test files 

#train_ex = train

#pop_test = pop 



train = pd.merge(train, pop, how="left",left_on='Country_Region', right_on='Location_column')

test = pd.merge(test, pop, how="left",left_on='Country_Region', right_on='Location_column')

test.head()
train = train.rename(columns={'ConfirmedCases': 'ConfirmedCases_old', 'Fatalities': 'Fatalities_old'})

train.head()
test.head()
submission.head()
#separate out the first date available to both train and test 

#that is jan 22 2020 for train and march 12 2020 for test 

#first_day_train = train[train.Date == '2020-01-22']

#first_day_test = train[train.Date == '2020-03-12']



#drop (keep needed) unneeded variables in both datasets 

#first_day_train=pd.DataFrame(first_day_train,columns=["Province/State","Country/Region","ConfirmedCases","Fatalities"])

#first_day_test=pd.DataFrame(first_day_test,columns=["Province/State","Country/Region","ConfirmedCases","Fatalities"])



#change names to first day confirmed and first day fatalities 

#first_day_train.rename(columns={'ConfirmedCases': 'FirstDayConfirmed', 'Fatalities': 'FirstDayFatalities'}, inplace=True)

#first_day_test.rename(columns={'ConfirmedCases': 'FirstDayConfirmed', 'Fatalities': 'FirstDayFatalities'}, inplace=True)



#merge both datasets to add this new variable 

#train = pd.merge(train, first_day_train, on=['Province/State', 'Country/Region'])

#test = pd.merge(test, first_day_test, on=['Province/State', 'Country/Region'])



#train.head()



#train[train["Location_column"]=="US"]
#investigate missing 

train.isnull().sum()
#investigate missing 

test.isnull().sum()
#Potentially sort the training database and prepare to take a new type of validation data set

print("sort the train file")

make_date(train, 'Date')

make_date(test, 'Date')
print("delete columns that might not be useful")

#train=train.drop(['Id', 'Province/State', 'Country/Region', 'Lat', 'Long', 'Date','ConfirmedCases', 'Fatalities'],axis=1)

#test=test.drop(['ForecastId', 'Province/State', 'Country/Region', 'Lat', 'Long','Date'],axis=1)



#create date variables in train and test 

print("create time variables in both train and test")

train_data = add_datepart(train, 'Date',drop=False)

test_data = add_datepart(test, 'Date',drop=False)



#add fatalities to test

test_data['Fatalities_old'] = 0

test_data['ConfirmedCases_old'] = 0



#procedures for cleaning data 

print("set the procedures for cleaning")

procs = [FillMissing, Categorify, Normalize]
#impute new additions with median 

PopMale19_median = train['PopMale0-19'].median()

train['PopMale0-19'].fillna(PopMale19_median,inplace=True)

test['PopMale0-19'].fillna(PopMale19_median,inplace=True)



PopMale39_median = train['PopMale20-39'].median()

train['PopMale20-39'].fillna(PopMale39_median,inplace=True)

test['PopMale20-39'].fillna(PopMale39_median,inplace=True)



PopMale59_median = train['PopMale40-59'].median()

train['PopMale40-59'].fillna(PopMale59_median,inplace=True)

test['PopMale40-59'].fillna(PopMale59_median,inplace=True)



PopMale60_median = train['PopMale60+'].median()

train['PopMale60+'].fillna(PopMale60_median,inplace=True)

test['PopMale60+'].fillna(PopMale60_median,inplace=True)



PopFemale19_median = train['PopFemale0-19'].median()

train['PopFemale0-19'].fillna(PopFemale19_median,inplace=True)

test['PopFemale0-19'].fillna(PopFemale19_median,inplace=True)



PopFemale39_median = train['PopFemale20-39'].median()

train['PopFemale20-39'].fillna(PopFemale39_median,inplace=True)

test['PopFemale20-39'].fillna(PopFemale39_median,inplace=True)



PopFemale59_median = train['PopFemale40-59'].median()

train['PopFemale40-59'].fillna(PopFemale59_median,inplace=True)

test['PopFemale40-59'].fillna(PopFemale59_median,inplace=True)



PopFemale60_median = train['PopFemale60+'].median()

train['PopFemale60+'].fillna(PopFemale60_median,inplace=True)

test['PopFemale60+'].fillna(PopFemale60_median,inplace=True)



PopTotal19_median = train['PopTotal0-19'].median()

train['PopTotal0-19'].fillna(PopTotal19_median,inplace=True)

test['PopTotal0-19'].fillna(PopTotal19_median,inplace=True)



PopTotal39_median = train['PopTotal20-39'].median()

train['PopTotal20-39'].fillna(PopTotal39_median,inplace=True)

test['PopTotal20-39'].fillna(PopTotal39_median,inplace=True)



PopTotal59_median = train['PopTotal40-59'].median()

train['PopTotal40-59'].fillna(PopTotal59_median,inplace=True)

test['PopTotal40-59'].fillna(PopTotal59_median,inplace=True)



PopTotal60_median = train['PopTotal60+'].median()

train['PopTotal60+'].fillna(PopTotal60_median,inplace=True)

test['PopTotal60+'].fillna(PopTotal60_median,inplace=True)







train.isnull().sum()
#set missing regions to the countries if necessary to fill blanks 

#train.shape

#train[train.Province_State.isnull()]["Province_State"]=train["Country_Region"]

#train.head()

#values = {'Province_State': "Blank", 'Location_column': "Blank"}

#train=train.fillna(value=values,inplace=True)

#test=test.fillna(value=values)

train.loc[train['Province_State'].isnull(), 'Province_State'] = "WholeCountry"

train.loc[train['Location_column'].isnull(), 'Location_column'] = "WholeCountry"

test.loc[test['Province_State'].isnull(), 'Province_State'] = "WholeCountry"

test.loc[test['Location_column'].isnull(), 'Location_column'] = "WholeCountry"



test.isnull().sum()
train['place'] = train['Province_State']+'_'+train['Country_Region']

test['place'] = test['Province_State']+'_'+test['Country_Region']

train.place
#examine data for train

train_data.dtypes

#examine data for test 

test_data.dtypes
#sort the data 

print("Sort the training data set for validation")

train_data = train_data.sort_values(by=['place','Date'], ascending=True)

train_data = train_data.reset_index(drop=True)
train_data.tail(5)
##

#fastLearner

#takes a train and test dataframe object and outputs the test file with predictions

#input: train and test pandas dataframe objects, size of validation set as numeric, 

#and the dep variable name 

#output: pandas dataframe object test with predictions 

##

def fastLearning(df1,df2,size,dep,databunch=25,initial_cycle=100,next_cycle=100,wd_size1=1e-1,wd_size2=1e-1):

    #instantiate variables  

    train_data = df1

    test_data =df2

    val_size = size 

    path = ''

    deep_var=dep

    db_size=databunch

    learn_cycle=initial_cycle

    one_cycle=next_cycle

    wd_decay1=wd_size1

    wd_decay2=wd_size2

    

    #model parameters 

    dep_var = deep_var

    

    cat_names = ['Province_State', 'Country_Region','Is_month_end',

             'Is_month_start','Is_quarter_end','Is_quarter_start','Is_year_end']

    

    cont_names = ['Year', 

              'Month', 'Week', 

              'Day', 'Dayofweek', 

              'Dayofyear', 'Elapsed','PopMale0-19',

              'PopMale20-39','PopMale40-59','PopMale60+',

              'PopFemale0-19','PopFemale20-39','PopFemale40-59',

              'PopFemale60+','PopTotal0-19','PopTotal20-39',

              'PopTotal40-59','PopTotal60+']

    

    #Start index for creating a validation set from train

    start_indx = len(train_data) - int(len(train_data) * val_size)



    #End index for creating a validation set from train

    end_indx = len(train_data)

    

    #TabularList for Validation

    #val = (TabularList.from_df(train_data.iloc[start_indx:end_indx].copy(), path=path, cat_names=cat_names, cont_names=cont_names))

    test = (TabularList.from_df(test_data, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs))

    data = (TabularList.from_df(train_data, path=path, cat_names=cat_names, cont_names=cont_names, procs=procs)

                           .split_by_idx(list(range(start_indx,end_indx)))

                           .label_from_df(cols=dep_var)

                           .add_test(test)

                           .databunch(bs=db_size))

    #create learner 

    #learn = tabular_learner(data, layers=[15,10], wd=wd_decay,ps=[0.001,0.01], 

    #                        emb_drop=0.04,metrics= [rmse])

    learn = tabular_learner(data, layers=[300,200], wd=wd_decay1,metrics= [rmse])

    

    #Exploring the learning rates

    #learn.lr_find(start_lr = 1e-03,end_lr = 1e+02, num_it = 100)

    #learn.lr_find()

    #learn.recorder.plot()

    

    #Fitting data and training the network

    learn.fit_one_cycle(learn_cycle,wd=wd_decay1)



    #save stage 1 learning 

    learn.save('stage-1')



    #unfreeze the learner

    learn.unfreeze()



    #Fitting data and training the network

    learn.fit_one_cycle(one_cycle,wd=wd_size2)



    #apply learning model to test 

    #print("#apply learning model to test ")

    test_predictions = learn.get_preds(ds_type=DatasetType.Test)[0]



    #Converting the tensor output to a list of predicted values

    #print("Converting the tensor output to a list of predicted values")

    test_predictions = [i[0] for i in test_predictions.tolist()]



    #Converting the prediction to . a dataframe

    test_predictions = pd.DataFrame(test_predictions, columns = [dep_var+"_new"])

    

    return test_predictions

################# Iterate Tablular Learner ##############

#make state/country column in both train and test 

#train_data["state_country"] = train_data["Province/State"].astype(str) + train_data["Country/Region"].astype(str)

#test_data["state_country"] = test_data["Province/State"].astype(str) + test_data["Country/Region"].astype(str)

#test_data.head()



#ensure both have state_country column 

#categories=test_data.groupby('state_country')['state_country'].count() #true 

#categories=pd.DataFrame(categories)



#view the categories

#print(categories.index)



#subset the file

#train_data=train_data.head(66+1)



#train_data.head()

#train_data.shape
################# Iterate Tablular Learner ##############

#make state/country column in both train and test 

#train_data["state_country"] = train_data["Province/State"].astype(str) + train_data["Country/Region"].astype(str)

#test_data["state_country"] = test_data["Province/State"].astype(str) + test_data["Country/Region"].astype(str)

#test_data.head()



#ensure both have state_country column 

categories=test_data.groupby("place")["place"].count() #true 

categories=pd.DataFrame(categories)

#print(categories.index)



#create holding dataframe for test

confirmed_holding = pd.DataFrame()

#fatalities_holding = pd.DataFrame()

#train_data.head(50)
#show categories 

categories
#for each state_country run the main program 

for i in categories.index:

    #print name for testing 

    #print(i)

    #subset both train and testing data 

    train_temp=train_data[train_data["place"]==i]

    test_temp=test_data[test_data["place"]==i]

    

    #run main AI function for the portion of data 

    confirmed_file=fastLearning(df1=train_temp,df2=test_temp,

                                size=.05,dep='ConfirmedCases_old')

    

    fatalities_file=fastLearning(df1=train_temp,df2=test_temp,

                                 size=.05,dep='Fatalities_old')

    

    #make test file 

    #test_temp = test_temp.assign(pd.Series(Fatalities_old_new=fatalities_file["Fatalities_old_new"]))

    fatalities_file = fatalities_file.set_index(test_temp.index)

    test_temp["Fatalities_old_new"] = fatalities_file



    confirmed_file = confirmed_file.set_index(test_temp.index)

    test_temp["Confirmed_old_new"] = confirmed_file

    

    #append output file to the holding dataframe 

    confirmed_holding=pd.concat([confirmed_holding,test_temp],ignore_index=True)

    #fatalities_holding=pd.concat([fatalities_holding,fatalities_file],ignore_index=True)

    

     



#ensure test and holding dataframe are the same    

#holding.shape==test_data.shape
confirmed_holding
#make submission file 

final=confirmed_holding[["ForecastId","Confirmed_old_new","Fatalities_old_new"]]

final = final.rename(columns={'Confirmed_old_new': 'ConfirmedCases', 'Fatalities_old_new': 'Fatalities'})

final.to_csv('submission.csv',index=False)

final.head()

#make test file 

confirmed_holding = confirmed_holding.rename(columns={'Confirmed_old_new': 'ConfirmedCases', 'Fatalities_old_new': 'Fatalities'})

del confirmed_holding['ConfirmedCases_old']

del confirmed_holding['Fatalities_old']

confirmed_holding.to_csv('complete_test.csv',index=False)