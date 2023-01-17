# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

print("Load Packages")

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from fastai.tabular import *  #fast.ai tabular models





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

train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv') 

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')

print(train.shape)

print(test.shape)

print(submission.shape)
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
#examine data for train

train_data.dtypes

#examine data for test 

test_data.dtypes
#sort the data 

print("Sort the training data set for validation")

train_data = train_data.sort_values(by='Date', ascending=True)

train_data = train_data.reset_index(drop=True)
##

#fastLearner

#takes a train and test dataframe object and outputs the test file with predictions

#input: train and test pandas dataframe objects, size of validation set as numeric, 

#and the dep variable name 

#output: pandas dataframe object test with predictions 

##

def fastLearning(df1,df2,size,dep):

    #instantiate variables  

    train_data = df1

    test_data =df2

    val_size = size 

    path = ''

    deep_var=dep

    

    #model parameters 

    # 'ConfirmedCases'

    dep_var = deep_var

    cat_names = ['Province/State', 'Country/Region','Is_month_end',

             'Is_month_start','Is_quarter_end','Is_quarter_start','Is_year_end']

    cont_names = ['Lat', 'Long','Year', 

              'Month', 'Week', 

              'Day', 'Dayofweek', 

              'Dayofyear', 'Elapsed']

    

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

                           .databunch(bs=20))

    #create learner 

    learn = tabular_learner(data, layers=[300,200], emb_drop=0.04,metrics= [rmse])



    #Exploring the learning rates

    #learn.lr_find(start_lr = 1e-03,end_lr = 1e+02, num_it = 100)

    #learn.lr_find()

    #learn.recorder.plot()

    

    #Fitting data and training the network

    learn.fit_one_cycle(10)



    #save stage 1 learning 

    learn.save('stage-1')



    #unfreeze the learner

    learn.unfreeze()



    #Fitting data and training the network

    learn.fit_one_cycle(2)



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

#train_data=train_data[train_data.state_country=="BeijingChina"]

#test_data=test_data[test_data.state_country=="BeijingChina"]



#test_data.head()
################# Iterate Tablular Learner ##############

#make state/country column in both train and test 

#train_data["state_country"] = train_data["Province/State"].astype(str) + train_data["Country/Region"].astype(str)

#test_data["state_country"] = test_data["Province/State"].astype(str) + test_data["Country/Region"].astype(str)

#test_data.head()



#ensure both have state_country column 

categories=test_data.groupby("Country/Region")["Country/Region"].count() #true 

categories=pd.DataFrame(categories)

#print(categories.index)



#create holding dataframe for test

confirmed_holding = pd.DataFrame()

#fatalities_holding = pd.DataFrame()

#train_data.head(50)
#for each state_country run the main program 

for i in categories.index:

    #print name for testing 

    #print(i)

    #subset both train and testing data 

    train_temp=train_data[train_data["Country/Region"]==i]

    test_temp=test_data[test_data["Country/Region"]==i]

    

    #run main AI function for the portion of data 

    confirmed_file=fastLearning(df1=train_temp,df2=test_temp,size=.1,dep='ConfirmedCases_old')

    fatalities_file=fastLearning(df1=train_temp,df2=test_temp,size=.1,dep='Fatalities_old')

    

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