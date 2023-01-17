import numpy as np

import pandas as pd

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestRegressor

#Load the dataset

train = pd.read_csv('../input/train_1.csv').fillna(0)

#Save the dates for future use

pages = train['Page'].copy()

dates = train.columns

#Drop the page coloumn for now as we are training individual model

train.drop(['Page'],inplace=True,axis=1)

#Stack the coloumns as rows

df_train = train.stack().reset_index(level=0, drop=True).reset_index()
#set the coloumn names to date and number of visits

df_train.columns = ['Date','number_of_visits']
#Let's see how the data looks like



df_train.head()
#Create the pages data frmae



pages_repeat = pd.DataFrame(np.repeat(pages,550))

#reset the index for not messing with it

df_train.reset_index(drop=True,inplace=True)

pages_repeat.reset_index(drop=True,inplace=True)
#Add the page coloumn again to train data  as we require them to group it

df_train['Page'] = pages_repeat.Page.copy()
#now group by Page to run individual model for each of them



grouped = df_train.groupby('Page')

#SMAPE calculate for each page

#This one is from CPMP,Thanks ;-)

def smape(y_true, y_pred,page_name):

    denominator = (np.abs(y_true) + np.abs(y_pred))

    diff = np.abs(y_true - y_pred) / denominator

    diff[denominator == 0] = 0.0

    print ("SMAPE score for "+str(np.asarray(page_name))+": "+str(200 * np.mean(diff)))
#Prepare our test data from January 1,2017 to November 10,2017



#generate dates between a range

date_test =[]

for dayes in pd.date_range('20170101','20171110'):

    date_test.append(dayes.strftime('%Y-%m-%d'))

    date_frame = pd.DataFrame(np.asarray(date_test))

    #Save the date_frame for future use

    date_append = date_frame

    date_frame.columns = ['Date']

    #Expand the date coloumn 

    date_frame = date_frame.Date.str.split('-',expand=True).astype(int)

    date_frame.columns = ['Year','Month','Day']

    #add the quarter to dataframe

    date_frame['Quarter'] = (date_frame.Month-1)//3

    #drop the year

    date_frame.drop(['Year'],inplace=True,axis=1)
def process_groups(test_template):

    #Initialize a dataframe to combine all the page predictions

    final_predictions = pd.DataFrame(columns=['Visits','Page','Date'])

    counter = 0

    #preprocess each group

    for group in grouped.groups.keys():

        #create a temp frame

        group_predictions = pd.DataFrame(columns=['Visits','Page'])

        data_train = grouped.get_group(group)

        #Expand the Date coloumn

        data_train_date= data_train.Date.str.split('-',expand=True).astype(int)

        data_train_date.columns =['Year','Month','Day']

        #concatenate to expanded date to data_train

        data_train = pd.concat([data_train,data_train_date],axis=1)

        targets = data_train.number_of_visits.copy()

        #Save the page name for future

        pages_frame = np.unique(data_train.Page.values)      

        #drop the Year and number_of_visits,Page coloumn

        data_train.drop(['Year','number_of_visits','Page','Date'],inplace=True,axis=1)

        #Add the quarter date

        data_train['Quarter'] = (data_train.Month-1)//3

        #KFold cross validation

        kfold = KFold(5)

        predictions =[]

        

    

        data_train = np.asarray(data_train)

        targets = np.asarray(targets)

        

        for train_index,test_index in kfold.split(data_train,targets):

            rf = RandomForestRegressor(n_estimators=100,max_depth=4)

            X_train, X_test = data_train[train_index], data_train[test_index]

            y_train, y_test = targets[train_index], targets[test_index]

            rf.fit(X_train,y_train)

            y_preds = rf.predict(X_test)

            #Calculate the SMAPE

            smape(y_test,y_preds,pages_frame)

            #predict on test_data

            predictions.append(rf.predict(test_template))

        #Average the results from cross validation

        pred_average = np.mean(np.asarray(predictions),0)

        group_predictions['Page'] = np.repeat(pages_frame,len(test_template))

        group_predictions['Visits'] = pred_average

        group_predictions['Date'] = date_append

        #Add the dates to the final predictions

        final_predictions = final_predictions.append(group_predictions)

        #Run only for three pages else it will run more than 3+ days :-(

        #Comment the below line to run for all the pages

        counter = counter+1

        if counter > 2:

            

            return final_predictions

    #Uncomment the line to run for all the pages

    #Caution! : This will take long time

    #return final_predictions

        

    
check_predictions=process_groups(date_frame)




check_predictions['Visits'] = check_predictions.Visits.round()
#Check the predictions 

check_predictions.head()