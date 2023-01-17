import os
import pandas as pd
os.chdir('/home/aplstudent/Max_Dickinson')
def loaddata():
        data = pd.read_csv('train_users_2.csv')
        datatest  = pd.read_csv('test_users.csv')
        return [data , datatest]
data , datatest = loaddata()
def dayssince(x):
        import matplotlib.dates as dates
        return dates.datestr2num(x)

    ## get years and months as a number so it can be a category 

def dataAugment(data): 
    def getyear(x):
        if type(x) == float:
            return -1
        else :
            return int(x[0:4])
    def getmonth(x):
        if type(x) == float:
            return -1
        else :
            return int(x[5:7])
    def getday(x):
        if type(x) == float:
            return -1
        else :
            return int(x[8:11])
    def getorder(x):
        return( (x[0:4],x[5:7],x[8:11]) )
    def getdate(x):
        return datetime.date(x).toordinal()
    def dayssince(x):
        import matplotlib.dates as dates
        return dates.datestr2num(x)
    data['year_created'] = data['date_account_created'].apply(getyear)
    data['month_created'] = data['date_account_created'].apply(getmonth)
    data['date_created'] = data['date_account_created'].apply(getday)
    data['year_booked'] = data['date_first_booking'].apply(getyear)
    data['month_booked'] = data['date_first_booking'].apply(getmonth)
    data['day_booked'] = data['date_first_booking'].apply(getday)
    data['num_created'] = data['date_account_created'].apply(dayssince)
    data['num_booked'] = data['date_first_booking'].dropna().apply(dayssince) #subset of data
    return data
out= dataAugment(data)
import numpy as np
def createohe(data):
    ohe = pd.get_dummies(data[['date_created', 'month_created', 'month_booked' , 'day_booked' , 'gender' , 'signup_method', 'signup_flow' , 'language' , 'affiliate_channel' , 'affiliate_provider' , 'first_affiliate_tracked', 'signup_app', 'first_device_type' , 'first_browser']])
    #add age and number of columns
    age = pd.concat((data[data['age'] < 100] , data[data['age']>14] ) , join = 'inner' , axis = 1) #subset of data
    minda = pd.concat((ohe, age.age), axis = 1 , join = 'outer' ) 
    mindat = pd.concat(( minda, data['num_created']), axis = 1) #add sequential dates
    mindata = pd.concat(( mindat, data['num_booked']), axis = 1 , join = 'outer') #add sequential date booked
    #splitintothree
    datad = mindata.dropna(axis = 1 , how= 'any') ##drop na's in all columns
    datadr = mindata.dropna(axis = 0 , how= 'any') ##drop na's in rows
    filldata = mindata.fillna(-1)
    return datad, datadr,filldata

datac, datar,filldata = createohe(out)    
