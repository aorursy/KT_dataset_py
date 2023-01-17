# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime, timedelta, date #Datetime for timestamp

def dateConvert(secondsStr):
    epoch = datetime(1970,1,1)
    if secondsStr == "":
        return epoch
    else:
        sec = float(secondsStr)
        return epoch + timedelta(seconds=sec)


data = pd.read_csv("../input/open-shopee-code-league-logistic/delivery_orders_march.csv",\
                   dtype={"orderid":str, "buyeraddress":str, "selleraddress":str},\
                  converters={"pick":dateConvert, "1st_deliver_attempt":dateConvert, "2nd_deliver_attempt":dateConvert})

print(data.head(15))
#function to convert two datetime into workdays
def toWorkdays(dtStart, dtEnd):
    holidays = [date(2020,3,8),\
                date(2020,3,25),\
                date(2020,3,30),\
                date(2020,3,31)]
    day = timedelta(days=1)
    res = 0
    workdays = []
    while(dtStart <= dtEnd):
        dtStart = dtStart + day
        if dtStart.date().isoweekday() == 7 or dtStart.date() in holidays:
            workdays.append(0)
        else:
            workdays.append(1)
    if workdays:
        res = sum(workdays)
    return res


data['1st_attempt_workdays'] = pd.Series(data=[ toWorkdays(start, end) for start, end in zip(data['pick'], data['1st_deliver_attempt']) ])
data['2nd_attempt_workdays'] = pd.Series(data=[ toWorkdays(start, end) for start, end in zip(data['1st_deliver_attempt'], data['2nd_deliver_attempt']) ])

print(data.head())
#Function to return SLA Location from address
def locSLA(address):
    res = ""
    locations = ["Metro Manila", "Luzon", "Visayas", "Mindanao"]
    for loc in locations:
        if loc.lower() in address.lower():
            res = loc
            break
    return res
        
data["buyer_slaloc"] = pd.Series(data = [ locSLA(address) for address in data['buyeraddress'] ])
data["seller_slaloc"] = pd.Series(data = [ locSLA(address) for address in data['selleraddress'] ])
data.head(10)
#Return SLA days limit from two sla location
def getSLA(sla_from, sla_to):
    indexer = {"Metro Manila":0, "Luzon":1, "Visayas":2, "Mindanao":3}
    sla_table = [[3,5,7,7],\
                 [5,5,7,7],\
                 [7,7,7,7],\
                 [7,7,7,7]]
    return sla_table[indexer[sla_from]][indexer[sla_to]]

data['sla_limit'] = pd.Series(data = [ getSLA(sla_from, sla_to) for sla_from, sla_to in zip(data['seller_slaloc'], data['buyer_slaloc'])])
data.head(15)
#Get final result
def getFinal(attempt1, attempt2, sla):
    if attempt2 > 3:
        return 1
    if attempt1 + attempt2 <= sla:
        return 0
    else:
        return 1
    
data['is_late'] = pd.Series(data = [getFinal(attempt1, attempt2, sla) for attempt1, attempt2, sla in zip(data['1st_attempt_workdays'], data['2nd_attempt_workdays'], data['sla_limit'])])

data.head(20)
submission = data[['orderid', 'is_late']]


submission.head(20)


submission.to_csv('submission.csv', index=False)