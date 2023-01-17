import pandas as pd

import time   

start = time.clock() #_________________ measure efficiency timing





# read data

test = pd.read_csv('../input/test.csv',encoding='utf8')

lentest=len(test)

train = pd.read_csv('../input/train.csv',encoding='utf8')

lentrain=len(train)

#train = pd.read_csv('c:/py/trainQ.csv',encoding='utf8')[:30000]

print(train.head(10))

train.fillna(value='leeg',inplace=True)







end = time.clock()

print('open:',end-start)
#make one big datafile

total=list(set( list(set(train['question1']))+list(set(train['question2']))+list(set(test['question1']))+list(set(test['question2'])) ))

del train,test

total=pd.DataFrame(total)

print(total.head())

print(total.shape)





print((lentest*2.0+lentrain*2.0)/len(total)*1.00)

end = time.clock()

print('open:',end-start)
print((lentest*2.0+lentrain*2.0)/len(total)*1.00)
total.columns = ['Q']

#print(total)

for xi in range(2, 10 ):

    print(xi,total.ix[xi]['Q'])

    
import urllib2

from urllib2 import Request, urlopen, URLError



testHTTP=[]

for xi in range(2,20):

    Q1=total.ix[xi].Q

    Q2='https://www.google.com/search?q='+Q1.replace(" ", "+").replace("?", "?+Quora")

    Q3='https://www.quora.com/'+Q1.replace(" ", "-").replace("?", "")

    print(Q2)



    # Add your headers

    headers = {'User-Agent' : 'Mozilla 5.10'}



    # Create the Request. 

    request = urllib2.Request(Q2.encode('ascii', 'ignore'), None, headers)



    # Getting the response

    response = urllib2.urlopen(request)

    Q_google = response.read()

    Q_google = unicode(Q_google, 'utf-8')

    # Print the headers

    #print(Q_google.body)

    print ( Q_google.find(Q3)    )

        # everything is fine