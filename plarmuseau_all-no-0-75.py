import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np #

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        if filename.find("rain_f")>0:

            print('reading train')            

            train=pd.read_csv(os.path.join(dirname, filename) )#,error_bad_lines=False)#,names=['id','body','headline']

        if filename.find("est")>0:

            print('reading test')

            test=pd.read_excel(os.path.join(dirname, filename) )#,error_bad_lines=False )

        if filename.find("_score")>0:

            print('reading rail')            

            rail=pd.read_csv(os.path.join(dirname, filename),error_bad_lines=False)#,names=['id','body','headline']

        if filename=="train.csv":

            print('reading news')            

            news=pd.read_csv(os.path.join(dirname, filename),delimiter=',',header=0,names=['class','headline','body'])

test
test['Churn']='No'

test[['customerID','Churn']].to_csv('submitNO.csv')