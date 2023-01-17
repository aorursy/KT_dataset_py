# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
my_data= pd.read_csv('../input/data.csv')

my_data.info()

my_data.head()
my_data.columns=my_data.columns.str.lower().str.replace(' ','_')
my_data.head()
my_data.channel_name.value_counts()
yourChannel=input("which channel are you looking for?\n")
uniDict= my_data.channel_name.unique()

uniDict=pd.DataFrame(uniDict)

uniDict.rename(columns={0: 'channel_name'}, inplace=True)

uniDict["jaccardValue"]=0



class advisor:

    "give advice and jaccard similarity score accrording to your input"

    def __init__(self,channel):  

        "attributes"

        #"__init__" is a reserved method in python classes. It is known as a constructor in OOP concepts. 

        #This method called when an object is created from the class and it allows the class to initialize the attributes of a class.

        

        self.channel=channel

        

    

    def get_jaccard_sim(self):

        "this func use your input and calculate jaccard similarity score"

        a = set(self.channel.lower())

        

        for row_index,value in uniDict.iterrows():

            b = set(value[0].lower())

            c = a.intersection(b)

            uniDict.loc[row_index,'jaccardValue'] = (len(c)) / (len(a) + len(b) - len(c))

        

        maxJaccard = uniDict['jaccardValue'].max()

        katVar=uniDict.loc[uniDict['jaccardValue']== maxJaccard,["channel_name"]]

        channel= list(katVar['channel_name'].str.lower())            

        filChan = my_data[(my_data['channel_name'].str.lower() ==channel[0] )]

    

        if  filChan.empty  :

            print('Lütfen verilen kategori doğru giriniz!\n')

        else:

            print("Jaccard_Score:",maxJaccard,'\nChoosen Channel:\n',filChan.channel_name ,"\nDetail:\n",filChan)

          



        
my_class=advisor(yourChannel)

my_class.get_jaccard_sim()