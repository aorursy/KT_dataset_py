# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

!pip install zeep

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import altair as alt

from altair import Chart, X, Y, Axis, SortField

import zeep

from collections import OrderedDict



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from zeep import Client

bill_list = []  #initializes my empty bill list



client = Client('http://wslwebservices.leg.wa.gov/LegislationService.asmx?WSDL') #sets up the client object in Zeep with data from WA State SOAP API

result_house = client.service.GetLegislationGovernorSigned("2017-18", "House") #grabs house originated legislation for the 2017-2018 session from the SOAP endpoint

result_senate = client.service.GetLegislationGovernorSigned("2017-18", "Senate") #grabs senate originated legislation for the 2017-2018 session from the SOAP endpoint



house_dict = zeep.helpers.serialize_object(result_house) #creates a variable that holds the result of this zeep helper which turns the zeep object into a standard python dictionary

senate_dict = zeep.helpers.serialize_object(result_senate)



#print(type(house_dict[0]))

for bill in house_dict: #iterates through each item in the dictionary

    bill_list.append(bill["BillId"]) #dumps the bill ID value into a list.



for bill in senate_dict:

    bill_list.append(bill["BillId"]) #same thing but does it for the Senate too 









    





    



    

sponsors_dict = {} #intializes my sponsors dict



for bill in bill_list: #iterates through my list

    legislator = client.service.GetSponsors("2017-18", bill) #grabs the sponsor dict from the API

    name = legislator[0].Name #grabs the primary sponsor from the sponsor dict entry for that bill

    if name in sponsors_dict: #if the name is already in the dict increment the pass count by 1

        sponsors_dict[name] = sponsors_dict[name] + 1

    else: 

        sponsors_dict[name] = 1 #otherwise add that sponsor to the dict and set the pass count to 1.

best_sponsor = pd.DataFrame.from_dict(sponsors_dict, orient ="index", columns = ["count"]) #creates my data frame with a column label.



alt.Chart(best_sponsor.reset_index()).mark_bar().encode( #creates my chart

    x = X("index", sort = "-y"), #sets my X axis to my index and sorts it in descending order based on the y value

    y = Y("count"), #sets my y axis as the count 

    color = alt.value("green") #sets the color to green

)






