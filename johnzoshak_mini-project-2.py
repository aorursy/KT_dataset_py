# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

!pip install zeep

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#basics

import networkx as nx #library for graph making

import zeep #SOAP API client

from collections import OrderedDict #necessary for making a python object out of the Zeep stuff.

from zeep import Client # this is so I can talk to the WA Leg API

client = Client('http://wslwebservices.leg.wa.gov/LegislationService.asmx?WSDL') #Address for the API

bill_list = [] #my empty bill list



bills = client.service.GetLegislationInfoIntroducedSince("2019") #gets all bills introduced in the Wa Leg in the 2019-2020 session.



bills_dict = zeep.helpers.serialize_object(bills) #makes the zeep object a standard python object.



for bill in bills_dict: #iterates through each item in the dictionary

    senate_bill = re.search("\ASB", bill["BillId"]) #regex to make sure I only grab Senate Bills and exclude House bills. 

    if(senate_bill): #if the bill starts with SB add it to my list, otherwise ignore it. 

        bill_list.append(bill["BillId"]) #dumps the bill ID value into a list.



#ok with this I have now grabbed every bill introduced in the 2019-2020 session to date
#Grabbing sponsors 



sponsors_dict = {} #intializes my sponsors dict



for bill in bill_list: #this chunk of code seeds my sponsors dictionary with the sponsor names. 

    senators = client.service.GetSponsors("2019-20", bill) #says hey! API! get me the sponsors for every bill in my bill list

    for senator in senators: #for every item in the sponsor object...

        senator_name = senator["Name"] #store the name value in this variable...

        if senator_name not in sponsors_dict: #then check to see if that value is already in my dict...

            sponsors_dict[senator_name] = {} #if it's NOT then add it to the dict as an empty object. 

            

#print(sponsors_dict)

for bill in bill_list: #creates a sub-dict of people they have sponsored with.

    senators = client.service.GetSponsors("2019-20", bill) #Hey API get me every sponsor for my bills.

    primary_sponsor = senators[0].Name #take the first person and they are the "primary sponsor"

    for senator in senators[1:]: #skips primary sponsor so we don't count the primary sponsor.

        senator_name = senator["Name"] #stores the name in this nested loop to a senator name variable

        if senator_name in sponsors_dict[primary_sponsor]: #if the co-sponsor is in the primary sponsor object increment the value by 1. 

            sponsors_dict[primary_sponsor][senator_name] =  sponsors_dict[primary_sponsor][senator_name] + 1 

        else: 

            sponsors_dict[primary_sponsor][senator_name] = 1 #creates a sub-key in my primary sponsor object that matches the senator name and set its value to 1

            



    



    



        





import matplotlib.pyplot as plt #need matlib



G = nx.Graph() #sets my graphx graph to G 

G.add_nodes_from(sponsors_dict.keys()) #adds nodes from my sponsors dict-keys so I have nodes of all the Senators in WA State.

for senator in sponsors_dict: #loops to add weighted edges where the weights are the number of times that senator has co-sponsored with another Senator.

    for cosponsor in sponsors_dict[senator]:

        G.add_edge(senator, cosponsor, weight = sponsors_dict[senator][cosponsor])

nx.draw(G, with_labels = True, node_size = 1000) #draws my first graph which looks like a MESS... see below as to how I made this better.



import community #imports fancy community algos from graphx



parts = community.best_partition(G) #runs my graph through the best partition algo

values = [parts.get(node) for node in G.nodes()] #sets the value for each node using the algo



plt.axis("off") #turns off the axis in my graph

plt.figure(3,figsize=(20,20)) #makes my graph big enough to view reasonably... lots of trial and error here

nx.draw_networkx(G, cmap = plt.get_cmap("summer"), node_color = values, node_size = 2500, with_labels = True) #set the color scheme to summer, the color should be based on the values, size should be big, and put the label on

#plt.savefig("results/communities.png", format = "PNG") -- saves it as a PNG if you want. 
