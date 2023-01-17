#Creates a weighted, directed graph of all of the Clinton emails of the type
# email_sender ------weight-------> email_recipient
# where "email_sender" and "email_recipient" are nodes and
# weight is the weight of the edge, defined
# as the number of emails sent by email_sender to email_recipient
# for example, .....

#first the imports

import pandas as pd
import networkx as nx
import numpy as np
from collections import Counter, defaultdict
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt

#setup the plotting
pylab.rcParams['figure.figsize'] = 16, 16


# read the main data source
emails = pd.read_csv("../input/Emails.csv")

#cleanup the names in the From and To fields
with open("../input/Aliases.csv") as f:
    file = f.read().split("\r\n")[1:] #skip the header line
    aliases = {}
    for line in file:
        line = line.split(",")
        aliases[line[1]] = line[2]

with open("../input/Persons.csv") as f:
    file = f.read().split("\r\n")[1:] #skip header line
    persons = {}
    for line in file:
        line = line.split(",")
        persons[line[0]] = line[1]
        
def resolve_person(name):
    name = str(name).lower().replace(",","").split("@")[0]
    #print(name)
    #correct for some of the common people who are resolved to several different
    # names by the given Aliases.csv file:  Cheryl Mills, Huma Abedin, Jake Sullivan
    # and Lauren Jiloty
    # Also convert "h" and variations to Hillary Clinton
    if ("mills" in name) or ("cheryl" in name) or ("nill" in name) or ("miliscd" in name) or ("cdm" in name) or ("aliil" in name) or ("miliscd" in name):
        return "Cheryl Mills"
    elif ("a bed" in name) or ("abed" in name) or ("hume abed" in name) or ("huma" in name) or ("eabed" in name):
        return "Huma Abedin"
    #elif (name == "abedin huma") or (name=="huma abedin") or (name=="abedinh"): 
    #    return "Huma Abedin"
    elif ("sullivan" in name)  or ("sulliv" in name) or ("sulliy" in name) or ("su ii" in name) or ("suili" in name):
        return "Jake Sullivan"
    elif ("iloty" in name) or ("illoty" in name) or ("jilot" in name):
        return "Lauren Jiloty"
    elif "reines" in name: return "Phillip Reines"
    elif (name == "h") or (name == "h2") or ("secretary" in name) or ("hillary" in name) or ("hrod" in name):
        return "Hillary Clinton"
    #fall back to the aliases file
    elif str(name) == "nan": return "Redacted"
    elif name in aliases.keys():
        return persons[aliases[name]]
    else: return name
    
emails.MetadataFrom = emails.MetadataFrom.apply(resolve_person)
emails.MetadataTo = emails.MetadataTo.apply(resolve_person)

#Extract the to: from: and Raw body text from each record

From_To_RawText = []
temp = zip(emails.MetadataFrom,emails.MetadataTo,emails.RawText)
for row in temp:
    From_To_RawText.append(((row[0],row[1]),row[2]))

#Create a dictionary of all edges, i.e. (sender, recipient) relationships 
# and store the individual email text in a list for each key
From_To_allText = defaultdict(list)
for people, text in From_To_RawText:
    From_To_allText[people].append(text)
len(From_To_allText.keys()), len(From_To_RawText)

#Set the weights of each directed edge equal to the number of emails 
# (number of raw text documents) associated with that edge
edges_weights = [[key[0], key[1], len(val)] for key, val in From_To_allText.items()]
edge_text = [val for key, val in From_To_allText.items()]

emails.columns
#emails.describe()
emails.describe()
emails.shape
emails.dtypes
x = emails[['MetadataDateSent']].values
x
date, hour, mins, secs = x.split(':')