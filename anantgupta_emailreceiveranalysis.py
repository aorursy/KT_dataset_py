# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import networkx as nx
import re
import os
import pandas as pd
import numpy as np
df_aliases = pd.read_csv('../input/Aliases.csv', index_col=0)
df_emails = pd.read_csv('../input/Emails.csv', index_col=0)
df_email_receivers = pd.read_csv('../input/EmailReceivers.csv', index_col=0)
df_persons = pd.read_csv('../input/Persons.csv', index_col=0)
# Creating GRAPHS for emails sent
# First we will have to cleanse the MetadataTo column
emaildata=df_emails

# Make the entire data lower case
emaildata.MetadataFrom=emaildata.MetadataFrom.str.lower()
mask = emaildata['MetadataFrom'].str.len() > 2
emaildata = emaildata.loc[mask]
emaildata=emaildata.dropna(subset=['MetadataFrom'])

def getValidRecipient(x):
	print(x)
	y=x.split(';')
	for recipient in y:
		if len(str(recipient)) > 2:
			return cleanRecipient(recipient)

def cleanRecipient(x):
	#print("Before cleaning " + x)
	# Remove text after the @ sign
	x = re.sub(r"@.*$", "", x)
	
	# Also we need to remove single characters in the names of the people
	x=re.sub(r"\s[a-zA-Z0-9]{1}\s"," ",x)
	x=re.sub(r"\s[a-zA-Z0-9]{1}$","",x)
	x=re.sub(r"^[a-zA-Z0-9]{1}\s","",x)
	
	# We need to replace the , with space for uniformity
	x=re.sub(r","," ",x)
	x=re.sub(r"\s+"," ",x)
	
	# Returning the final value. We can further improve on this function
	#print("After cleaning " + x)
	return x
emaildata['fromaddress']=map(getValidRecipient,emaildata.MetadataFrom)

SentGraph = pd.pivot_table(emaildata, index=['fromaddress'], aggfunc=np.sum)
SentGraphTop10 = SentGraph.sort_values(by='SenderPersonId')
SentGraphTop10 = SentGraphTop10.tail(10)
# Create a node for Hillary
EmailReceived = nx.DiGraph()
EmailReceived.add_node("Hillary",label="hillary")

for x in SentGraphTop10.index.values:
	EmailReceived.add_edge("Hillary",x,label=x)

import matplotlib.pyplot as plt
graphObject=nx.shell_layout(EmailReceived)
nx.draw_networkx_nodes(EmailReceived,graphObject)
nx.draw_networkx_edges(EmailReceived,graphObject)
nx.draw_networkx_labels(EmailReceived, graphObject)
plt.show()