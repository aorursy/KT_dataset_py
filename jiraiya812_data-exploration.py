# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import json



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



with open("../input/rdany_conversations_2016-01-31.json", encoding='utf-8') as data_file:

    data = json.load(data_file)



# Convert to Pandas

messages = {"hashed_chat_id": [],

            "hashed_message_id": [],

            "date": [],

            "source": [],

            "text": [],

           }

for conversation in data:

    for message in data[conversation]:

        messages["hashed_chat_id"].append(conversation)

        messages["hashed_message_id"].append(message["hashed_message_id"])

        messages["date"].append(message["date"])

        messages["source"].append(message["source"])

        messages["text"].append(message["text"])





pd_data = pd.DataFrame(messages)



print ("{0} conversations".format(len(data)))

print ("{0} messages".format(len(pd_data)))

print ("{0} messages by a human being".format(len(pd_data[pd_data["source"]=="human"])))

print ("{0} messages by a rdany".format(len(pd_data[pd_data["source"]=="robot"])))