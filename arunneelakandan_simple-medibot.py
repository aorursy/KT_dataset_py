# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import random

import time

import json



Action1={"user":("I'm not feeling well", "I'm sick","i'm sick","i am sick","I am sick","I AM SICK"

                 "Feeling sick","Feeling sick", "can you help me?", "feeling sick"),

         "bot":("can you tell me symptoms?", "can you elaborate!", "Please tell me in brief",

                "what did you feel?" )}

Action2={"user":("Bye", "thanks", "Okay", "Thankyou", "See you"),

         "bot":("Bye","You're welcome","Have a nice day","See you, Bye")}



class chatterbot:

    def greet(message):

        message=input("You:")     

        if message!="no":

            flag=True

            while(flag== True):

                if message in Action1["user"]:

                    reply=random.choice(Action1["bot"])

                    print("Medibot:",reply)

                message=input("You:")

                if message not in Action1["user"]:

                    symptoms=message.split(" ")      

                    b={"symptoms":[i for i in symptoms]}

                    b=json.dumps(b)

                    print(b,"\n your symptoms are noted, thanks")

                    flag=False



print("Medibot: Hi I'm a Medibot, You can begin conversation by typing in a message and pressing enter.")

time.sleep(1)

print("Medibot: May I Help You?")

message=input("You:")

a=("no","No","NO")

if message in a:

    reply=random.choice(Action2["bot"])

    print("Medibot:",reply)

else:

    print("Medibot: What can I do?")     

    chatterbot.greet(message)

    

message=input("You:")            

if message in Action2["user"]:

    reply=random.choice(Action2["bot"])

    print("Medibot:",reply)