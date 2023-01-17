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
#taken and adapted from https://towardsdatascience.com/build-your-first-chatbot-using-python-nltk-5d07b027e727

#next should with ML training



from nltk.chat.util import Chat, reflections



pairs = [

    [

        r"my name is (.*)",

        ["Hello %1, How are you today ?",]

    ],

     [

        r"what is your name ?",

        ["My name is Chatty and I'm a chatbot ?",]

    ],

    [

        r"how are you ?",

        ["I'm doing good\nHow about You ?",]

    ],

    [

        r"sorry (.*)",

        ["Its alright","Its OK, never mind",]

    ],

    [

        r"i'm (.*) doing good",

        ["Nice to hear that","Alright :)",]

    ],

    [

        r"hi|hey|hello",

        ["Hello", "Hey there",]

    ],

    [

        r"(.*) age?",

        ["I'm a computer program dude\nSeriously you are asking me this?",]

        

    ],

    [

        r"what (.*) want ?",

        ["Make me an offer I can't refuse",]

        

    ],

    [

        r"(.*) created ?",

        ["Nagesh created me using Python's NLTK library ","top secret ;)",]

    ],

    [

        r"(.*) (location|city) ?",

        ['Chennai, Tamil Nadu',]

    ],

    [

        r"how is weather in (.*)?",

        ["Weather in %1 is awesome like always","Too hot man here in %1","Too cold man here in %1","Never even heard about %1"]

    ],

    [

        r"i work in (.*)?",

        ["%1 is an Amazing company, I have heard about it. But they are in huge loss these days.",]

    ],[

        r"(.*)raining in (.*)",

        ["No rain since last week here in %2","Damn its raining too much here in %2"]

    ],

    [

        r"how (.*) health(.*)",

        ["I'm a computer program, so I'm always healthy ",]

    ],

    [

        r"(.*) (sports|game) ?",

        ["I'm a very big fan of Football",]

    ],

    [

        r"who (.*) sportsperson ?",

        ["Messy","Ronaldo","Roony"]],

    [

        r"who (.*) (moviestar|actor|waiter)?",

        ["Brad Pitt"]],

    [

        r"(quit|exit|ciao)",

        ["Bye take care. See you soon :) ","It was nice talking to you. See you soon :)"]],

]

def chatty():

    print("Hi, I'm Chatty and I chat alot ;)\nPlease type lowercase English language to start a conversation. Type quit or exit to leave ") #default message at the start    

    chat = Chat(pairs, reflections)

    chat.converse(quit="quit")

if __name__ == "__main__":

    chatty()