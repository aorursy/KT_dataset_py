import os



# in python we must import specific modules that allow us to use certain functions. 

# Here we have imported the os module which allows us to specify what to do with files, paths, and similar. functionality.



PATH="../input/caselaw-dataset-illinois/"



# we have specified the path for the location of the folder or directory containing the Illinois dataset.



os.listdir(PATH)



# now, we are outputting a list of what is included in that location.

# Click on this cell, then press the big blue Play button next to this code cell to see the output.

import lzma

import json



# lzma will help us extract the files, and json will help us work with the json data.



import pandas as pd

import numpy as np



# pandas is a powerful tool for working with data. We will see some examples of how it works below.

# numpy is a tool for working with numerical data.



from datetime import datetime



# the datetime module allows us to work with dates. 



print('imports done')



# Click on this cell, then press the big blue Play button next to this code cell.



json_path = PATH + 'text.data.jsonl.xz'



# first, I am specifying the location of the path to the json file I want to work with.



json_path



# Click on this cell, then press the big blue Play button next to this code cell. 

# You should see the output of the path of the json file below.

with open(json_path, 'rb') as in_file:

    with lzma.LZMAFile(in_file) as cases:

        for index, i in enumerate(cases):

            if index > 500:

                break

            print(json.loads(i))



# The 'if index > 500' line specifies how many json files or 'cases' to print out. 

# This file contains all the cases, so setting this number very high would take a long time to  print.

# However, you can adjust this number to experiment with how long it takes to run.

# Press the Blue Play button to the left to view the output of the file. 

# You can read through the output or copy and paste it into a text editor. My favorite is Sublime Text.

            
casedata = [[]]



# First we create an empty list space to hold the data called 'casedata'.





# Now we get down to actually processing the data.



with open(json_path, 'rb') as in_file:

    with lzma.LZMAFile(in_file) as cases:

        for index, i in enumerate(cases):

            if index > 250000:

                break

            j = json.loads(i)

            opinions = j['casebody']['data']['opinions']

            citations = j['citations']

            for opinion in opinions:

                casedata.append([opinion['author'], j['name_abbreviation'],citations[0]['cite'], j['decision_date'],opinion['type'], opinion['text']])

        casedf = pd.DataFrame(casedata, columns=['Author', 'Name', 'Citation', 'DecisionDate', 'OpinionType', 'OpinionText']) 



# You'll notice in this instance we set 'index > 250000' This will mean it takes much longer to run.

# We are also pulling out data such as citation, opinion, author, name of the case, decision date, and opinion type.

# next we are adding that data to a dataframe called casedf with columns for the data.

            

casedf.tail()



# casedf.tail() displays the last 5 records in a dataframe. 

# This will show us if the code worked and give us a sense of what the data looks like. 

# Click the Blue Play button, this code may take a few minutes to run, but at the end the last 5 records of the dataframe should display below. 
author_count = casedf.Author.value_counts()

author_count.head(10)
opinion_count = casedf.OpinionType.value_counts()

opinion_count.head(10)
AuthorDissents = casedf[(casedf.OpinionType == 'dissent')].Author.value_counts()

AuthorDissents.head()