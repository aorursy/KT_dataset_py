import json

import re # Regular expressions

import os # To walk through the data files provided

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Code provided to flatten Python arrays into the format required for

# submission.



# Convert the array format to the submission format

def flattener(pred):

    str_pred = str([row for row in pred])

    str_pred = str_pred.replace(', ', '')

    str_pred = str_pred.replace('[[', '|')

    str_pred = str_pred.replace('][', '|')

    str_pred = str_pred.replace(']]', '|')

    return str_pred
# We only need to look at the problems in the test directory.

testDirectory = "/kaggle/input/abstraction-and-reasoning-challenge/test/"
# Function to read a task file and return the parsed data

def readTaskFile(filename):

    #print("Reading file: "+filename)

    

    # Open the file

    f = open(filename, "r")

    

    # Parse the JSON

    data = json.loads(f.read())

    

    # Add in an 'id' that's extracted from the filename

    data["id"] = re.sub("(.*/)|(\.json)", "", filename)

    

    # Close the file

    f.close()

    

    # Return the parsed data

    return data
# Quick test to see that our readTaskFile function is working.

# Note the addition of the 'id' field.

filename = testDirectory+"3b4c2228.json"

readTaskFile(filename)
# One way to "cheat" would be to just copy each question, the "noop" solution.



# Function to extract the test questions and return them as our answer

def getNoopAnswer(filename):

    data = readTaskFile(filename)

    #print(data)

    testSection = data["test"]

    ident = data["id"]

    

    numTests = len(testSection)

    answer = {}

    for i in range(numTests):

        answer[ident+"_"+str(i)] = flattener(testSection[i]["input"])

        

    return answer
# Quick test with the first file "19bb5feb.json"

# "3b4c2228.json" has two tests

filename = testDirectory+"3b4c2228.json"

getNoopAnswer(filename)
# A function to loop through the questions in the given directory

# applying a function 'f' to each question.

def getAnswers(directory, f):

    answers = {}

    for _, _, filenames in os.walk(directory):

        for filename in filenames:

            answers.update(f(directory+filename))



    return answers
# Execute our 'noop' function on the test directory

answers = getAnswers(testDirectory, getNoopAnswer)



# Write out the answers into our submission file        

f = open("submission.csv", "w")

f.write("output_id,output\n")

keys = answers.keys()

for key in keys:

    value = answers[key]

    f.write(key + "," + value + "\n")

f.close()
# Let's make sure we did indeed save that file

f2 = open("submission.csv", "r")

print(f2.read())

f2.close()