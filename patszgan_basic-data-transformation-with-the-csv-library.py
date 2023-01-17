# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



'''

Input data files are available in the read-only "../input/" directory

For example, running this (by clicking run or pressing Shift+Enter) will list all files 

under the input directory if you change /kaggle to /kaggle/inpuit. Else it will list all files in /kaggle

'''



import os

for dirname, _, filenames in os.walk('/kaggle'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# print directories in /kaggle/

for dirname, _, filenames in os.walk('/kaggle'):

    print(dirname)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Good library for working with csv files

import csv



#    Open the csv      'r' for read        write into variable csv_file (not the content of the file, but the opened csv file is called by using this variable)

with open('../input/chess/games.csv', 'r') as csv_file:

    csv_reader = csv.reader(csv_file) # The reader module reads the csv file, it expects a comma separated file, but can also work with other separators if you give it the right argument.

    

    print(csv_reader) # Prints information on the standard variable in it's object form



    row_count = sum(1 for row in csv_reader)

    print(row_count) # Prints the row count + header, e.g. 20.059 = 1 header row + 20.058 records
with open('../input/chess/games.csv', 'r') as csv_file:

    csv_reader = csv.reader(csv_file)

        

    # This prints all the content from the file. Do not do this 

    for line in csv_reader:

        print(line)
with open('../input/chess/games.csv', 'r') as csv_file:

    csv_reader = csv.reader(csv_file)

    

    # This prints out only the first column / field / index 

    for line in csv_reader:

        print(line[0])
with open('../input/chess/games.csv', 'r') as csv_file:

    csv_reader = csv.reader(csv_file)

        

    next(csv_reader) # jumps over the first value (in this case, the header)



    for line in csv_reader:

        print(line[0])
with open('../input/chess/games.csv', 'r') as csv_file:

    csv_reader = csv.reader(csv_file)

    

    # We create or open here a new csv file and tell the new_file variable, that we want to write in here something.

    with open('game_over.csv', 'w') as new_file:

    

        csv_writer = csv.writer(new_file, delimiter='-') # We want to write the content of the csv file into a new file, with another delimiter. In this case, dashes.

        # The csv writer knows what are values and what are delimiters, as it puts dashes around values, that contain the delimiter choosen

        # another delimiter would be \t for tab separation

        for line in csv_reader:

            csv_writer.writerow(line)



data = pd.read_csv('game_over.csv') 

print(data.head())
# I do not know the fieldnames

data = pd.read_csv('../input/chess/games.csv') 

print(data.head())
with open('../input/chess/games.csv', 'r') as csv_file:

    csv_reader = csv.DictReader(csv_file)

    

    # We create or open here a new csv file and tell the new_file variable, that we want to write in here something.

    with open('game_over.csv', 'w') as new_file:

        # If I do not mention all field names, it will only take the choosen field names in this variable. Rated and Winner are missing here

        fields = ['id', 'increment_code', 'opening_ply', 'black_rating', 'victory_status', 'white_id', 'created_at', 

                  'last_move_at', 'opening_name', 'turns', 'black_id', 'opening_eco', 'white_rating', 'moves']

        

        csv_writer = csv.DictWriter(new_file, fieldnames=fields, delimiter=',') # here we add the fieldnames

        

        csv_writer.writeheader() # Writes the first line as headers

        

        for line in csv_reader:

            # An error that comes in handy here: You just choose the desired field names 

            # you want and put them in the variable fields above. After running the code, 

            # you got an error that lists all lines you forogtot to choose.  If you do not want

            # those lines, just copy and paste them in the line below, like I did.

            # I am sure there are better ways, to do this. 

            del line['rated']

            del line['winner']

            csv_writer.writerow(line)



# Print the head of the new file

data = pd.read_csv('game_over.csv') 

print(data.head())
import math 



with open('../input/chess/games.csv', 'r') as csv_file:

    csv_reader = csv.DictReader(csv_file)

    

    # get the split

    row_count = sum(1 for row in csv_reader) # counts all lines

    row_count = row_count - 1 # -1 because I only want the records

    split_level = math.ceil(row_count * 0.8) 

        # how big should the first file be relative to the total record count

        # math.ceil, because we want an integer

        # you might also change this to 0.2 or something else. If the

        # training data is smaller then 0.5, you might change names, as I do

        # use always use more data for training, then for testing



# Somehow the csv file did not work, when I did not open it again

with open('../input/chess/games.csv', 'r') as csv_file:

    csv_reader = csv.DictReader(csv_file)



    with open('game_over_training.csv', 'w') as new_file:

        fields = ['id', 'rated', 'winner', 'increment_code', 'opening_ply', 'black_rating', 'victory_status', 'white_id', 

                    'created_at', 'last_move_at', 'opening_name', 'turns', 'black_id', 'opening_eco', 'white_rating', 'moves']

        

        csv_writer = csv.DictWriter(new_file, fieldnames=fields, delimiter=',')

        csv_writer.writeheader()

        

        count = 1 # start the counter at 1

        for line in csv_reader:

            if count < split_level: # only write in the csv file for items below the split level

                csv_writer.writerow(line)

            count = count + 1



data = pd.read_csv('game_over_training.csv') 

print(data.head())



# Let's write the test data

with open('../input/chess/games.csv', 'r') as csv_file:

    csv_reader = csv.DictReader(csv_file)



    with open('game_over_test.csv', 'w') as new_file:

        fields = ['id', 'rated', 'winner', 'increment_code', 'opening_ply', 'black_rating', 'victory_status', 'white_id', 

                    'created_at', 'last_move_at', 'opening_name', 'turns', 'black_id', 'opening_eco', 'white_rating', 'moves']

        

        csv_writer = csv.DictWriter(new_file, fieldnames=fields, delimiter=',')

        csv_writer.writeheader()

        

        count = 1 # start the counter at 1 again, needs always to be equal, so the first lines are now ignored until we are at the split level

        for line in csv_reader:

            if count >= split_level: # only write in the csv file for items below and equal the split level

                csv_writer.writerow(line)

            count = count + 1



data = pd.read_csv('game_over_test.csv') 

print('\n------------------------------------------------------------')

print(data.head())



# The last one is the demo data

with open('../input/chess/games.csv', 'r') as csv_file:

    csv_reader = csv.DictReader(csv_file)



    with open('game_over_demo.csv', 'w') as new_file:

        fields = ['id', 'rated', 'winner', 'increment_code', 'opening_ply', 'black_rating', 'victory_status', 'white_id', 

                    'created_at', 'last_move_at', 'opening_name', 'turns', 'black_id', 'opening_eco', 'white_rating', 'moves']

        

        csv_writer = csv.DictWriter(new_file, fieldnames=fields, delimiter=',')

        csv_writer.writeheader()

        

        count = 1 # start the counter at 1 again

        for line in csv_reader:

            if count < 1000: # for demo data to play around with, 1.000 lines should be fine

                csv_writer.writerow(line)

            count = count + 1



data = pd.read_csv('game_over_demo.csv')

print('\n------------------------------------------------------------')

print(data.head())