# Preparations

import random as rnd

import pandas as pd

# function for, uh... well... data generation, duh!

def generate_data(ids, mails, entries):

    return [{'id': str(rnd.randint(1, ids)), 'email': str(rnd.randint(1, mails)) + '@mail.ru'} for i in range(0, entries)]

# data sample from a riddle giver

test_data = [ 

    {"id":1,"email":"abc@ya.ru"},

    {"id":1,"email":"cde@ya.ru"},

    {"id":2,"email":"rtg@ya.ru"},

    {"id":3,"email":"abc@ya.ru"},

    {"id":4,"email":"cde@ya.ru"},

] # where ideal result sample is: [[1, 3, 4],[2]]
def get_rid_of_the_riddle(data):

    users = [] # where result will be stored

    while len(data) > 0: # we will cut initial data sample piece by piece

        uid = [data[0]['id']] # any first id

        umail = [data[0]['email']] # any first email

        loop_must_go_on = True # "Queen" reference

        # find all mails and ids related to user

        while loop_must_go_on == True:

            loop_must_go_on = False # always stop unless any new user entries found

            for i in range(1, len(data)): # rolling all data. Actually, what's left of it.

                if data[i]['id'] in uid: # Did we see this id before?

                    if data[i]['email'] not in umail: # If id is known and mail is not - let's remember the mail

                        umail.append(data[i]['email'])

                        loop_must_go_on = True

                elif data[i]['email'] in umail: # ...or did we see this mail before?

                    if data[i]['id'] not in uid: # If mail is known and id is not - nail this bastard

                        uid.append(data[i]['id'])

                        loop_must_go_on = True

        # exclude those found ids & mails from data

        for i in range(0, len(data))[::-1]: # reverse order, because pop function reduces list size

            if (data[i]['id'] in uid) or (data[i]['email'] in umail):

                data.pop(i)

        users.append(uid) # store found ids a portion related to a single user

    return users
get_rid_of_the_riddle(test_data)
get_rid_of_the_riddle(generate_data(ids=20, mails=20, entries=15))
# input: data, ids // order is important, size must be equal

def merge_sets(data, ids):

    for i in range(len(data)-1, -1, -1):

        for j in range(i-1, -1, -1):

            if not data[i].isdisjoint(data[j]):

                data[j] = data[i].union(data[j])

                ids[j] = ids[j] + ids[i]

                del data[i]

                del ids[i]

                break

    return ids
%%time

# data preparation

use_generator = True

n = 4

if use_generator == True:

    data = generate_data(ids=2*10**n, mails=2*10**n, entries=10**n)

else:

    data = pd.DataFrame(test_data)

data = pd.DataFrame(data)

data = data.drop_duplicates()

data = data.groupby('id')['email'].apply(list)

ids = [[i] for i in list(data.index)]

data = list(data)

data = [set(i) for i in data]

#data, ids, len(data), len(ids)
%%time

# actual calculations

a = merge_sets(data, ids)

if not use_generator: print(a)
# https://stackoverflow.com/questions/3170055/test-if-lists-share-any-items-in-python

# not set(data['email'][0]).isdisjoint(data['email'][4])