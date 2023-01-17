# library we'll need

import json



# read in data

with open('../input/diplomacy_data.json') as json_data:

    data = json.load(json_data)
### let's check and see what interactions there are

for season in range(0, len(data[0]["seasons"])):

    print(data[0]["seasons"][season]['interaction'])
# let's use pandas for this

import pandas



# make an empty dataframe to put all our interactions in

allInteractions = pandas.DataFrame(columns=['game', 'season', 'victim', 'betrayer'])



# for every game in the dataframe....

for game in range(0, len(data)):

    # for every season in that game...

    for season in range(0, len(data[game]["seasons"])):

        # get the action of the betrayer & victim

        victim = data[game]["seasons"][season]['interaction']['victim']        

        betrayer = data[game]["seasons"][season]['interaction']['betrayer']

        

        # put everything in a new dataframe

        newLine = pandas.DataFrame([[game, season, victim, betrayer]],

                           columns=['game', 'season', 'victim', 'betrayer'])



        # and put them in our dataframe

        allInteractions = allInteractions.append(newLine, 

                                       ignore_index=True)

        

# check out the first couple rows

allInteractions[:10]
# what actions did the victim take?

allInteractions['victim'].value_counts()
# what actions did the betrayer take?

allInteractions['betrayer'].value_counts()
# write out our data

with open("allInteractions.csv","w") as file:

    allInteractions.to_csv(file)