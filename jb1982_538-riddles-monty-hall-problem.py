# import modules

import numpy as np
#initialise empty lists for the user's door choice outcome and the remaining door outcome 

user_choice_outcomes = []

remaining_door_outcomes = []

monty_choice_outcomes = []



# DEFINE THE METHOD, AND RE-RUN 500000 TIMES



for x in range(500000):



    # initialise variable containing number of goats to be placed (either 0, 1, 2, or 3)

    no_of_goats = np.random.choice([0, 1, 2, 3])

    orig_no_of_goats = no_of_goats.copy()

    

    # intialise the door variables as 0 (indicating no goat placed there), and a list of door_choices

    door_a = 0

    door_b = 0

    door_c = 0

    doors = ['a', 'b', 'c']



    # place the goats randomly behind the doors

    while no_of_goats > 0:

        goat_placed = np.random.choice(doors)

        if goat_placed == 'a':

            door_a = 1

        elif goat_placed == 'b':

            door_b = 1

        else:

            door_c = 1

        doors.remove(goat_placed)

        no_of_goats = no_of_goats - 1



    # randomly select a door for the user, and append the result to user_choice_outcome

    # remove this door from the remaining door choices list

    door_choices = [door_a, door_b, door_c]

    user_choice = np.random.choice(door_choices)

    door_choices.remove(user_choice)

    

    # determine whether there are any other goats placed, and therefore if Monty can choose a door

    # note: we do not need to record outcomes where there are no other goats remaining behing the doors

    if np.sum(door_choices) == 2:

        monty_choice_outcomes.append(1)

        remaining_door_outcomes.append(1)

        user_choice_outcomes.append(user_choice)

    elif np.sum(door_choices) == 1:

        monty_choice_outcomes.append(max(door_choices))

        remaining_door_outcomes.append(min(door_choices))

        user_choice_outcomes.append(user_choice)

        

user_odds = 1 - np.mean(user_choice_outcomes)

remaining_door_odds = 1 - np.mean(remaining_door_outcomes)



print("If Monty is able to open a door containing a goat then...\n")

print("The odds on the user having selected a door with no goat is " + str(user_odds))

print("The odds on the remaining door having no goat is " + str(remaining_door_odds))
