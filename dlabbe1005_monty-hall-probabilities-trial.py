import numpy as np
num_trials = 1000
def newGame():

    door = np.random.randint(1,4)

    if door == 1:

        return {'door1': 'car', 'door2': 'goat', 'door3': 'goat'}

    if door == 2:

        return {'door1': 'goat', 'door2': 'car', 'door3': 'goat'}

    if door == 3:

        return {'door1': 'goat', 'door2': 'goat', 'door3': 'car'}



def guestChoice():

    return 'door' + str(np.random.randint(1,4))



def openOneDoor(game, chosen_door):

    options = []

    for door in game:

        if game[door] != 'car' and chosen_door != door:

            options.append(door)

    game[np.random.choice(options)] = 'open'

    return game



def guestChange(game, chosen_door):

    options = []

    for door in game:

        if game[door] != 'open' and chosen_door != door:

            options.append(door)

    return options[0]

    

def checkResult(game, chosen_door):

    if game[chosen_door] == 'car':

        return 'win'

    else:

        return 'lose'
change_wins = 0 

dont_change_wins = 0



for n in range(num_trials):

    game = newGame()

    chosen_door = guestChoice()

    game = openOneDoor(game, chosen_door)

    new_chosen_door = guestChange(game, chosen_door)



    if checkResult(game, chosen_door) == 'win':

        dont_change_wins = dont_change_wins +1

    if checkResult(game, new_chosen_door) == 'win':

        change_wins = change_wins +1



print ('Dont Change: {} out of {} trials ({:0.2f}%)'.format(dont_change_wins, num_trials, dont_change_wins/num_trials*100))

print ('Change: {} out of {} trials ({:0.2f}%)'.format(change_wins, num_trials, change_wins/num_trials*100))