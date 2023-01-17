# computer will Pick a number between 1 and 10

import random
guess = random.randint(1,11)

game_over = False

while game_over == False:
        print("Guess:")
        user_input = input()
        value = int(user_input)
        # if higher or lower then tell them
        # else if its correct then set game over to True

# you pick a number and tell  computer its high or low
# how smart can we make this?
