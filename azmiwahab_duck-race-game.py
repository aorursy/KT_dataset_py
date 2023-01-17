import random #import random library
# time of race is 10 sec. Imagine every second is consider 1 loop. We are going to make this random game in 10 loops
player = int(input('How many ducks in this race: '))

winning_number = random.randint(1,(player + 1))

           
if player <= 1:
    print("No Race!")
else:
    print('There are', player, 'ducks in this race!')


# start race by pressing ENTER

print(input('Press ENTER to start race'))

for timer in range(1, 11):
    winner = random.randint(1,(player + 1))
    print(timer, 'sec lead: ', winner)
import random

time = int(input('How long this race: '))
print(time, 'seconds')
player = int(input('How many ducks in this race: '))

# build condition for race must have at least 2 ducks
if player <= 1:
    print("No Race!")
else:
    print(player, 'ducks will start race to the finish line. Are you ready?')

#   start race by pressing ENTER

    print(input('Press ENTER to start race'))

    for timer in range(1, (time + 1)):
        winner = random.randint(1,(player + 1))
        print(timer, 'sec lead: ', winner)

    print('Congratulation! The winner is duck', str(winner) +'!')

# What if I want the last second will be announced immediately
# What if I accidentaly press enter without put any value. 

import random

# time = int(input('How long this race: '))
time = int(input("How long this race: ") or "0") # If enter with no value is consider 0
print(time, 'seconds')
if time != 0:
    player = int(input('How many ducks in this race: ') or '0')
    if player != 0:

        if player <= 1:
            print("No Race!")
        else:
            print(player, 'ducks will start race to the finish line. Are you ready?')

        #     start = ''
            print(input('Press ENTER to start race'))

            for timer in range(1, (time + 1)):
                winner = random.randint(1,(player + 1))
                while timer < time:                      #while loop to break print for final loop
                    print(timer, 'sec lead: ', winner)
                    break

            print('\nFinishing Line!')
            print('Congratulation! The winner is duck', str(winner) +'!')
            
    else:
        print("You need to have duck to race or there will be no race")
else:
    print("You need to set the timer or there will be no race")


# Add: What if want to start again the game after finish game
# Add: If the time and number of duck were not given, can I start the game over again?

import random

# Put the code in function so this code can be used again

def loop(): # <--- function name
    time = int(input("How long this race: ") or "0") # If enter with no value is consider 0
    print(time, 'seconds')
    if time != 0:
        player = int(input('How many ducks in this race: ') or '0')
        if player != 0:

            if player <= 1:
                print("No Race!")
            else:
                print(player, 'ducks will start race to the finish line. Are you ready?')

            #     start = ''
                print(input('Press ENTER to start race'))

                for timer in range(1, (time + 1)):
                    winner = random.randint(1,(player + 1))
                    while timer < time:                      #while loop to break print for final loop
                        print(timer, 'sec lead: ', winner)
                        break

                print('\nFinishing Line!')
                print('Congratulation! The winner is duck', str(winner) +'!')
                restart = input('Start again? [y/n]: ')      # Add option question to start again
                if restart == 'y':
                    loop() # <---- send back to start function
                else:
                    print('Goodbye!')

        else:
            print("You need to have duck to race or there will be no race")
            restart = input('Start again? [y/n]: ') or ''     # Add option question to start again
            if restart == 'y':
                loop()
            elif restart == 'n':
                print("Goodbye!")
            else:
                print('Selection error!')
                loop()   # <---- send back to start function
    else:
        print("You need to set the timer or there will be no race")
        restart = input('Start again? [y/n]: ') or ''
        if restart == 'y':
            loop()
        elif restart == 'n':
            print('Goodbye!')
        else:
            print('Selection error!')
            loop()    # <---- send back to start function

    
    
loop()




