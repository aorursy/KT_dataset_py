import random

import time



print("""**********************************



Welcome to Guessing Number Game...



Guess number between 1 and 100



**********************************

""")



number=random.randint(1,100)

guess_right = 7

while True:

    guess_number = int(input('Your guessing number:'))

    if guess_number < number:

        print('Loading...')

        time.sleep(1)



        print('Please guess higher number')

        guess_right-=1

    elif guess_number > number:

        print('Loading...')

        time.sleep(1)



        print('Please guess lower number')

        guess_right-=1

    else:

        print('Loading...')

        time.sleep(1)



        print('Congratulation,you guessed it correct')

        break

    if guess_right ==0:

        print("your guess right is over")

        print("Correct number is:",number)

        break