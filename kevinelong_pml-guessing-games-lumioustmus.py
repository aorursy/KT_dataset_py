# Game Loop
playing = True
score = 0
while playing:
        print("Action: e.g. 'go north' or 'quit': ")
        action = input()
        if action == "quit":
            playing=False
print("GAME OVER!")

import random 
print("Number guessing game") 
# randint function to generate the  
# random number b/w 1 to 9 
number = random.randint(1, 9) 
# number of chances to be given  
# to the user to guess the number  
# or it is the inputs given by user 
# into input box here number of   
# chances are 5 
chances = 0
print("Guess a number (between 1 and 9):")  
# While loop to count the number 
# of chances 
while chances < 5: 
    # Enter a number between 1 to 9  
    guess = int(input()) 
    # Compare the user entered number   
    # with the number to be guessed  
    if guess == number: 
        # if number entered by user   
        # is same as the generated   
        # number by randint function then   
        # break from loop using loop  
        # control statement "break" 
        print("Congratulation YOU WON!!!") 
        break
    # Check if the user entered   
    # number is smaller than   
    # the generated number  
    elif guess < number: 
        print("Your guess was too low: Guess a number higher than", guess) 
    # The user entered number is   
    # greater than the generated  
    # number              
    else: 
        print("Your guess was too high: Guess a number lower than", guess) 
    # Increase the value of chance by 1 
    chances += 1
# Check whether the user   
# guessed the correct number  
if not chances < 5: 
    print("YOU LOSE!!! The number is", number) 
# no need to hold the human number in a variable
import random
limit = 10
def guess():
    #TODO use feedback responses to hone in on actual hiddne human
    return random.randint(1,limit)

response = ""
while response is not "c":
    g = guess()
    print(g)
    print("respond with h for higher, l for lower, or c if correct")
    response = input()
print("CORRECT!")

# Extra play again.

import random

class Picker():
    def __init__(self, limit=100):
        self.limit = limit
    def pick(self):
        self._pick = random.randint(1, self.limit)
        return self._pick
    def respond(self, guess):
        if self._pick > guess:
            return "HIGHER"
        elif self._pick < guess:
            return "LOWER"
        else:
            return "CORRECT"
        
class Guesser():
    def __init__(self, limit):
        self.limit = limit
        self._guess = int(limit / 2)
        self._highest = limit
        self._lowest = 1
    def learn(self, response):
        if response == "HIGHER":
            self._lowest = self._guess
            self._guess = int((self._highest + self._guess) / 2 + 0.5) 
        if response == "LOWER":
            self._highest = self._guess
            self._guess = int(self._guess / 2 + 0.5)

    def guess(self):
        return self._guess

class Game():
    def __init__(self, limit=100):
        self.limit = limit
        self.picker = Picker(limit)
        self.guesser = Guesser(limit)
        self.guesses = []
    def play(self):
        response = ""
        pick = self.picker.pick()
        print("Picked:", pick)
        while response != "CORRECT":
            guess = self.guesser.guess()
            print("Guess:", guess)
            response = self.picker.respond(guess)
            print("Response:", response)
            self.guesses.append(guess)
            self.guesser.learn(response)
import time
t0= time.clock()

results = []
epochs = 10
limit = 10
for e in range(0,epochs):
    print("===GAME=== ", e)
    g = Game(limit)
    g.play()
    results.append(g.guesses)
# for r in results:
#     print(len(r))
    
print("FINISHED: ", epochs)
counts = []
minimums = []
maximums = []
for r in results:
    counts.append(len(r))
    minimums.append(min(r))
    maximums.append(max(r))
print("AVERAGE: ", sum(counts) / epochs )
print("MIN ALL: ", min(counts) )
print("MAX ALL: ", max(counts) )

t1 = time.clock() 
print("Time elapsed: ", t1 - t0) # CPU seconds elapsed (floating point)

'''
         Ai Guessing Game

Enter a number, not exceeding the
SIZE variable. Computer will guess
your number.

To try other sizes, change the
size variable. 

Input a whole number

'''

import random

size=1000

num=int(input())
counter=1
cpunum=random.randint(0,size)
run=True
bigger=size
lower=0


while run==True:
    if num > size:
        num=int(input("Number has to be between 0 and {}".format(size)))

    if cpunum==num:
        print("AI guessed your number({}) in {} tries!".format(num,counter))
        break
        
    if cpunum > num:
        if bigger >= cpunum:
            bigger=cpunum
        cpunum=random.randint(lower+1,cpunum-1)
        print(cpunum)
        counter+=1
        continue
        
    if cpunum < num:
        if lower <= cpunum:
            lower=cpunum
        cpunum=random.randint(cpunum+1,bigger-1) 
        print(cpunum)
        counter+=1
        continue