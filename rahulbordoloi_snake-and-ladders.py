import random
class StopExecution(Exception):
    def _render_traceback_(self):
        pass
class Snake:

    # Default Constructor
    def __init__(self):

        print("###### Welcome to Snakes & Ladders Game ######")
        self.name = input("Enter the Name of Player 1 : ")
        self.comp = input("Enter the Name of Player 2 : ")
        print("###### Let us Start ######")
        # self.name = name
        # self.comp = comp
        # self.p1 = 0
        # self.p2 = 0
        self.game = [0, 0]    # Steps of the Players

        # Goto Dictionaries
        self.pos_of_snakes = {17 : 7, 54 : 34, 62 : 19, 98 : 79}
        self.pos_of_ladders = {3 : 38, 24 : 33,  42 : 93, 72 : 84}
    
    # Display Player's Name
    def displayName(self):

        print("Player 1's Name : {}, Player 2's Name : {}".format(self.name, self.comp))

    # Display Winner's Name
    def displayWinner(self, number):

        if number == "1":   winner = self.name
        else:   winner = self.comp
        print("Player {} won the Game!".format(number))
        print("Congratulations {} !!".format(winner))
        print("###### Game Successfully Finished ######")
        raise StopExecution                       # ipynb equivalent of exit(0)

    # Check for Any Snakes or Ladders in the current position
    def checkSnakeLadder(self, pos):

        if pos in self.pos_of_snakes.keys():    pos = self.pos_of_snakes.get(pos)
        elif pos in self.pos_of_ladders.keys():     pos = self.pos_of_ladders.get(pos)
        return pos

    # Check for 'quit'
    def quitGame(self, number):

        if number == "1":   self.displayWinner("2")
        else:   self.displayWinner("1")
        raise StopExecution                       # ipynb equivalent of exit(0)
    
    # Check for Overflow Position
    def checkMoreThanHundred(self, pos, x):

        if (pos + x) > 100:   pass                  # Check for >100
        else:   pos += x
        return pos
    
    # Check for the Range of Input in Manual Mode
    def checkManualMode(self, inp):
        x = int(inp)
        if x not in range(2,20):
            print("Invalid Input! The Number you Entered isn't within the range of between 1 and 20")
            x = int(input("Please Enter a Number within the given Range : "))
        return x

    # Check for the Move Input if it's Valid or Illegal
    def checkMoveInput(self, number):

        inp = input("Player {} : ".format(number))
        if inp == "roll":   x = random.randint(1, 6)
        elif inp == "quit":     self.quitGame(number)
        elif inp.isnumeric():   x = self.checkManualMode(inp)
        else:   
            print("Illegal Input, Please Input a Valid Input!")
            x = self.checkMoveInput(number)
        print("You got a ", x)
        return x
    
    # Player Position Function
    def playerPosition(self, number):
        
        pos = 0
        pos = self.game[int(number)-1]
        x = self.checkMoveInput(number)            # Check for Valid Input
        pos = self.checkMoreThanHundred(pos, x)    # Check for the Validity of the Position
        pos = self.checkSnakeLadder(pos)           # Check for Snakes and Ladder
        print(" Your Final Position is ", pos)
        if pos == 100:                             # Check for Winner
            self.displayWinner(number)
        self.game[int(number)-1] = pos             # Lastly assigning the step into the list

    # Snake Game Function
    def snakeGame(self):

        while True:
            self.playerPosition('1')
            self.playerPosition('2')
            # break
# Main Method
if __name__ == "__main__":
    snake = Snake()
    # snake.displayName()
    snake.snakeGame()
    del snake                                     # Delete the Object