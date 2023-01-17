# . . .
# . . .
# . . .
# CREATE GRID
grid = [
    [ ".", ".", ".", ],
    [ ".", ".", ".", ],
    [ ".", ".", ".", ],    
]

# INIT TOKENS
row_index = 1
column_index = 1

grid[row_index][column_index] = "X"

# DISPLAY GRID
for row in grid:
    for column in row:
        print(column, end=" ")
    print("")
def get_grid():
    return [
        [ ".", ".", ".", ],
        [ ".", ".", ".", ],
        [ ".", ".", ".", ],    
    ]

def place_token_on_grid(grid, row_index, column_index, token):
    grid[row_index][column_index] = token

def display_grid(grid):
    # DISPLAY GRID
    for row in grid:
        for column in row:
            print(column, end=" ")
        print("")
            
def initialize_game():
    # INIT TOKENS
    grid = get_grid()
    place_token(grid, 1,1, "X")
    display_grid(grid)

initialize()
class Grid:
    
    def __init__(self):
        self.make_grid()
        
    def make_grid(self):
        self.board = [
            [ ".", ".", ".", ],
            [ ".", ".", ".", ],
            [ ".", ".", ".", ],    
        ]

    def place(self, row_index, column_index, token):
        self.board[row_index][column_index] = token

    def display(self):
        for row in self.board:
            for column in row:
                print(column, end=" ")
            print("")
            
# INIT TOKENS
grid = Grid()
grid.place(1,1, "X")
grid.display()

class Player:
    def __init__(self, symbol, name = ""):
        self.token = symbol
        self.name = name
class TicTacToe:
    
    def __init__(self):
        self.winner = None
        self.grid = Grid()
        self.players = [
            Player("X"),
            Player("O")
        ]
        self.turn_index = 0
    
    def play(self):
        while(
#                 self.grid.hasSpaces() 
#                 and 
                self.winner == None
            ):
            self.grid.display()
            symbol = self.players[self.turn_index].token
            print(f"PLayer {symbol} its your turn.")
            print("Row? (0, 1, or 2)")
            row = int(input())
            print("Column? (0, 1, or 2)")
            column = int(input())
            self.grid.place(row, column,symbol)
            
            # check to see if player has won
            
            #move to next players turn
            next_turn_index = self.turn_index + 1
            if next_turn_index >= len(self.players):
                next_turn_index = 0
            self.turn_index = next_turn_index
            

game = TicTacToe()
game.play()
# #  Class/Objects,           Attibutes/Property, Functions/Methods
# #  NOUNS(Person,Place,Thing)   ADJECTIVES          VERB/ACTION

# Pharmacy
#     Perscriptions
#     Fill
    
# Deli
#     Make(Sandwich)
    

# Kitchen
#     Refrigerator
#         COlor = Beige
#         MakeIce
#         setTemp(t)

# Bike
#     Color=Red
#     ChangeGear(n)
#     Peddle()
    
# Dog
#     kind="Poodle"
#     Walk()
#     Nap()

# Road
#     signs
#     lanes=2
#     Accident()
#     CountCar()

# FootBall
#     color=blue
#     Toss()

# AirPlane
#     Heading
#     Altitude
#     AirSpeed
#     lift
#     weight
#     drag
    
#     Fly()
#     Accel()
#     Decel()
#     Warn()
    
# Fan
#     speed
#     change_speed(n)
    
# Farm
#     products
#     fields
#     livestock
#     water()
#     sell()

# Laptop
#         weight = 3lbs
#         compute()
#         display()
    
