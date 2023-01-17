
board = ". . ."
board_list = board.split(" ")
board_list[1] = "X"
print(board_list)

board2 = "X X X"
board_list2 = board2.split(" ")
print(board_list2)

# print(board_list[0] != ".")
# print(board_list2[0] != ".")

def isWinner(board_list):
    has_won = False
    # return true if all values are the same and the values are not "." ie=not blank
    #TODO YOUR CODE HERE
    
    non_blank_quantity = 0
    
    for item in board_list:
        if item != ".":
            non_blank_quantity += 1
            
    has_won = (non_blank_quantity == 3)
    return has_won

print(isWinner(board_list))
print(isWinner(board_list2))

#L2 what if we need 3 in a row but the board is 9 positions wide? ". X X . X X X . ."
coins = {
    1: 4,
    25: 3
}
# print(25 in coins)
# print(5 in coins)

print(10 in coins) #false
coins[10] = 2 # add two dimes
coins[10] += 1 # increase by one 
print(10 in coins) #true

for c in coins:
    print(c, coins[c] )

# WEB REQUEST ERROR LOG list of lists where inner list has a path and an error code. 500 is an error code.
data = [
    ["/", 200],
    ["/", 500],
    ["/", 200],
    ["/", 300],
    ["/foo/", 500],
    ["/", 200],
    ["/", 300],
    ["/bar/", 200],    
    ["/foo/", 500],
    ["/", 200],
    ["/", 300],
    ["/bar/", 500],
]

row = data[0]
value = row[1]
print(value)

print(data[0][1])

print(data[-1][0]) # ???

# How many 500 errors total?
error_total = 0
for item in data:
    if 500 == item[1]:
        error_total += 1
print(error_total)

# How many failed requests of each type?
output_dict = {}
for item in data:
    path = item[0]
    code = item[1]
    if code == 500: #FILTER
        if path not in output_dict:
            output_dict[path] = 1
        else:
            output_dict[path] += 1
print(output_dict)
# EXAMPLE
# L1
# request_quantity = {
#     "/" : 0,
#     "/foo/": 0,
#     "/bar": 0
# }

# L2
# summary_output ={
#     "/" : {
#         "TOTAL" : 8,
#         200: 4,
#         300: 3,
#         500: 1
#     },
#     "/foo/": {
#         "TOTAL" : 8,
#         200: 4,
#         300: 3,
#         500: 1
#     },
#     "/bar/": {
#         "TOTAL" : 8,
#         200: 4,
#         300: 3,
#         500: 1
#     }
# }
# EXAMPLE QUESTION: Any path have only errors?


# how can we print every value of every column in every row of a list of lists?
data = [
    [ 'O', 'X', 'O'],
    [ 'X', 'X', 'O'],
    [ 'X', 'O', '.'],
]

# L1 How can we place a letter at the empty bottom right position?

for row in data:
    for column in row:
        print(column, end=" ")
    print("")
    
# L2 How can we make the above into a function we can re-use?

# L3 How can we combine the data and the function into a Board class?

# L4 How can we say if X, O, or no-one has won? Can we put this in a Game class that uses the Board class?

# how can we print every value of every column in every row of a list of lists?
data = [
    [ 'O', 'X', 'O'],
    [ 'X', 'X', 'O'],
    [ 'X', 'O', '.'],
]
for row in data:
    for column in row:
        print(column, end=" ")
    print("")
# L1 How can we place a letter at the empty bottom right position?
data[2][2] = 'X'

# L2 How can we make the above into a function we can re-use?
def show(data):
    for row in data:
        for column in row:
            print(column, end=" ")
        print("")
show(data)    

# L3 How can we combine the data and the function into a Board class?
class Board:
    def __init__(self):
        self.data = [
            [ 'O', 'X', 'O'],
            [ 'X', 'X', 'O'],
            [ 'X', 'O', '.'],
        ]
    def show(self):
        for row in self.data:
            for column in row:
                print(column, end=" ")
            print("")
            
# L4 How can we say if X, O, or no-one has won? Can we put this in a Game class that uses the Board class?
class Game:
    def __init__(self):
        self.board = Board()
    def who_has_won(self):
        # loop through symbols ["X","O"]
            #won = #check horizontal
            #or
            #check vertical
            #or
            #check diagonal1
            #or
            #check diagonal2
            #             if won:
            #                 return symbol
        return None