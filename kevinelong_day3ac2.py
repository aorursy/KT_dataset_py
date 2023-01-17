data = [
    ["id","first","last", "eligability", "friend"],
    [123,"kevin","long", True, [234,345]],
    [234,"nina","maria", False, [234,123]],    
    [345,"larry","long", True, [234,123]],
    [456,"moe","maria", False, [123,345]],    
]

id_of_first = data[1][0] #second ro is at index 1, first colum is at index 0.
print(id_of_first)

larry_eli = data[3][3]
print(larry_eli)

for item in data:
    for column in item:
        print(column, end=" ")
    print("")
game_board = [
    ['.','.','.'],
    ['.','.','.'],
    ['.','.','.'],
]

print(game_board)
game_board[0][2] = "X"
game_board[1][1] = "X"
game_board[2][0] = "X"

game_board[1] = ['O','O','O'] #replace whole row
print(game_board)

#TODO PUT X three in a row

#L2 extra credit print it pretty
for row in game_board:
    for column in row:
        print(column, end=" ")
    print("")
class Game:
    def __init__(self):
        self.game_board = [
            ['.','.','.'],
            ['.','.','.'],
            ['.','.','.'],
        ]
    def show_board(self):
        for row in self.game_board:
            for column in row:
                print(column, end=" ")
            print("")
        print("")
    def place_token(self, row, col, token = "X"):
        self.game_board[row][col] = token

g = Game()
g.show_board()

for i in range(3):
    g.place_token(i, 2 - i, "O")
    # or
    # game_board[i][i] = "O"
g.show_board()
        
patient_data = {
   "123" : { 
       "first" : "kevin", 
       "last" : "long"
   },
   "234" : { 
       "first" : "nina", 
       "last" : "marie"
   },
}
#How can we get the last name of patient 234?
print(patient_data["234"]["last"])
# patient_data = [
#    { 
#        "first" : "kevin", 
#        "last" : "long"
#    },
#    { 
#        "first" : "nina", 
#        "last" : "marie"
#    },
# ]
patient_data = [{'first': 'kevin', 'last': 'long'}, {'first': 'nina222', 'last': 'marie'}]

#How can we do the same with a list of objects
#get the first name of the last patient?
print(patient_data[-1]["first"])
print(patient_data)
patient_data = {
    "outpatients":[
       { 
           "first" : "kevin", 
           "last" : "long"
       },
       { 
           "first" : "nina", 
           "last" : "marie"
       },
    ],
    "inpatients":[
        { 
           "first" : "Larry", 
           "last" : "long"
       },       
        { 
           "first" : "Moe", 
           "last" : "long"
       },       
    ]
}
#show the first name of the last outpatient
print(patient_data["outpatients"][-1]["first"])

#long and wrong but works
out = patient_data["outpatients"]
last = out[-1]
name = last["first"]
print(name)