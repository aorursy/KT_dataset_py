tictac = ['*', '*', '*', '*', '*', '*', '*', '*', '*']
#Display by slicing
print(str(tictac[0:3]) + '\n' + str(tictac[3:6]) + '\n' + str(tictac[6:9]))
x = 7
y = 3

print(x % y)
def draw_board():
    v = '|    |    |    |'
    h = ' ____ ____ ____ '
    for i in range(0,10):
        if i % 3 == 0: #modulus
            print(h)
        else:
            print(v)
draw_board()
1 % 2
def draw_ref_board():
    row = 1
    v1 = '|  1  |  2  |  3  |'
    v2 = '|  4  |  5  |  6  |'
    v3 = '|  7  |  8  |  9  |'
    h = ' _____ _____ _____ '
    for i in range(0,7):
        if i % 2 == 0: #modulus
            print(h)
        else:
            #print (row)
            if row == 2:
                print(v1)
            elif row == 4:
                print(v2)
            elif row == 6:
                print (v3)
        row = row +1
draw_ref_board()
theBoard = {'7': ' ' , '8': ' ' , '9': ' ' ,
            '4': ' ' , '5': ' ' , '6': ' ' ,
            '1': ' ' , '2': ' ' , '3': ' ' }

board_keys = []

def printBoard(board):
    print(board['7'] + '|' + board['8'] + '|' + board['9'])
    print('-+-+-')
    print(board['4'] + '|' + board['5'] + '|' + board['6'])
    print('-+-+-')
    print(board['1'] + '|' + board['2'] + '|' + board['3'])
    
printBoard(theBoard)
game = [['#', '#', '#'],
        ['#', '#', '#'],
        ['#', '#', '#']]
game[:]
for row in game:
    print(row)
    
fruit_basket = ['raspberry', 'grapes', 'banana', 'orange', 'apple']
for counter, value in enumerate(fruit_basket):
    print (counter, value)
print("   0     1    2")
for count, row in enumerate(game):
    print(count, row)
x_coord = input("Enter x coordinates: ")
y_coord = input("Enter y coordinates: ")
game[int(x_coord)][int(y_coord)] = 'X'
for row in game:
    print(row)

x_coord = input("Player 2, Enter x coordinates: ")
y_coord = input("Player 2, Enter y coordinates: ")
game[int(x_coord)][int(y_coord)] = 'O'
for row in game:
    print(row)
draw_ref_board()
tictac = [0, 0, 0, 0, 0, 0, 0, 0, 0]
for count, row in enumerate(tictac):
    print(count +1, row)
draw_ref_board()
player1 = input("Player 1, Enter board # position: ")
tictac[int(player1)-1] = 'X'
tictac[:]
player2 = input("Player 2, Enter board # position: ")
tictac[int(player2)-1] = 'O'
tictac[:]
tictac = ['*', '*', '*', '*', '*', '*', '*', '*', '*']
tictac[:]
#Display by slicing
print(str(tictac[0:3]) + '\n' + str(tictac[3:6]) + '\n' + str(tictac[6:9]))
#Display by iterating items in the list
item = 0
while item < len(tictac) / 3:
    if item ==0:
        print (str(tictac[0]) + ' | ' + str(tictac[1]) + ' | ' + str(tictac[2]))
    elif item ==1:
        print (str(tictac[3]) + ' | ' + str(tictac[4]) + ' | ' + str(tictac[5]))
    else:
        print (str(tictac[6]) + ' | ' + str(tictac[7]) + ' | ' + str(tictac[8]))
    item +=1
draw_ref_board()
def draw_tictac():
    row = 1
    v1 = '|  1  |  2  |  3  |'
    v2 = '|  4  |  5  |  6  |'
    v3 = '|  7  |  8  |  9  |'
    h = ' _____ _____ _____ '
    
    v1 = '|  ' + str(tictac[0]) + '  |  ' + str(tictac[1])  + '  |  ' + str(tictac[2]) + '  |'
    v2 = '|  ' + str(tictac[3]) + '  |  ' + str(tictac[4])  + '  |  ' + str(tictac[5]) + '  |'
    v3 = '|  ' + str(tictac[6]) + '  |  ' + str(tictac[7])  + '  |  ' + str(tictac[8]) + '  |'
    h = ' _____ _____ _____ '
    
    for i in range(0,7):
        if i % 2 == 0: #modulus
            print(h)
        else:
            #print (row)
            if row == 2:
                print(v1)
            elif row == 4:
                print(v2)
            elif row == 6:
                print (v3)
        row = row +1
draw_tictac()
print ("### GET Ready to Play TIC-TAC-TOE!!! ### \nA Mr. Oseguera's Python Class production\n\n")
draw_ref_board()
tictac = ['*', '*', '*', '*', '*', '*', '*', '*', '*']
moves = 0
while moves < 9:
    player1 = input("Player 1, Enter board # position: ")
    tictac[int(player1)-1] = 'X'
    moves +=1
    draw_tictac()
    player2 = input("Player 2, Enter board # position: ")
    tictac[int(player2)-1] = 'O'
    draw_tictac()
    moves +=1

