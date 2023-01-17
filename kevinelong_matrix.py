matrix_board = [
    [0,0,0],
    [0,0,0],
    [0,0,0],
]

matrix_board[0][2] = 3

matrix_board[1][1] = 2

matrix_board[2][0] = 1


#TODO turn into function

for row in matrix_board:
    for column in row:
        print(column, end=" ")
    print(" end row")

print("---")
#ALTERNATIV
for y in range(0,len(matrix_board)):
    for x in range(len(matrix_board[y])):
        print(matrix_board[y][x], end=" ")
    print(" end row")

#TODO 
# how to append to a list https://www.w3schools.com/python/ref_list_append.asp
# 1. BUILD A MATIX 19x19
# 2. Set some values using double square bracket notation
# 3. Print out again
# Extra credit number the rows , Letter the Columns A,B,C ... like Excel


go_board = []
for y in range(0,19):
    row = []
    for x in range(0,19):
        row.append(".")
    go_board.append(row)

go_board[3][5] = "X"
go_board[13][9] = "O"

A = 65

i = 1
for index in range(0,len(go_board)):
    print(chr(index + A), end=" ")
    
print("")
for row in go_board:
    
    for column in row:
        print(column, end=" ")
    print(f" {i}")
    i += 1


for i in range(0,255):
    print(i , chr(i))