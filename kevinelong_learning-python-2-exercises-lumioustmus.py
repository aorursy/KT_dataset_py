people = ["Bob", "Carol", "Ted", "Alice"]

print("# Silly Way")
print( people[0] + " " + people[1])
print( people[2] + " " + people[3])

print("# Loopy way")
for i in range(0,2):
    print(people[i], end="")
print("")    
for i in range(-2,0):
    print(people[i], end="")
print("")

print("# Pythonic way")

print(people[:2])
print(people[-2:])

#GIVEN THE FOLLOWING LIST OF LIST
test_data = [
    ["2014-06-01", "APPL", 100.11],
    ["2014-06-02", "APPL", 110.61],
    ["2014-06-03", "APPL", 120.22],
    ["2014-06-04", "APPL", 100.54],
    ["2014-06-01", "MSFT", 20.46],
    ["2014-06-02", "MSFT", 21.25],
    ["2014-06-03", "MSFT", 32.53],
    ["2014-06-04", "MSFT", 40.71],
]

#CREATE TWO NEW LISTS ONE FOR EACH STOCK TICKER SYMBOL e.g. APPL and MSFT

appl = []

msft = []

for item in test_data:
    if item[1] == "APPL":
        appl.append(item)
    elif item[1] == "MSFT":
        msft.append(item)
        
print("appl = ", appl)
print("msft = ", msft)

#EXTRA CREDIT
#ONCE THAT WORKS THEN what would need to change to copy with an unknown number of stock ticker symbols?
# How can we remove redundant key?

output = {}
for item in test_data:
    if item[1] not in output:
        output[item[1]] = []
    output[item[1]].append(item)
#     output[item[1]] + [item]
    
    item.pop(1) 
    # remove ticker symbol from list, 
    # note this works even on the already appended item
    # because it is an object reference not a cimple value type.

output


for x in range(0, 3):
    for y in range(0, 3):
        print( " .", end="")
    print("")

# define the function
def grid(size):
    for x in range(0, size):
        for y in range(0, size):
            print( " .", end="")
        print("")


# call the function
grid(4)


board = [
    [".", ".", "."],
    [".", ".", "."],
    [".", ".", "."],
]

print(board)


def build_board(size):
    all_rows = []
    for x in range(0, size):
        one_row = []
        for y in range(0, size):
            one_row.append(".")
        all_rows.append(one_row)
    return all_rows


my_board = build_board(12)
print(my_board)


def get_board_string(board):
    output = ""
    for r in board:
        for c in r:
            output = output + " " + c
        output = output + "\n"
    return output


print( get_board_string(my_board))

my_board[6][6] = "X"
print( get_board_string(my_board))

my_board[3][5] = "O"
print( get_board_string(my_board))

