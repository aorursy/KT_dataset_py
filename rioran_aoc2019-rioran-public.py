# libraries I need
import pandas as pd
from collections import Counter
data = pd.read_csv('/kaggle/input/AoC_2019_puzzle_01-1.txt', header=None)
data.columns = ['mass']
data['fuel'] = data['mass'].apply(lambda x: x//3-2)
data['fuel'].sum()
def fuel_recursion(mass):
    fuel_sum = max(mass//3-2, 0)
    if fuel_sum > 8:
        fuel_sum += fuel_recursion(fuel_sum)
    return fuel_sum
data['fuel_recursion'] = data['mass'].apply(lambda x: fuel_recursion(x))
data['fuel_recursion'].sum()
file = open('/kaggle/input/AoC_2019_puzzle_02-1.txt','r')
str_master_data = file.readline().strip()
master_data = str_master_data.split(',')
master_data = list(map(int, master_data))

def run_intcode(noun, verb):
    global master_data
    data = [x for x in master_data]
    data[1], data[2] = noun, verb
    position = 0 # to track what instruction is being processed
    step = 4 # to move cursor
    try:
        while position < len(data):
            if data[position] == 1:
                data[data[position+3]] = data[data[position+1]]+data[data[position+2]]
            elif data[position] == 2:
                data[data[position+3]] = data[data[position+1]]*data[data[position+2]]
            elif data[position] == 99:
                break
            else:
                print(f'Unknown instruction at position -{position}-')
            position += step
    except:
        print(f'Error: verb = {verb}, noun = {noun}')
    return data[0]

run_intcode(12, 2)
desired_output = 19690720
def brutforce_output():
    for i_noun in range(0, 100):
        for i_verb in range(0, 100):
            if run_intcode(i_noun, i_verb) == desired_output:
                return f'{i_noun}{i_verb}'
brutforce_output()
# My approach will be to collect coordinates sequences and to find where sets of coords overlap
file = open('/kaggle/input/AoC_2019_puzzle_03-1.txt','r')
data = [x.strip() for x in file.readlines()] # read 2 lines of input
wire_a = [(x[0:1], int(x[1:])) for x in data[0].split(',')] # convert each input line into data lists
wire_b = [(x[0:1], int(x[1:])) for x in data[1].split(',')]

def parse_wire_coordinates(wire):
    wire_coords = []
    pos_x, pos_y = 0, 0 # our cursor to track where we are
    for command in wire:
        x_coords = [pos_x+(1+x)*(command[0] in 'RL')*(-1)**(command[0]=='L') for x in range(command[1])]
        y_coords = [pos_y+(1+y)*(command[0] in 'UD')*(-1)**(command[0]=='D') for y in range(command[1])]
        temp = list(zip(x_coords, y_coords))
        wire_coords += temp
        pos_x += (command[1])*(command[0] in 'RL')*(-1)**(command[0]=='L')
        pos_y += (command[1])*(command[0] in 'UD')*(-1)**(command[0]=='D')
    return wire_coords

a_coords, b_coords = parse_wire_coordinates(wire_a), parse_wire_coordinates(wire_b)
wire_crossings = list(set(a_coords).intersection(set(b_coords)))
min([abs(cross[0])+abs(cross[1]) for cross in wire_crossings])
# since my coordinates are ordered - I can get the answer right away
min([a_coords.index(cross)+b_coords.index(cross)+2 for cross in wire_crossings])
# well, 372037 can be rewritten as 377777.
# It's the lowest applicable number in my range.
# The highest one is: 899999
password_count = 1
number = [3,7,7,7,7,7]
upper_limit = 899999

# will always keep digits in number variable in ascending order
def iterate_password():
    global number
    for i in range(5,-1,-1):
        if number[i] < 9:
            number[i] += 1
            for j in range(i+1,6,1):
                number[j] = number[i]
            return

def duplicates_found():
    global number
    return len(set(number)) < 6

def list_to_number():
    global number
    return sum([number[x]*10**(5-x) for x in range(6)])

while list_to_number() < upper_limit:
    iterate_password()
    if duplicates_found():
        password_count += 1

password_count
password_count = 0 # because basic number no longer satisfies the 2 digit rule
number = [3,7,7,7,7,7]

def exactly_two_digits_are_near():
    global number
    return 2 in Counter(number).values()

while list_to_number() < upper_limit:
    iterate_password()
    if exactly_two_digits_are_near():
        password_count += 1

password_count