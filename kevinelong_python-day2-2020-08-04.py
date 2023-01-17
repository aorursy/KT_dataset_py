# Create functions that:

# First function take two parameters, text_symbol, number_of_times
# Print that symbol the number of times specified.

# Second will the same but return a string of symbols that can then be printed.

# The third  will all in one line. using print's "end" property.

# EXTRA CREDIT print a square grid that is the same width and height using the "number_of_times" parameter for both.



def print_symbol(symbol="*", number_of_times=3):
    for _ in range(number_of_times):
        print(symbol)
print_symbol()
    
def make_symbol(symbol="*", number_of_times=3):
    output = ""
    for _ in range(number_of_times):
#         output = output + symbol #CONCATENATION CONCATENATE -= 
        output += symbol
    return output

print(make_symbol("#",2))
def print_symbol_one_line(symbol="*", number_of_times=3):
    for _ in range(number_of_times):
        print(symbol, end="")
        
print_symbol_one_line()
def print_symbol_grid(symbol="*", number_of_times=3):
    for row in range(number_of_times):
        for column in range(number_of_times):
            print(symbol, end="  ")
        print() #new line
print_symbol_grid(".", 5)
def make_symbol_grid(symbol="*", number_of_times=3):
    output = ""
    for row in range(number_of_times):
        for column in range(number_of_times):
            output = output + symbol + " "
        output += "\n"
    return output

result = make_symbol_grid()
print(result)