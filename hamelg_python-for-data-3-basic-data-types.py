type(12)
# Check if 12 is an instance of type "int"



isinstance(12, int)
1/3  # A third is not a whole number*
type(1/3)  # So the type of the result is not an int
type(1.0)
isinstance(0.33333, float)
5 + 1.0
int(6.0)
float(6)
type ( float ("Inf") )
type ( float ("NaN") )
type(True)
isinstance(False, bool)  # Check if False is of type bool
# Use >  and  < for greater than and less than:

    

20 > 10 
20 < 5
# Use >= and  <= for greater than or equal and less than or equal:



20 >= 20
# Use == (two equal signs in a row) to check equality:



10 == 10
40 == 40.0  # Equivalent ints and floats are considered equal
# Use != to check inequality. (think of != as "not equal to")



1 != 2
# Use the keyword "not" for negation:



not False
# Use the keyword "and" for logical and:



(2 > 1) and (10 > 11)
# Use the keyword "or" for logical or:



(2 > 1) or (10 > 11)
2 > 1 or 10 < 8 and not True
((2 > 1) or (10 < 8)) and (not True)
bool(1)
bool(0)
type("cat")
type('1')
type(None)  
# Define a function that prints the input but returns nothing*



def my_function(x):

    print(x)

    

my_function("hello") == None  # The output of my_function equals None