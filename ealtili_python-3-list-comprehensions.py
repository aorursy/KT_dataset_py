# Grab every letter in string
lst = [x for x in 'word']
# Check
lst
# Square numbers in range and turn into list
lst = [x**2 for x in range(0,11)]
lst
# Check for even numbers in a range
lst = [x for x in range(11) if x % 2 == 0]
lst
# Convert Celsius to Fahrenheit
celsius = [0,10,20.1,34.5]

fahrenheit = [((9/5)*temp + 32) for temp in celsius ]

fahrenheit
lst = [ x**2 for x in [x**2 for x in range(11)]]
lst