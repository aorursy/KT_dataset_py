x = 10                # Assign some variables

y = 5



if x > y:             # If statement

    print("x is greater than y")
y = 25

x = 10



if x > y:

    print("x is greater than y")

else:

    print("y is greater than x")
y = 10



if x > y:

    print("x is greater than y")

elif x == y:

    print("x and y are equal!")

else:

    print("y is greater than x")
my_sequence = list(range(0,101,10))    # Make a new list



for number in my_sequence:  # Create a new for loop over the specified items

    print(number)           # Code to execute
for number in my_sequence:

    if number < 50:

        continue              # Skip numbers less than 50

    print(number)             
for number in my_sequence:

    if number > 50:

        break              # Break out of the loop if number > 50

    print(number)     
x = 5

iters = 0



while iters < x:      # Execute the contents as long as iters < x

    print("Study")

    iters = iters+1   # Increment iters by 1 each time the loop executes
while True:            # True is always true!

    print("Study")

    break              # But we break out of the loop here
import numpy as np



# Draw 25 random numbers from -1 to 1

my_data = np.random.uniform(-1,1,25)  



for index, number in enumerate(my_data):  

    if number < 0:               

        my_data[index] = 0            # Set numbers less than 0 to 0



print(my_data)
my_data = np.random.uniform(-1,1,25)  # Generate new random numbers



my_data = np.where(my_data < 0,       # A logical test

                   0,                 # Value to set if the test is true

                   my_data)           # Value to set if the test is false



print(my_data)