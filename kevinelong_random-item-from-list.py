# HOW TO USE RANDOM TO GET A RANDOM NUMBER IN A RANGE 1-4
import random

r = random.randint(1,4)
print(r)

#ON ONE LINE
print("A", "B", "C")

# GET RANDOM ITEM FROM A LIST - LIKE A SLOT MACHINE DIAL
data_list = [
    "CHERRY",
    "BELL",
    "BAR"
]

# TODO DISPLAY THREE RANDOM ITEMS IN A ROW; TEN TIMES
# e.g.
# BAR CHERRY BELL
# BELL BAR BAR
# CHERRY BAR BAR
# BELL CHERRY CHERRY
# BELL CHERRY BAR
# CHERRY BAR BELL
# CHERRY BAR CHERRY
# CHERRY CHERRY CHERRY
# CHERRY CHERRY BELL
# BAR BAR CHERRY

for i in range(10): # loop ten times
    
    maximum_index = len(data_list) - 1 # e.g. 3 - 1 = 2 as two would be the max if we have three items as we start with zero.
    
    index1 = random.randint(0, maximum_index) # pick a random index 
    v1 = data_list[index1] # pull text from the data using the index
    
    index2 = random.randint(0,maximum_index)
    v2 = data_list[index2]
    
    index3 = random.randint(0,maximum_index)
    v3 = data_list[index3]
    
    print(v1, v2, v3) #print all three on one line


#  Extra: 
#  1. Use a loop for the columns (might need end="" in your print statement). 
#  2. print("Winner!") if all three columns are the same.
#  3. Use input() to pause between spins. Use while?
#  4. How could you easily make BELL three times as likely to come up as CHERRY? Change data only?

data_list = [
    "CHERRY",
    "BELL",
    "BAR"
]
for i in range(10):
    a = data_list[random.randint(0,2)]
    b = data_list[random.randint(0,2)]
    c = data_list[random.randint(0,2)]
    print(a,b,c)
#  EXTRA 1. Use Loop
data_list = [
    "CHERRY",
    "BELL",
    "BAR"
]
for i in range(10):
    output_list = []
    for n in range(3):
        output_list.append(data_list[random.randint(0,2)])
    print("-".join(output_list))
#  EXTRA 2. Show Win Message if all three are the same.
data_list = [
    "CHERRY",
    "BELL",
    "BAR"
]
for i in range(10):
    a = data_list[random.randint(0,2)]
    b = data_list[random.randint(0,2)]
    c = data_list[random.randint(0,2)]
    
    print(a,b,c)
    if a == b and b == c:
        print("    WINNER!")

# loop waiting for quit

text = ""
while text != "q":
    print("C C C")
    print("ENTER to play again. q to quit.")
    text = input("ENTER to play again. q to quit.")
    
playing = True
while playing:
    print("C C C")
    print("ENTER to play again. q to quit.")
    text = input()
    if text == "q":
        playing = False
#  4. How could you easily make BELL three times as likely to come up as CHERRY? Change data only?
data_list = [
    "CHERRY",
    "BELL",
    "BELL",
    "BELL",
    "BAR"
]
for i in range(10):
    a = data_list[random.randint(0,2)]
    b = data_list[random.randint(0,2)]
    c = data_list[random.randint(0,2)]
    
    print(a,b,c)
    if a == b and b == c:
        print("    WINNER!")
