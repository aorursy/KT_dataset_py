#while condition:

#     action    
loopCounter=0

while loopCounter<5:

    print("loopCounter has a value of : "+str(loopCounter))

    loopCounter=loopCounter+1
favNumber=input("Type your favorite number or press q to quit: ")

while favNumber!='q':

    print("Your favorite number is "+favNumber)

    favNumber=input("Type your favorite number or press q to quit: ")

print("You quit!")