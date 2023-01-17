target_number = 10
print(target_number)
Guessed_number = int(input("Enter the number you guessed: "))

while(target_number != Guessed_number):
    print("!!! Sorry. The Number you entered isnot correct !!!",Guessed_number)
    Guessed_number = int(input("Enter the number again \n Hint:(0-10)"))
    
print("Congrats, The number you entered", Guessed_number , "is correct.")
