x = float(input(" Please enter total Marks: "))

total = x

percentage = (total / 500) * 100

print("Marks Percentage = %.2f"  %percentage)

if(percentage >= 90):

    print("A Grade")

elif(percentage >= 80):

    print("B Grade")

elif(percentage >= 70):

    print("C Grade")

elif(percentage >= 60):

    print("D Grade")

elif(percentage >= 40):

    print("E Grade")
