print("Program to print half pyramid: ");



rows = input("Enter number of rows ")

rows = int (rows)



for i in range (0, rows):

    for j in range(0, i + 1):

        print("*", end=' ')



    print("\r")