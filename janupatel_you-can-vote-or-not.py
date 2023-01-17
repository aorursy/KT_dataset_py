while True:



    print('Enter 1 to add person & Enter 0 to exit.')

    b = int(input())

    if b == 1:

        print('What is your name?')

        a = input()

        print('How old are you?')

        age = int(input())

        if age >= 18:

            print('You are old enough to vote!')

        else:

            print("You can't vote yet.")

    elif b == 0:

        break

    else:

        print('Please Enter valid option')