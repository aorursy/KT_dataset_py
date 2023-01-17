while True:

    print('Enter 1 to calculate loan & Enter 0 to exit.')

    b = int(input())

    

    if b == 1:

        print("Enter Principal Amount")

        P = int(input())

        print("Enter Annual Interest Rate")

        r = int(input())

        R = (r/12)

        print("Enter total number of years")

        N = int(input())

        A = (P*R*(1+R)**N)/((1+R)**N - 1)

        print("Total Amount payable after", N, "years")

        print("%.2f" % A)

        interest = A-P

        print("Total interest payable after" ,N ,"years")

        print("%.2f" % interest)

    if b == 0:

        break