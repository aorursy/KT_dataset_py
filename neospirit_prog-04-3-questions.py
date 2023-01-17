# homework week 4

# game of N-questions

# example game "what is the number in your mind?"

# with 4 questions



print("Think of a number between 1..16")

ans = input("Is it greater than 8? ").strip()



if( ans == "yes" ):

    ans = input("Is it greater than 12? ").strip()

    

    if( ans == "yes" ):

        ans = input("Is it greater than 14? ").strip()

        

        if( ans == "yes" ):

            ans = input("Is it greater than 15? ").strip()

            if( ans == "yes" ):

                print("the number is 16")

            elif( ans == "no" ):

                print("the number is 15")

            else:

                print("invalid answer")

                

        elif( ans == "no"):

            ans = input("Is it greater than 13? ").strip()

            if( ans == "yes" ):

                print("the number is 14")

            elif( ans == "no"):

                print("the number is 13")

            else:

                print("invalid answer")

                

        else:

            print("invalid answer")

            

    elif( ans == "no"):

        ans = input("Is it greater than 10? ").strip()

        

        if( ans == "yes" ):

            ans = input("Is it greater than 11? ").strip()

            if( ans == "yes" ):

                print("the number is 12")

            elif( ans == "no" ):

                print("the number is 11")

            else:

                print("invalid answer")

                

        elif( ans == "no" ):

            ans = input("Is it greater than 9? ").strip()

            if( ans == "yes" ):

                print("the number is 10")

            elif( ans == "no"):

                print("the number is 9")

            else:

                print("invalid answer")

                

        else:

            print("invalid answer")

            

    else:

        print("invalid answer")

                    

elif (ans == "no"):

    ans = input("Is it greater than 4? ").strip()

    

    if( ans == "yes" ):

        ans = input("Is it greater than 6? ").strip()

        if( ans == "yes" ):

            ans = input("Is it greater than 7? ").strip()

            if( ans == "yes" ):

                print("the number is 8")

            elif( ans == "no" ):

                print("the number is 7")

            else:

                print("invalid answer")

                

        elif( ans == "no"):

            ans = input("Is it greater than 5? ").strip()

            if( ans == "yes" ):

                print("the number is 6")

            elif( ans == "no"):

                print("the number is 5")

            else:

                print("invalid answer")

                

        else:

            print("invalid answer")

            

    elif( ans == "no"):

        ans = input("Is it greater than 2? ").strip()

        if( ans == "yes" ):

            ans = input("Is it greater than 3? ").strip()

            if( ans == "yes" ):

                print("the number is 4")

            elif( ans == "no" ):

                print("the number is 3")

            else:

                print("invalid answer")

        elif( ans == "no" ):

            ans = input("Is it greater than 1? ").strip()

            if( ans == "yes" ):

                print("the number is 2")

            elif( ans == "no"):

                print("the number is 1")

            else:

                print("invalid answer")

        else:

            print("invalid answer")

    else:

        print("invalid answer")

                    

else:

    print("invalid answer")