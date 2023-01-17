#Your code goes here

def least_difference(a, b, c):

    diff1 = abs(a - b)

    diff2 = abs(b - c)

    diff3 = abs(a - c)

    return min(diff1, diff2, diff3)
#Your code here test the function you created 



print(least_difference(1, 10, 100))

print(least_difference(1, 10, 10))

print(least_difference(5, 6, 7))

#Your code here - storing the return value into a variable



answer = least_difference(5,25,100)



print(answer)
#Your code goes here copy paste the above function you created an take out return



def least_difference_no_return(a, b, c):

    diff1 = abs(a - b)

    diff2 = abs(b - c)

    diff3 = abs(a - c)

    min(diff1, diff2, diff3)
#Your code goes here

print(least_difference_no_return(5,25,100))
#Your code goes here

def greet(who="World"):

    print("Hello,", who)
#Your code goes here

greet()

greet('Mars')
#Set Up Code



list_of_numbers = [70,4,102,88]



#Your code goes here start by defining a function that takes 1 arguement



### Without Bonus

def largest_element(a_list):



    largest = 0

    

    for i in a_list:

        if i > largest:

            largest = i

            

    return largest





### With Bonus 

def largest_element_bonus(a_list):

    if type(a_list)== list: 

    

        largest = 0

        for i in a_list:

            if type(i) == int: 

                if i > largest:

                    largest = i

            else:

                print("please make sure each element in the list is of type int")

                return

        return largest

                    

    else: 

        print("please input a list")

        return



print(largest_element(list_of_numbers))