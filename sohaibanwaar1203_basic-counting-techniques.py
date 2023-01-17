k = 5

n = 10

print("Total Numbers between {} and {} is {}".format(k,n,n-k+1))
import time

def magic_math(k,n,divisor):

    '''

    Description

    This funtion will find out how many numbers from range k to n are divisible by every element in a list

    e.g 

        if k=10, n= 19999 and divisor = 3 than This funtion will find out the count of numbers

        in range 10 to 19999 which are divisible by 3

    

    '''

    # loop from (5 -- 10) taking first element which is divisible by 3

    first_divisible_element = [i for i in range(k,(k + divisor) + 1) if i % divisor == 0][0]  

    # loop from (95 -- 100) taking last element which is divisible by 3

    last_divisible_element = [i for i in range(n - divisor, n + 1) if i % divisor == 0][-1]

    return int(((last_divisible_element - first_divisible_element) / divisor) + 1)



def cross_check(k,n,divisor):

    return sum([1 for i in range(k,n+1) if i % divisor ==0])





# Starting Point

k = 3

# End Point

n = 100000000

# Divisor 

divisor = 3





import time

start_time = time.time()

magic_function = magic_math(k,n,divisor)

magic_funtion_time = (time.time() - start_time)



start_time = time.time()

cross_check = cross_check(k,n,divisor)

cross_check_time = (time.time() - start_time)

print("Magic Funtion {} Time Taken: {:.16f}".format(magic_function, magic_funtion_time))

print("Cross Check: {} Time Taken: {:.16f}".format(cross_check, cross_check_time ))





import numpy as np

def cross_check_divisor(k, n, divisor_list):

    '''

    Description

    This funtion will find out how many numbers from range k to n are divisible by every element in a list

    e.g 

        if k=10, n= 19999 and divisor_list = [2,3,4,5] than This funtion will find out the count of numbers

        in range 10 to 19999 which are divisible by 2,3,4,5

    

    '''

    single_divisor_pass_count = 0

    all_divisors_pass_count = 0 

    for i in range(k, n+1):

        single_divisor_pass_count = 0

        for divisor in divisor_list:

            if i % divisor == 0:

                single_divisor_pass_count = single_divisor_pass_count + 1

        if single_divisor_pass_count == len(divisor_list): 

            all_divisors_pass_count = all_divisors_pass_count + 1

    return all_divisors_pass_count





# Start Point

k = 3

# End Point

n = 10000000

# Divisor List

divisor_list = [3,4,5]



start_time = time.time()

magic_function = magic_math(k,n,int(np.lcm.reduce(divisor_list)))

magic_funtion_time = (time.time() - start_time)



start_time = time.time()

cross_check = cross_check_divisor(k, n, divisor_list)

cross_check_time = (time.time() - start_time)



print("Magic Funtion {} Time Taken: {:.16f}".format(magic_function, magic_funtion_time))

print("Cross Check: {} Time Taken: {:.16f}".format(cross_check, cross_check_time ))
# What if i say that tell me number of words that which have possible 5 characters init..

# What will be the answer



# we have 26 alphabets 

# Every character in a word possibly includes a charater from 26 words.



print (26 * 26 * 26 * 26 * 26)  # 11881376 total words



# Now what if I say that no letter will repeat in the word. so on first place we have 26 possible characters

# on second place we have 25 possible charaters on third place we have 24 possible charaters and so on.



print(26 * 25 * 24 * 23 * 22) # 7893600 total words



# Now what if I say find out all combination of number of length 3 and number should not consists 0 init 

# i.e 111, 223, 332 etc Here we have have 9 possible

# digits to place at every position



print(9 * 9 * 9) # 729 number will be formed in length of 3



# Now what if I say find out all the numbers of length 3, number shouldnot consists 0 init and all the numbers

# will be odd (means ending with odd number) i.e 111,223 , 335 etc Try it yourself see what you got.



#Now for this we can take a different approach we will start from right to left. How many odd digits we have?

#(1,3,5,7,9) i.e 5 so __5 we have 5 possible values to fill on the right most side it ensure that it will be odd

#now on the middle place we have 8 numbers left (think it in your mind you will be amazed) and on the left place

#we have 7 digits left.



#so the answer will be 



print (7*8*5) # 280 odd numbers will be the list of 3 digits numbers



# these things are very simple but will help us out in solving really big problems