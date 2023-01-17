sum_of_multiples = 0



for i in range(1, 1000):

    if (i % 3 == 0) or (i % 5 == 0):

        sum_of_multiples += i



print(sum_of_multiples)
def SumMult(limit, multiples):

    sum_of_multiples = 0

    for i in range(limit):

        for mult in multiples:

            if i % mult == 0:

                sum_of_multiples += i 

                break

                

    return(sum_of_multiples)        



# Answer to the problem

print(SumMult(1000,(3,5)))



# Example with other multiples

print(SumMult(1000,(2,7)))



# Example with other limit

print(SumMult(2000,(3,5)))



# Example with more multiple numbers

print(SumMult(2000,(2,5,7)))