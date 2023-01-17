from learntools.core import binder; binder.bind(globals())

from learntools.python.ex5 import *

print('Setup complete.')
def has_lucky_number(nums):

    """Return whether the given list of numbers is lucky. A lucky list contains

    at least one number divisible by 7.

    """

   

    for num in nums:

        if num % 7 == 0:

            return True

       

    return False

        

nums = [1,2,3,4,5,6,7]

has_lucky_number(nums)        

        

                

               
def has_lucky_number(nums):

    """Return whether the given list of numbers is lucky. A lucky list contains

    at least one number divisible by 7.

    """

    for num in nums:

        if num % 7 == 0:

            return True

       

    return False

        

nums = [1,2,3,4,5,6,7]

has_lucky_number(nums) 



 #Check your answer

#q1.check()
#q1.hint()

#q1.solution()
l=[1, 2, 3, 4] 

for i in l:

    if i>2:

        print(i)

            

    else:

        pass

         

      

        





        

def elementwise_greater_than(L, thresh):

    """Return a list with the same length as L, where the value at index i is 

    True if L[i] is greater than thresh, and False otherwise.

    

    >>> elementwise_greater_than([1, 2, 3, 4], 2)

    [False, False, True, True]

    """

  

    for i in range(0,len(L)-1):

        for j in range(0,len(thresh)-1):

   

            if [L[i]==thresh[j]]:

               return True

            else:

                return False

            

L=[1,2,3,4,5,6,7,8]

thresh=[0,2,45,4,8,5,7,8]

elementwise_greater_than(L,thresh)



# Check your answer

#q2.check()
#q2.solution()
def menu_is_boring(meals):

    """Given a list of meals served over some period of time, return True if the

    same meal has ever been served two days in a row, and False otherwise.

    """





for l in range(0,len(meals)-1):

    if [meals[l] == meals[l+1]]:

            return True

    else:

            return False

    

meals = [rice,dal,fish,chicken,chicken,sweet,tikka]

menu_is_boring(meals)

    

#pass



# Check your answer

#q3.check()
#q3.hint()

#q3.solution()
play_slot_machine()
def estimate_average_slot_payout(n_runs):

    """Run the slot machine n_runs times and return the average net profit per run.

    Example calls (note that return value is nondeterministic!):

    >>> estimate_average_slot_payout(1)

    -1

    >>> estimate_average_slot_payout(1)

    0.5

    """

   

    result = []

    for x in range(1, n_runs):

        result.append(play_slot_machine())

    return (sum(result) - n_runs)/n_runs



n_runs=(1,3,4,78,56,34)

estimate_average_slot_payout(n_runs)

    #pass
# Check your answer (Run this code cell to receive credit!)

q4.solution()