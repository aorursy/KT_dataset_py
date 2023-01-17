# Create a list of numbers for numerical calculations

nums = [1,3,4,8,9,10,11,2,4,4,9,8,3,1,6,29,82,82,33,85,2,68,93,58,12,3,5,6,9]
## Create a function to calculate averages for any list of numbers

## Calculate the average of the nums list

## Create a function to calculate the 14-index moving average for a given list of numbers



def movingAvg(the_list = None):

    """ Calculates the 14-index moving average for a given list of numbers



        Args:

            the_list (List): List of numbers.



        Returns:

            (List) of movingAvg numbers for each number in the list

    """

    # Add guard statements

    if not the_list:

        print("The list is empty! Please pass in a list of numbers")

        return []

    elif len(the_list) < 14:

        print("The list size is less than 14! Please pass in a bigger list of numbers")

        return []



    ## Start the For loop at the 14th number, which is index 13

    



        # Use list slice to pass in the correct subset of numbers to calculate the moving_average



        ## Print out each list slice to confirm the subsets are correct first

        
## Call the movingAvg function, passing in the nums list

## Verify that the list slice correctly retrieves the previous 14 numbers for each iteration

## Continue to modify the movingAvg function and calculate the averages for each iteration

## Create a new list that will contain all the calculated averages for each iteration

def movingAvg(the_list = None):

    """ Calculates the 14-index moving average for a given list of numbers



        Args:

            the_list (List): List of numbers.



        Returns:

            (List) of movingAvg numbers for each number in the list

    """

    # Add guard statements

    if not the_list:

        print("The list is empty! Please pass in a list of numbers")

        return []

    elif len(the_list) < 14:

        print("The list size is less than 14! Please pass in a bigger list of numbers")

        return []



    # Start the For loop at the 14th number, which is index 13



    ## Create a list to contain all the moving average numbers. The first 13 numbers will be zero





    for index in range(13,len(the_list)):



        # Use list slice to pass in the correct subset of numbers to calculate the moving_averageself.



        ## Print out each list slice to confirm the subsets are correct first

        # print(the_list[index - 13: index + 1])

        

        ## Calculate average and save the average to a running list

        avg = average(the_list[index - 13: index + 1])

        list_ma.append(avg)



    return list_ma
## Call the movingAvg function, passing in the nums list and print out the moving average results
