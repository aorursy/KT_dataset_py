# Task 01 - To Do: write a function to check if a given number is a palindrome



def is_palindrome(user_input=None):

    """

    Check if a positive integer is equal to itself

    if reversed backwards.

    

    The return value is a Boolean for string or

    positive integer data types, None for other

    types and errors.

    """

    try:

        if not type(user_input) in [str, int]:

            raise ValueError

            return None

        # List of ints has efficiency advantage against strings

        arr_digits = [int(x) for x in str(user_input)]

        # We need to iterate only half of the number

        # excluding the middle digit if number len is odd

        len_to_check = len(arr_digits)//2

        for i in range(len_to_check):

            if arr_digits[i] != arr_digits[-1-i]:

                return False

        return True

    except ValueError:

        print(f'\nERROR: Incorrect value starting with: "{str(user_input)[:20]}"'

             , 'Value must be either positive integer or it\'s string representation.'

             , sep='\n', end='\n\n')

        return None

    except:

        print(f'\nERROR: Unknown error for value starting with: "{str(user_input)[:20]}"'

             , sep='\n', end='\n\n')

        return None



samples_to_check = [10001, "12345", 123321, "01", -7, None, 3.14, 7874787, '-670.3']



for value in samples_to_check:

    print(f'{value} => is palindrome? => {is_palindrome(value)}')
help(is_palindrome)