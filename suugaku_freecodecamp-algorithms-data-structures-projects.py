#This is the basic solution, which does a straightforward comparison.

import re

def palindrome(string):

    lowercase = string.lower()

    punc_stripped = re.sub("[\W_]", "", lowercase)

    reversed_string = punc_stripped[::-1]

    return punc_stripped == reversed_string
palindromes = ["eye", "_eye", "race car", "not a palindrome", "A man, a plan, a canal. Panama",

              "never odd or even", "nope", "almostomla", "My age is 0, 0 si ega ym.",

              "1 eye for of 1 eye.", "0_0 (: /-\ :) 0-0", "five|\_/|four"]

for phrase in palindromes:

    print(phrase, palindrome(phrase))
# Note that the use of the continue statement tells us to continue

# with the next iteration of the loop after appropriately skipping

# non-alphanumeric characters.

def efficient_palindrome(string):

    # Assign pointers for the front and back of the string

    front = 0

    back = len(string) - 1

    # Since the front and back pointers won't always meet in the middle, 

    # use (back > front) as the stopping condition

    while (back > front):

        # Increment front pointer if current character isn't alphanumeric

        if (not string[front].isalnum()):

            front += 1

            continue

        # Decrement back pointer if current character isn't alphanumeric

        if (not string[back].isalnum()):

            back -= 1

            continue

        # Compare the current pair of characters

        if (string[front].lower() != string[back].lower()):

            return False

        front += 1

        back -= 1

    # If the whole string has been compared without returning false, it's a palindrome!

    return True
for phrase in palindromes:

    print(phrase, efficient_palindrome(phrase))
def convertToRoman(num):

    # Make a dictionary of digits and their Roman numerals

    roman_numerals = {1: "I", 5: "V", 10: "X", 50: "L", 100: "C", 500: "D", 1000: "M"}

    # Make an array of the digits by converting to a string and splitting

    # before converting each digit back into an integer

    digits = [int(char) for char in str(num)]

    converted_num = ""

    #Loop through the digits and convert them

    for i in range(len(digits)):

        converted_digit = ""

        # Store the current place (current exponent for 10)

        # E.g. the ones place corresponds to 10^0, the tens to 10^1, etc

        current_place = len(digits) - (i + 1)

        if (digits[i] == 4):

            # Digit converts to the five numeral for the corresponding place, preceded by the one numeral

            converted_digit = roman_numerals[10**current_place] + roman_numerals[5*10**current_place]

        elif (5 <= digits[i] <= 8):

            # Digit converts to the five numeral for the corresponding place, followed by the one numeral repeated up to three times

            converted_digit = roman_numerals[5*10**current_place] + roman_numerals[10**current_place]*(digits[i] - 5)

        elif (digits[i] == 9):

            # Digit converts to the one numeral for the next place, preceded by the one numeral for the corresponding place

            converted_digit = roman_numerals[10**current_place] + roman_numerals[10**(current_place + 1)]

        else:

            # Digit converts to the one numeral for the corresponding place, repeated up to three times

            converted_digit = roman_numerals[10**current_place]*(digits[i])

        converted_num += converted_digit

    return converted_num
roman_check = [2, 3, 4, 5, 9, 12, 16, 29, 44, 45, 68, 83, 97, 99, 400, 500, 501, 649, 798, 891, 1000, 1004, 1006, 1023, 2014, 3999]

for check in roman_check:

    print(check, "converts to", convertToRoman(check))
def caesar_shift(string, shift = 13):

    alphabet = "abcdefghijklmnopqrstuvwxyz"

    shift_alpha = "".join([alphabet[(i + shift) % 26] for i in range(26)])

    # After creating strings for the alphabet and the shifted alphabet,

    # create a translation dictionary for both upper and lower case letters

    # using the str.maketrans() function

    shift_dict = str.maketrans(alphabet + alphabet.upper(), shift_alpha + shift_alpha.upper())

    # Use the translation dictionary to apply the cipher.

    return string.translate(shift_dict)
caesar_check = ["SERR PBQR PNZC", "SERR CVMMN!", "SERR YBIR?",

               "GUR DHVPX OEBJA SBK WHZCF BIRE GUR YNML QBT."]

for check in caesar_check:

    print("Before translating:", check)

    print("After translating:", caesar_shift(check))

message = "This is a Caesar shifted message!"

encoded = caesar_shift(message, 5)

print("Shifted by 5:", encoded)

print("Decoded message:", caesar_shift(encoded, -5))
import re

def telephoneCheck(string):

    # First check to make sure that the first character is either a digit

    # or an opening parenthesis. This will disallow "-1 (757) 622-7382"

    first_char_check = string[0].isdecimal() or string[0] == "("

    # Extract just the digits from the string

    tele_digits = [int(char) for char in string if char.isdecimal()]

    num_digits = len(tele_digits)

    # Check if the length is 10 or 11

    # If the length is 11, check that the country code is 1

    length_check = num_digits == 10 or (num_digits == 11 and tele_digits[0] == 1)

    # Create a regular expression to check for the pattern "(XXX)"

    # Where X is any character

    parens_regex = r"\(.{3}\)"

    # Check to see if the string contains any parentheses

    # If it does, make sure that it matches the parentheses regex exactly once

    parens_check = True

    if ("(" in string) or (")" in string):

        parens_match = re.findall(parens_regex, string)

        #print(parens_match)

        parens_check = (parens_match is not None) and (len(parens_match) == 1)

    return first_char_check and length_check and parens_check
telephone_numbers = ["555-555-5555", "1 555-555-5555", "1 (555) 555-5555", "5555555555", "555-555-5555",

                    "(555)555-5555", "1(555)555-5555", "555-5555", "5555555", "1 555)555-5555",

                    "1 555 555 5555", "1 456 789 4444", "123**&!!asdf#", "55555555", "(6054756961)",

                    "2 (757) 622-7382", "0 (757) 622-7382", "-1 (757) 622-7382", "2 757 622-7382", "10 (757) 622-7382",

                    "27576227382", "(275)76227382", "2(757)6227382", "2(757)622-7382", "555)-555-5555",

                    "(555-555-5555", "(555)5(55?)-5555"]

for number in telephone_numbers:

    print(number, telephoneCheck(number))
def checkCashRegister(price, cash, cid):

    # Create a dictionary mapping currency denominations to their values

    currency_dict = {"PENNY":0.01, "NICKEL":0.05, "DIME":0.1, "QUARTER":0.25, "ONE":1, "FIVE":5, "TEN":10, "TWENTY":20, "ONE HUNDRED":100}

    # Compute the change due and a secondary variable of the remaining

    # change due for use in the computation of the change

    change_due = cash - price

    remaining_due = cash - price

    change_given = [] # An array to store the change to be given

    cash_available = 0 # For computing the total cash available in the drawer

    # Loop through the currency in the drawer to both count the total

    # and also compute how the change would be given

    # Go from highest value currency to lowest value

    for i in range(len(cid) - 1, -1, -1):

        cash_available += round(cid[i][1], 2)

        if remaining_due >= currency_dict[cid[i][0]]:

            change_to_give = min(cid[i][1], (remaining_due//currency_dict[cid[i][0]])*currency_dict[cid[i][0]])

            change_given.append([cid[i][0], change_to_give])

            remaining_due -= change_to_give

            remaining_due = round(remaining_due, 2) #To account for floating point arithmetic rounding issues

            #print(change_to_give)

            #print(remaining_due)

    if (change_due == cash_available):

        return {"status": "CLOSED", "change": cid}

    elif (change_due > cash_available or remaining_due > 0):

        return {"status": "INSUFFICIENT FUNDS", "change": []}

    else:

        return {"status": "OPEN", "change": change_given}
print(checkCashRegister(19.5, 20, [["PENNY", 1.01], ["NICKEL", 2.05], ["DIME", 3.1], ["QUARTER", 4.25], ["ONE", 90], ["FIVE", 55], ["TEN", 20], ["TWENTY", 60], ["ONE HUNDRED", 100]]))

print(checkCashRegister(3.26, 100, [["PENNY", 1.01], ["NICKEL", 2.05], ["DIME", 3.1], ["QUARTER", 4.25], ["ONE", 90], ["FIVE", 55], ["TEN", 20], ["TWENTY", 60], ["ONE HUNDRED", 100]]))

print(checkCashRegister(19.5, 20, [["PENNY", 0.01], ["NICKEL", 0], ["DIME", 0], ["QUARTER", 0], ["ONE", 0], ["FIVE", 0], ["TEN", 0], ["TWENTY", 0], ["ONE HUNDRED", 0]]))

print(checkCashRegister(19.5, 20, [["PENNY", 0.01], ["NICKEL", 0], ["DIME", 0], ["QUARTER", 0], ["ONE", 1], ["FIVE", 0], ["TEN", 0], ["TWENTY", 0], ["ONE HUNDRED", 0]]))

print(checkCashRegister(19.5, 20, [["PENNY", 0.5], ["NICKEL", 0], ["DIME", 0], ["QUARTER", 0], ["ONE", 0], ["FIVE", 0], ["TEN", 0], ["TWENTY", 0], ["ONE HUNDRED", 0]]))