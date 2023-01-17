one_digits = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
two_digits = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]  # "fourty"
teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
to_nineteen = one_digits+teens
count = 0

# 1 to 19
count = sum([len(element) for element in to_nineteen[1:]])
[print(el) for el in to_nineteen]


# count = 0
# 20 to 99
for tens in two_digits:
    for ones in one_digits:
        count += len(tens+ones)
        print(tens+ones)

# count = 0
# from 100 to 999
for digit in one_digits[1:]:
    count += len(digit + "hundred")  # *00
    print(digit + "hundred")
    for n in to_nineteen[1:]:  # *01-*19
        count += len(digit + "hundred" + "and" + n)
        print(digit + "hundred" + "and" + n)
    for t in two_digits:  # *20-*99
        for o in one_digits:
            count += len(digit + "hundred" + "and" + t + o)
            print(digit + "hundred" + "and" + t + o)
            
count += len("onethousand")
print("onethousand")
count