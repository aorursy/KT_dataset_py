def IsHarshad(num):

    sum_digits = 0

    test_num = num

    while test_num > 0:

        remainder = test_num % 10

        sum_digits += remainder

        test_num = test_num // 10

    if num % sum_digits == 0:

        return True

    return False



IsHarshad(201)
def IsHarshadRight(num):

    while num > 0:

        num = num // 10

        if IsHarshad(num) == True:

            return True

            continue

        else:

            return False



IsHarshadRight(2011)
def IsPrime(num):

    if num > 1:

        for x in range(2, int(num)):

            if (num % x) == 0:

                return False

                break

        else:

            return True

    return False



IsPrime(2011)
def IsHarshadStrong(num):

    sum_digits = 0

    test_num = num

    while test_num > 0:

        remainder = test_num % 10

        sum_digits += remainder

        test_num = test_num // 10

    if num % sum_digits == 0:

        if IsPrime(num / sum_digits) == True:

            return True

        else:

            return False

    else:

        return False



IsHarshadStrong(201)
def IsHarshadRightStrong(num):

    if IsPrime(num) == True:

        trun_num = num // 10

        if IsHarshadRight(trun_num) == True:

            if IsHarshadStrong(trun_num) == True:

                return True

    return False

        

IsHarshadRightStrong(2011)
lower = 100 #100 is the lowest number possible

upper = 10000



sum_of_rightstrong = 0



for i in range(lower, upper):

    if IsHarshadRightStrong(i) == True:

        sum_of_rightstrong += i



print(sum_of_rightstrong)