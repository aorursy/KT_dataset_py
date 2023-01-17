def digit_powers(number, power):
    sum_dp = sum([int(digit)**power for digit in str(number)])
    return sum_dp
for num in (1634, 8208, 9474):
    print(digit_powers(num, 4))
digit_powers(10**6, 4)
for i in range(10): print(i**5)
special_nums = []
limit = 6 * 9**5

for num in range (2, limit):
    if digit_powers(num, 5) == num:
        special_nums.append(num) 

solution = sum(special_nums)
solution
special_nums