majors = ['AI', 'PM', 'WEB', 'JAVA']

for major in majors:

    print(major, end=' ') # 打印在一行
numbers = (2, 2, 3, 5, 6)

product = 1

for number in numbers:

    product = product * number

product
s = 'A Boy Can Do Everything For Girls.'

# 打印字符串s中所有的大写字母

for char in s:

    if char.isupper():

        print(char, end='')        
for i in range(5):

    print("do something. i =", i)
r = range(5)

r
help(range)
list(range(5))
nums = [1, 2, 4, 8, 16]

for i in range(len(nums)):

    nums[i] = nums[i] * 2

nums
def double_odd(nums):

    for i, num in enumerate(nums):

        if num % 2 == 1:

            nums[i] = num * 2



x = list(range(10))

double_odd(x)

x
list(enumerate(['a', 'b']))
nums = [

    ('one', 1, 'I'),

    ('two', 2, 'II'),

    ('three', 3, 'III'),

    ('four', 4, 'IV'),

]



for word, integer, roman_numeral in nums:

    print(integer, word, roman_numeral, sep=' = ', end='; ')
for tup in nums:

    word = tup[0]

    integer = tup[1]

    roman_numeral = tup[2]

    print(integer, word, roman_numeral, sep=' = ', end='; ')
i = 0

while i < 10:

    print(i, end=' ')

    i += 1
squares = [n**2 for n in range(10)]

squares
squares = []

for n in range(10):

    squares.append(n**2)

squares
majors = [major for major in majors if len(major) < 3]

majors
lower_majors = [major.lower() + '!' for major in majors if len(major) < 3]

lower_majors
[

    major.lower() + '!' 

    for major in majors 

    if len(major) < 3

]
[10 for major in majors]
def count_negatives(nums):

    """统计并返回给定列表中负数的个数。

    

    >>> count_negatives([3, -4, 0, 1, -3])

    2

    """

    count = 0

    for num in nums:

        if num < 0:

            count = count + 1

    return count
def count_negatives(nums):

    return len([num for num in nums if num < 0])
def count_negatives(nums):

    # 回想在“布尔表达式”章节中，有关 True 和 False 的计算。

    return sum([num < 0 for num in nums])
def has_lucky_number(nums):

    """返回数值列表中是否含有幸运数字。

    列表中至少有一个数可以被7整除，则返回 `True`，否则返回 `False`。

    """

    for num in nums:

        if num % 7 == 0:

            return True

        else:

            return False
def element_greater_than(L, target):

    """将列表 `L` 的每个元素与 `target` 进行比较，并返回一个布尔值的列表。

    

    >>> element_greater_than([1, 2, 3, 4], 2)

    [False, False, True, True]

    """

    pass