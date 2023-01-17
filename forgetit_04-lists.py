L = [1, 2, 3, 4]
majors = ['AI', 'PM', 'WEB', 'JAVA']
cards = [

    ['J', 'Q', 'K', 'A'],

    ['2', '2', '2', '2'],

    ['6', '6', '6', 'K'], # 最后一个元素之后的逗号是可选的，留着也不会报错

]

# 也可以写成一行，但读起来就不容易了

cards = [['J', 'Q', 'K', 'A'], ['2', '2', '2', '2'], ['6', '6', '6', 'K']]
favorites = [10, 'show me the code', help]
majors[0]
print(majors[-1])

print(majors[-2])
majors[0:3]
majors[:3]
majors[3:]
# 除了第一个和最后一个的所有元素

majors[1:-1]
# 最后三个元素

majors[-3:]
majors[3] = 'Python'

majors
majors[:3] = ['Artificial', 'Product', 'World Wide Web']

print(majors)

# 恢复成原状

majors[:4] = ['AI', 'PM', 'WEB', 'JAVA',]
len(majors)
sorted(majors)
L = [1, 2, 3, 4]

sum(L)
max(L)
x = 10

# x是一个实数，它的虚部是0

print(x.imag)

# c是一个复数

c = 10 + 3j

print(c.imag)
x.bit_length()
help(x.bit_length)
majors.append('PYTHON')
help(majors.append)
majors
majors.pop()
majors
majors.index('AI')
"DENO" in majors
"WEB" in majors
help(majors)
t = (1, 2, 3)
t = 1, 2, 3 # 等价于上方表达式，省略圆括号

t
t[0] = 100
x = 0.25

x.as_integer_ratio()
numerator, denominator = x.as_integer_ratio()

print(numerator / denominator)
a = 1

b = 0

a, b = b, a

print(a, b)
def select_second(L):

    """返回给定列表的第二个元素。如果没有，返回 None。

    """

    pass
a = [1, 2, 3]

b = [1, [2, 3]]

c = []

d = [1, 2, 3][1:]



# 将预测结果填写在下面的列表中。应包含4个数字，第一个是列表a的长度，第二个是b的长度，以此类推。

lengths = []
