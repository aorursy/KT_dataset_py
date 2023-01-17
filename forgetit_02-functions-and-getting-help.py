help(round)
help(round(-3.14))
help(print)
def max_diff(a, b, c):

    diff1 = abs(a - b)

    diff2 = abs(b - c)

    diff3 = abs(a - c)

    return max(diff1, diff2, diff3)
help(max_diff)
def max_diff(a, b, c):

    """返回a、b、c三个数两两之间最大的差值。

    

    >>> max_diff(2, 3, -4)

    7

    """

    diff1 = abs(a - b)

    diff2 = abs(b - c)

    diff3 = abs(a - c)

    return max(diff1, diff2, diff3)
help(max_diff)
def max_diff(a, b, c):

    """返回a、b、c三个数两两之间最大的差值。

    """

    diff1 = abs(a - b)

    diff2 = abs(b - c)

    diff3 = abs(a - c)

    max(diff1, diff2, diff3)

    

print(

    max_diff(1, 1, 1),

    max_diff(1, 10, 100),

    max_diff(3, 4, 5),

)
unknown = print()

print(unknown)
print(1, 2, 3, sep = ' < ')
print(1, 2, 3)
def greet(name = "World"):

    print("Hello, ", name)

    

greet()

greet(name = "Python")

greet("Python")
def my_add(x, y, f):

    return f(x) + f(y)



my_add(5, -8, abs)
def mod_5(x):

    """返回x除以5后的余数。"""

    return x % 5



print(

    max(100, 51, 14), 

    max(100, 51, 14, key=mod_5)

)
def smallest_abs(x, y):

    #请输入你的代码

    pass



smallest_abs(5, -10)
def f(x):

    y = abs(x)

return y



print(f(-5))