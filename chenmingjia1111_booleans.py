from learntools.core import binder; binder.bind(globals())
from learntools.python.ex3 import *
print('Setup complete.')
x = True
print(x)
print(type(x))
def can_run_for_president(age):
    """这些年龄的人能在美国竞选总统吗?"""
    # 美国宪法规定你必须 "达到35岁"
    return age >= 35

print("Can a 19-year-old run for president?", can_run_for_president(19))
print("Can a 45-year-old run for president?", can_run_for_president(45))
3.0 == 3
'3' == 3
def is_odd(n):
    return (n % 2) == 1

print("Is 100 odd?", is_odd(100))
print("Is -1 odd?", is_odd(-1))
# 在这里写代码，定义 'sign' 函数

q1.check()
#q1.solution()
def can_run_for_president(age, is_natural_born_citizen):
    """Can someone of the given age and citizenship status run for president in the US?"""
    # The US Constitution says you must be a natural born citizen *and* at least 35 years old
    return is_natural_born_citizen and (age >= 35)

print(can_run_for_president(19, True))
print(can_run_for_president(55, False))
print(can_run_for_president(55, True))
True or True and False
def prepared_for_weather(have_umbrella, rain_level, have_hood, is_workday):
    # 不要改代码，我们的目标是找出问题，不是修复问题!
    return have_umbrella or rain_level < 5 and have_hood or not rain_level > 0 and is_workday

# 改变输入值来让 prepared_for_weather
# 返回错误的结果.
have_umbrella = True
rain_level = 0.0
have_hood = True
is_workday = True

# 检查函数根据上述输入返回的结果
actual = prepared_for_weather(have_umbrella, rain_level, have_hood, is_workday)
print(actual)

q3.check()
#q3.hint()
#q3.solution()
def inspect(x):
    if x == 0:
        print(x, "is zero")
    elif x > 0:
        print(x, "is positive")
    elif x < 0:
        print(x, "is negative")
    else:
        print(x, "is unlike anything I've ever seen...")

inspect(0)
inspect(-15)
def f(x):
    if x > 0:
        print("Only printed when x is positive; x =", x)
        print("Also only printed when x is positive; x =", x)
    print("Always printed, regardless of x's value; x =", x)

f(1)
f(0)
print(bool(1)) # all numbers are treated as true, except 0
print(bool(0))
print(bool("asf")) # all strings are treated as true, except the empty string ""
print(bool(""))
# Generally empty sequences (strings, lists, and other types we've yet to see like lists and tuples)
# are "falsey" and the rest are "truthy"
if 0:
    print(0)
elif "spam":
    print("spam")
def quiz_message(grade):
    if grade < 50:
        outcome = 'failed'
    else:
        outcome = 'passed'
    print('You', outcome, 'the quiz with a grade of', grade)
    
quiz_message(80)
def quiz_message(grade):
    outcome = 'failed' if grade < 50 else 'passed'
    print('You', outcome, 'the quiz with a grade of', grade)
    
quiz_message(45)