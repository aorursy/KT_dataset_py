x = True

print(x)

print(type(x))
def can_apply_license(age):

    """判断该年龄是否能申请驾驶证。"""

    # 年龄必须 "达到18岁"

    return age >= 18



print(

    can_apply_license(16),

    can_apply_license(50)

)
print(3.0 == 3)

print('3' == 3)
def is_even(n):

    return (n % 2) == 0



print(

    is_even(10),

    is_even(-1)

)
def can_apply_license(age, is_taking_drugs):

    """根据年龄和是否吸毒判断是否可以申请驾照。"""

    # 年龄达到18岁，未在吸食毒品

    return (age >= 35) and (not is_taking_drugs)



print(can_apply_license(16, True))

print(can_apply_license(50, False))

print(can_apply_license(50, True))
True or True and False
def inspect(x):

    if x == 0:

        print(x, "is zero")

    elif x > 0:

        print(x, "is positive")

    elif x < 0:

        print(x, "is negative")

    else:

        print(x, "is unknown")



inspect(0)

inspect(-15)
def f(x):

    if x > 0:

        print("x =", x)

    print("x =", x)



f(1)

f(0)
# 非零数值、非空字符串、非空list等，就判断为True，否则为False。

print(bool(1))

print(bool(0))

print(bool("asf"))

print(bool(""))
if 0:

    print(0)

elif 3:

    print("meals")
def quiz_result(score):

    if score < 60:

        result = 'failed'

    else:

        result = 'passed'

    print('You', result, 'the quiz with a score of', score)

    

quiz_result(80)
def quiz_result(score):

    result = 'failed' if score < 60 else 'passed'

    print('You', result, 'the quiz with a score of', score)

    

quiz_result(59)
a = 5 > 3 and 3 > 1

b = 2 > 4 or 4 > 2

c = not 1 > 2

d = bool(0)

e = bool("")

f = 1 + 2 if () else 4