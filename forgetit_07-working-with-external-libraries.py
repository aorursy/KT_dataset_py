import math



type(math)
print(dir(math))
print("pi to 2 significant digits = {:.2}".format(math.pi))
math.log(8, 2)
help(math.log)
help(math)
import math as mt

mt.pi
import math

mt = math
from math import *

print(pi, log(8, 2))
from math import *

from numpy import *

print(pi, log(32, 2))
from math import log, pi

from numpy import asarray
import numpy

print("numpy.random is a", type(numpy.random))

print("it contains names such as...",

      dir(numpy.random)[-10:]

     )
rolls = numpy.random.randint(low=1, high=6, size=10)

rolls
type(rolls)
print(dir(rolls))
# 求均值

rolls.mean()
# 转换为列表

rolls.tolist()
help(rolls.ravel)
help(rolls)
[3, 4, 1, 2, 2, 1] + 10
rolls + 10
# 判断数组各个位置的元素是否不大于3

rolls <= 3
xlist = [[1,2,3],[2,4,6],]

# 二维数组

x = numpy.asarray(xlist)

print("xlist = {}\nx =\n{}".format(xlist, x))
# 获取数组第二行最后一个元素

x[1,-1]
# Get the last element of the second sublist of our nested list?

xlist[1,-1]
import tensorflow as tf



a = tf.constant(1)

b = tf.constant(1)

a + b
print(dir(list))
import matplotlib.pyplot as plt

plt.plot([1,2,3,4],[30,40,45,50])

# 完成题目要求



plt.show()
def blackjack_hand_greater_than(hand_1, hand_2):

    """

    Examples:

    >>> blackjack_hand_greater_than(['K'], ['3', '4'])

    True

    >>> blackjack_hand_greater_than(['K'], ['10'])

    False

    >>> blackjack_hand_greater_than(['K', 'K', '2'], ['3'])

    False

    """

    pass