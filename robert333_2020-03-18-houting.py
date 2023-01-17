# 第三方库

from scipy.special import comb

from scipy.misc import derivative

from scipy.integrate import quad

import math
# 定义H函数

def H_fun(n, k, ai):

    temp_sum = 0

    for j in range(0, k):

        temp = comb(n, j) * (k - j) * math.pow(ai, n-j) * math.pow(1-ai, j)

        temp_sum += temp

    return temp_sum / k



# 定义h函数（H函数关于ai的导数）

def h_fun(n, k, ai):

    H = lambda x: H_fun(n, k, x) 

    return derivative(H, ai, dx=1e-6)
# test

print('测试H函数')

print(h_fun(10, 2, 0.3))
# 定义R函数中的被积函数

def R_fun_ai(n, k, ai):

    R_1 = math.pow(H_fun(n, k, ai), k-1) * h_fun(n, k, ai)

    R_2 = ai * math.pow(H_fun(n, k, ai), k-1)

    R_3 = quad(lambda x:H_fun(n, k, x), 0, ai)[0]

    return R_1 * math.pow(R_2-R_3, 1/2)



# 定义R函数（关于ai积分）

def R_fun(n, k):

    R_1 = quad(lambda x: R_fun_ai(n, k, x), 0, 1)[0]

    return math.pow(k / 2 * R_1, 2)
# test

print('测试R函数')

print(R_fun_ai(10, 2, 0.3))

print(R_fun(10, 2))
# 由于pi函数过大，所以首先将pi被积函数写出来，然后再积分

# pi被积函数也很大，所以将其分成三个小函数

# 第一个小函数部分

def Prij_sum(n, k, ai):

    temp_sum = 0

    for j in range(1, k+1):

        temp = comb(n-1, j-1) * math.pow(ai, n-j) * math.pow(1-ai, j-1)

        temp_sum += temp

    return temp_sum



# 第二个小函数部分

def H_intergal(n, k, ai):

    return quad(lambda x: math.pow(H_fun(n, k, x), k-1), 0, ai)[0]



# 第三个小函数部分

def supp_fun(n, k, t):

    temp_sum = 0

    for j in range(1, k+1):

        temp = comb(n-1, j-1) * math.pow(t, n-j-1) * math.pow(1-t, j-2) * (n-j-(n-1)*t)

        temp_sum += temp

    return temp_sum



# 定义pi函数

def pi_fun(n, k, ai):

    part1 = Prij_sum(n, k, ai) * H_intergal(n, k, ai)

    part2 = H_intergal(n, k, ai) * quad(lambda t: supp_fun(n, k, t), 0, ai)[0]

    print('n, k, ai=', n, k, ai)

    print('此时，pi的值为')

    return R_fun(n, k) * (part1 - part2) / ai
# test

print(pi_fun(10, 2, 0.2))