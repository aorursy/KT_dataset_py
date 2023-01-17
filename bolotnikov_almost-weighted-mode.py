#don't use it. It has turtle speed...

import pandas as pd

from functools import reduce

import operator

import os

train_path = "../input/train.csv"
def list_split(arr):

    lst = list(map(int, arr[1:].split(" ")))

    return lst



def w_i_N(i, d, sigma):

    return ((d - i + 1.0)/d)**sigma



def calc_w(sigma, d):

    w = [w_i_N(i, d, sigma) for i in range(1, d+1)]

    s = sum(w)

    w = [x / s for x in w]

    return w



def calc_p(arr, j, w, d):

    visits = map(lambda x: 1 if (7 * x + j) in arr else 0, range(d, 0, -1))

    pj = sum(list(map(operator.mul, w, visits)))

    return pj



def calc_arr_p(arr, w, d):

    p = list(map(lambda x: calc_p(arr, x, w, d), range(1, 8)))

    return p

    

def calc_future_p(p, j):

    return p[j] * (1 if j == 0 else reduce(operator.mul, map(lambda x: 1 - x,p[:j])))



def ans(p):

    return p.index(max(p)) + 1

    

def normalize(arr):

    arr_sum = sum(arr)

    return list(map(lambda x: x / arr_sum, arr))



def predict(arr, w, d):  

    p = calc_arr_p(arr, w, d)

    p = normalize(p)

    new_p = [0 for i in range(len(p))]

    for i in range(len(p)):

        new_p[i] = calc_future_p(p, i)

    return ans(new_p)



def add_space(smth):

    return " " + str(smth)



def cross_val(train, sigma):

    d = 157

    w = calc_w(sigma, d)

    true_positive = 0

    all = 0

    answer = {}

    r_bound = min(60000, len(train))

    for i in range(0, r_bound, len(train)//r_bound):

        answer[i] = predict(train[i][:-1], w, d)

        all += 1

        true_positive += 1 if (train[i][-1] - 1) % 7 + 1 == answer[i] else 0

    return true_positive * 100.0 / all
train = pd.read_csv(train_path)
train["visits"] = train["visits"].apply(list_split)
def apply_cross_val():

    best_cv = -100

    best_sigma = -100

    for i in range(21):

        sigma = i / 5

        cv = cross_val(train["visits"], sigma)

        if cv > best_cv:

            best_cv = cv

            best_sigma = sigma

        print(i, sigma, cv)

    print("best_cv =", best_cv, "best_sigma =", best_sigma)

    print(" ")
answers = []

sigma = 2.2

d = 157

w = calc_w(sigma, d)

answers = [predict(train["visits"][i], w, d) for i in range(len(train))]
answerDF = pd.DataFrame(

    {'id': list(range(1, len(answers) + 1)),

     'nextvisit': answers,

    })
answerDF["nextvisit"] = answerDF["nextvisit"].apply(add_space)

answerDF.to_csv('solution.csv', index=False)