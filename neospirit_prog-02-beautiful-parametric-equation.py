# Prog-02: Beautiful Parametric Equation

# 6330011321 นายกฤติพงศ์ มานะชำนิ

# $x(t)=t-1.6cos(24t)$

# $y(t)=t-1.6sin(25t)$

# โปรแกรมนี้วาดโดยแต่ละเกลียวทะลวงและเบียดเสียดกับเกลียวอื่นรอบข้างพร้อมๆกัน

# ผมเป็นผู้เลือกสมการข้างบนและเขียนนิพจน์คณิตศำสตร์ด้วยตัวเอง

# คุณสามารถสำรวจสมการอื่นๆหรือการปรับเปลี่ยนสีของผมเพิ่มเติมได้ที่นี่ https://www.kaggle.com/neospirit/prog-02-beautiful-parametric-equation



import math

import matplotlib.pyplot as plt



#------------------------------------

def setup_T(min_t, max_t, dt):

    T = []; t = min_t

    while t <= max_t:

        T.append(t)

        t += dt

    if t != max_t: T.append(max_t)

    return T

#------------------------------------

def plot(x, y, min_t, max_t, dt):

    T = setup_T(min_t, max_t, dt)

    X = [x(t) for t in T]

    Y = [y(t) for t in T]

    plt.plot( X, Y, color='blue' )

#====================================

def x(t):

    xt = t -1.6*math.cos(24*t)

    return xt

def y(t):

    yt = t -1.6*math.sin(25*t)

    return yt



plot(x, y, -11, 8, 0.01)

plt.show()
