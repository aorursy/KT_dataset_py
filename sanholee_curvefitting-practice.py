%matplotlib inline



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
def objective(x):

    f = np.poly1d([2, -10,1])

    return f(x)
x = np.arange(10)

y = objective(x)
plt.plot(x,y)
rough_ = {

    "x_":[0,3,4,5,7,9,11],

    "y_":[-1,0,0,4,10,43,50]

}
df = pd.DataFrame(data=rough_)

df
x=df.x_

y=df.y_



plt.scatter(x,y)
model_1d = np.polyfit(x,y,1)

model_1d
predict_1d = np.poly1d(model_1d)

predict_1d(1)
from sklearn.metrics import r2_score

r2_score(y, predict_1d(x))
x_lin = range(0,12)

y_lin = predict_1d(x_lin)

plt.scatter(x,y)

plt.plot(x_lin, y_lin, c ='r')
model_2d = np.polyfit(x,y,2)

model_2d
predict_2d = np.poly1d(model_2d)

r2_score(y, predict_2d(x))
y_lin = predict_2d(x_lin)

plt.scatter(x,y)

plt.plot(x_lin, y_lin, c = 'r')
def make_polyEquation(data, n=1):

    '''

    1변수 다항식을 근사한다.

    

    input(object type) :

        data = {

        "x":[...],

        "y":[...]

        }

        

        n = 차수, 

    

    output : r2_score, 근사식 모델

    

    '''

    df = pd.DataFrame(data)

    x = df[df.columns[0]]

    y = df[df.columns[1]]

    

    # 1. calculating coefficients for polynomial equation 

    model = np.polyfit(x,y,n)

    

    # 2. mapping coefficients for polynominal equation

    predict_model = np.poly1d(model)

    print(predict_model)

    

    # calculating R2 Score

    r2_result = r2_score(y, predict_model(x))

    print("r2 :", r2_result)

    

    # Visualizing equation

    plt.scatter(x,y)

    plt.plot(x,predict_model(x))

    

    return {"r2":r2_result, "predict": predict_model}

poly_1d = make_polyEquation(data = rough_)
poly_2d = make_polyEquation(data = rough_, n=2)
poly_3d = make_polyEquation(data = rough_, n=3)
equation_3 = poly_3d["predict"]
def exp_func(x,coef):

    return coef["a"]*np.exp(coef["b"]*x+coef["c"])+coef["d"]

    
data_numbers = 50

coef = {

    "a":5,

    "b":-1,

    "c":1,

    "d":1

}
x_exp = np.linspace(1,6,data_numbers)

y_exp = exp_func(x=x_exp, coef=coef)

plt.plot(x_exp,y_exp)
# np.random.uniform(range_start,range_end, [array form; single number of n,m type])

# and making y_data for fitting

y_noise = 1*np.random.uniform(-1,1,[data_numbers])

y_data = y_exp + y_noise
plt.scatter(x_exp, y_data, c='r')

plt.plot(x_exp,y_exp)
model_ln = np.polyfit(x_exp, np.log(y_data), 1)

predict_ln = np.poly1d(model_ln)

print(predict_ln)
plt.scatter(x_exp, y_data, c='r')

plt.plot(x_exp, np.exp(predict_ln(x_exp)))
# calculating r2_score

r2_result_ln = r2_score(y_data, np.exp(predict_ln(x_exp)))

r2_result_ln
# 함수 계수를 2개

def exp_func2(x,a,b):

    return a*np.exp(b*x)
# 함수 계수 3개

def exp_func3(x,a,b,c):

    return a*np.exp(b*x+c)
# 함수 계수 4개

def exp_func4(x,a,b,c,d):

    return a*np.exp(b*x+c) + d
# function for R^2 Score

def r2_score_func(testData, predictData):

    return r2_score(testData, predictData)
from scipy.optimize import curve_fit
# 계수 2개 테스트

# x,y data 정의

xdata = np.linspace(0,6,data_numbers)



y = exp_func2(xdata, 5,-1)

# y_noise = 0.2 * np.random.normal(size=xdata.size)

y_noise = 1*np.random.uniform(-1,1,[data_numbers])

ydata = y + y_noise
popt, pcov = curve_fit(exp_func2, xdata, ydata)

popt
plt.plot(xdata, exp_func2(xdata, *popt), 'r-', label='fit : a = %5.3f, b = %5.3f' % tuple(popt))

plt.plot(xdata,ydata, 'bo', label='data')

plt.xlabel('x')

plt.ylabel('y')

plt.legend()

plt.show()
r2_score_func(ydata, exp_func2(xdata, *popt))
# 계수 3개 테스트

# x,y data 정의

xdata = np.linspace(0,6,data_numbers)



y = exp_func3(xdata, 5,-1,1)

# y_noise = 0.2 * np.random.normal(size=xdata.size)

y_noise = 1*np.random.uniform(-1,1,[data_numbers])

ydata = y + y_noise



popt, pcov = curve_fit(exp_func3, xdata, ydata)

popt
plt.plot(xdata, exp_func3(xdata, *popt), 'r-', label='fit : a = %5.3f, b = %5.3f, c = %5.3f' % tuple(popt))

plt.plot(xdata,ydata, 'bo', label='data')

plt.xlabel('x')

plt.ylabel('y')

plt.legend()

plt.show()
r2_score_func(ydata, exp_func3(xdata, *popt))
# 계수 4개 테스트

# x,y data 정의

xdata = np.linspace(0,6,data_numbers)



y = exp_func4(xdata, 5,-1,1,1)

# y_noise = 0.2 * np.random.normal(size=xdata.size)

y_noise = 1*np.random.uniform(-1,1,[data_numbers])

ydata = y + y_noise



popt, pcov = curve_fit(exp_func4, xdata, ydata)

popt
plt.plot(xdata, exp_func4(xdata, *popt), 'r-', label='fit : a = %5.3f, b = %5.3f, c = %5.3f, d = %5.3f ' % tuple(popt))

plt.plot(xdata,ydata, 'bo', label='data')

plt.xlabel('x')

plt.ylabel('y')

plt.legend()

plt.show()
r2_score_func(ydata, exp_func4(xdata, *popt))
# 임의 데이터 생성을 위한 함수 정의

def log_func(x,a=1,b=1):

    return a*np.log(x)+b
xdata = np.linspace(1,10,data_numbers)



y = log_func(xdata, a=2.5, b=5.89)

y_noise = 1*np.random.uniform(-1,1,[data_numbers])

ydata = y + y_noise
plt.scatter(xdata, ydata)

plt.plot(xdata, y, "r-")
popt, pcov = curve_fit(log_func, xdata, ydata)

popt
plt.scatter(xdata, ydata, label="raw data")

plt.plot(xdata, y, "r-", label="original curve : a = 2.5, b = 5.89")

plt.plot(xdata, log_func(xdata, *popt),"b-", label="fit : a = %5.3f, b = %5.3f " % tuple(popt))

plt.legend()
r2_score_func(ydata, log_func(xdata, *popt))
# define function for having 2 variables.

def multivar_func(x, a,b,c):

    return a*x[0]**2+b*x[1]**2+c*np.sin(x[1])
# this is fake data...

x_set = {

    "x1":np.random.uniform(23,31,[100]),

    "x2":np.random.randint(50,70, [100])

}

xdata = [x_set["x1"],x_set["x2"]]



# we just set up function coeffiecient just like this...

# we will compare this values with regression model .

fake_popt = [10, 1, -7]

y = multivar_func(xdata, *fake_popt)

# y = np.random.uniform(180,200, [100])

y_noise = 2*np.random.uniform(-1,1,[y.size])

ydata = y + y_noise
popt_mul, pcov_mul = curve_fit(multivar_func, xdata, ydata)

popt_mul
# 3차원 그래프를 그리기 위한 라이브라이 임포트

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.tri as mtri
# 1d Data

xdata_1 = xdata[0]

xdata_2 = xdata[1]



# # 2d Dimensioning

# xdata_1 = np.outer(xdata[0],np.ones(xdata[0].size))

# xdata_2 = np.outer(xdata[1],np.ones(xdata[1].size))

# ydata = popt[0]*xdata_1+popt[1]*xdata_2+popt[2]





# 데이터의 연결성을 위해서, 2d 평면에서 삼각형 요소로 이어준다.

triang = mtri.Triangulation(xdata_1, xdata_2)



fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.triplot(triang, c="#D3D3D3", marker='.', markerfacecolor="#DC143C",

    markeredgecolor="black", markersize=15)


fig = plt.figure(figsize=(15,8))

ax = plt.axes(projection="3d")

ax.set_xlabel("max_temp.")

ax.set_ylabel("the num. of people that have a lunch at next door.")

ax.set_zlabel("the num. of Ice Americano sold at the Day")



# Define scatter plot

ax.scatter(xs=xdata_1,ys=xdata_2,zs=ydata, c="black")



# Define trisurf plot

ax.plot_trisurf(triang, multivar_func(xdata,*popt_mul), cmap='jet', edgecolor='green')

ax.view_init(elev=55,azim=-45)
df_IA = pd.DataFrame(x_set)

# 컬럼이 많을 경우에 아래와 같이 바꾸고 싶은 컬럼만 지정해서 설정.

# df_IA.rename(columns={"x1":"temp.","x2":"number of people"}, inplace=True)

df_IA.columns = ['temp.','# of people visited next door for lunch.']

df_IA["ydata_for_model"] = ydata

df_IA["predicted_result"] = multivar_func(xdata,*popt_mul)

df_IA
r2_score_func(testData=ydata, predictData=multivar_func(xdata,*popt_mul))
test_sample = {

    "a1" :  np.array([

        0.0011,0.0013,0.0020,0.0030,0.0013,0.0013,0.0005

    ]),

    "a2" : np.array([

        5.2339, 4.6202, 4.6202, 4.6202, 3.000, 6.000, 4.6202

    ]),

    "result" : np.array([

        2.154, 1.216,3.984, 5.443, 1.140, 3.900, 0.798

    ])

}