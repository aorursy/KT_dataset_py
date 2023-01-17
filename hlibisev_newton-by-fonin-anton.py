import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
def trapezoid_matrix(A,b):

        e = 1.0e-30

        M = np.concatenate((A,b),axis=1)

        i, j = 0, 0

        while i != M.shape[0]  and j != M.shape[1]:

            while abs(M[i, j]) <= e:

                print(i,j,"Тут")

                for k in range(i+1, M.shape[0]):

                    if abs(M[k, j]) >= e:

                        M[i, :], M[k, :] = M[k,:].copy(), M[i, :].copy()

                        break

                j = (j+1) if abs(M[i, j]) <= e else j

                if j ==  M.shape[1]:

                    return matrix(M)

            for k in range(i+1,M.shape[0]):

                M[k,:] = M[k, :] - M[k, j]/M[i, j]*M[i, :]



            i+=1

            j+=1

        return M

    

def Gauss_method(A,b):

    M = trapezoid_matrix(A,b)

    answers = np.zeros(M.shape[1]-1)

    i = len(M) - 1

    j = M.shape[1] - 2

    while i != -1 and j!=-1:

            answers[j] = (M[i, -1] - sum(answers * M[i, :-1])) / M[i, j]

            i -= 1

            j -= 1

    return answers.reshape(-1,1)
def df1_dx(x,y):

    return 1



def df1_dy(x,y):

    return(np.cos(y-1))



def df2_dx(x,y):

    return(np.cos(x+1))



def df2_dy(x,y):

    return 1



def f1(x,y):

    return np.sin(y-1)+x-1.3



def f2(x,y):

    return y - np.sin(x+1) -0.8

import plotly.graph_objects as go



# Generate curve data

t = np.linspace(-10, 10, 1000)

x1 = -np.sin(t-1) + 1.3

y1 = t

x2 = t

y2 = np.sin(t+1) +0.8

# Create figure

fig = go.Figure(

    data=[go.Scatter(x=x1, y=y1,

                     name="sin(y-1) + x = 1.3",

                     mode="lines",

                     line=dict(width=2, color="royalblue")),

          go.Scatter(x=x2, y=y2,

                     mode="lines",

                     name="y + sin(x+1) = 0.8",

                     line=dict(width=2, color="firebrick"))])

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Поиск начального приближения для второго задания',

                              font=dict(family='Arial',

                                        size=25,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

fig.update_layout(annotations=annotations)

fig.show()
def S_Newton(xn,yn,df1_dx,df1_dy,df2_dx,df2_dy,f1,f2):

    x_old = float("inf")

    y_old = float("inf")

    while (xn-x_old)**2 + (yn-y_old)**2 > 0.0001:

        An = np.array([[df1_dx(xn,yn),df1_dy(xn,yn)],[df2_dx(xn,yn),df2_dy(xn,yn)]])

        bn = np.array([[-f1(xn,yn)],[-f2(xn,yn)]])

        gh = Gauss_method(An,bn)

        x_old = xn

        y_old = yn

        xn += gh[0][0]

        yn += gh[1][0]

    return(xn,yn)
print("Ответ:",S_Newton(0.5,0.7,df1_dx,df1_dy,df2_dx,df2_dy,f1,f2))
def f(x):

    return(x**2+4*np.sin(x)-2)



def df_dx(x):

    return(2*x+4*np.cos(x))



def d2f_dx2(x):

    return(2 - 4*np.sin(x))
# Локализируем корни графическим способом

t = np.linspace(-10, 10, 1000)

x1 = t

y1 = t**2 + 4*np.sin(t)

x2 = t

y2 = np.zeros(len(t))+2

# Create figure

fig = go.Figure(

    data=[go.Scatter(x=x1, y=y1,

                     name="sin(y-1) + x = 1.3",

                     mode="lines",

                     line=dict(width=2, color="royalblue")),

          go.Scatter(x=x2, y=y2,

                     mode="lines",

                     name="y + sin(x+1) = 0.8",

                     line=dict(width=2, color="firebrick"))])

annotations = []

annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,

                              xanchor='left', yanchor='bottom',

                              text='Поиск начального приближения для первого задания',

                              font=dict(family='Arial',

                                        size=25,

                                        color='rgb(37,37,37)'),

                              showarrow=False))

fig.update_layout(annotations=annotations)

fig.show()
x0 = 0.47
print("1) f(0.4)*f(0.5) = "+str(f(0.4)*f(0.5))+",","то есть < 0\n")



print("2) 2ая производная не имеет корней на (0.4,0.5)")

print(" 2ая производная: 2 - 4*sin(x) => sin(x) = 1/2 => x = arcsin(1/2)")

print(" arcsin(1/2) = ",np.arcsin(1/2), "не принадлежит [0.4, 0.5]")

print(" Так как 2ая производная в 0.4 =",d2f_dx2(0.4),"и не обнуляется на отрезке, то она положительна\n")



print("3) Получается, если 2ая производная не обнуляется и положительна,\n значит первая производная растет на [0.4, 0.5]")

print(" А значение первой производной в 0.4 =", df_dx(0.4))

print(" Значит у первой производной тоже нет корней на [0.4, 0.5]\n")



print('4) Проверим условия начального приближения: \n f(x0)*f"(x0)>0')

print(' f(x0)*f"(x0) =',f(x0)*d2f_dx2(x0)," => все условия выполнены")
def Newton(xn,f,df_dx):

    x_old = float("inf")

    while abs(xn-x_old)>0.0001:

        x_old = xn

        xn -= f(xn)/df_dx(xn)

    return xn
print("Ответ:",Newton(x0,f,df_dx))