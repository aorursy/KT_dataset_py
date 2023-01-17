from sympy import *

x,y = symbols('x y')

# сюда пишем уравнения, f1 = x', f2 = y'

f1 = (x-y+1)**(-I*I/5)+1

f2 = atan(y-2-x)
f1
f2
# особые точки

solve([f1,f2])
# сюда особую точку

x0,y0 = 0,2
# частные производные f1

f1.diff(x)
f1.diff(y)
# частные производные f2

f2.diff(x)
f2.diff(y)
# линейное приближение

x_ = f1.diff(x).subs(x,x0).subs(y,y0)*(x-x0) + f1.diff(y).subs(x,x0).subs(y,y0)*(y-y0) + f1.subs(x,x0).subs(y,y0)

x_
y_ = f2.diff(x).subs(x,x0).subs(y,y0)*(x-x0) + f2.diff(y).subs(x,x0).subs(y,y0)*(y-y0) + f2.subs(x,x0).subs(y,y0)

y_
# Находим ламбды

a = x_.subs(x,1).subs(y,0) - x_.subs(x,0).subs(y,0)

b = x_.subs(x,0).subs(y,1) - x_.subs(x,0).subs(y,0)

c = y_.subs(x,1).subs(y,0) - y_.subs(x,0).subs(y,0)

d = y_.subs(x,0).subs(y,1) - y_.subs(x,0).subs(y,0)



M = Matrix([[a,b],[c,d]])

M
# если будет "list index out of range", то корень 1 кратности 2

print("lambda1 = ",M.eigenvects()[0][0])

print("lambda2 = ",M.eigenvects()[1][0])
#Собственные векторы, если нужны

print("h1 = ",M.eigenvects()[0][2][0])

print("h2 = ",M.eigenvects()[1][2][0])