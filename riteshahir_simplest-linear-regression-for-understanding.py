import numpy as np

x=[1,2,3,4,5,6,7,8,9,10]

y=[21,26,85,105,46,56,95,73,81,35]

xarr=np.array(x)

yarr=np.array(y)

print("x is :-",xarr)

print("y is :-",yarr)
nOfX=len(x)

nOfY=len(y)

print("n is :-",nOfX)

print("n is :-",nOfY)
xBar=sum(x)/nOfX

yBar=sum(y)/nOfY

print("xbar is :- ",xBar)

print("ybar is :- ",yBar)
x_xBar=0

for i in range (0,nOfX):

    x_xBar=xarr-xBar

print("(x-xbar) is :- ",x_xBar)

print("total is Σ :-",sum(x_xBar))
y_yBar=0

for i in range (0,nOfY):

    y_yBar=yarr-yBar

print("(x-xbar) is :- ",y_yBar)

print("total is Σ :-",sum(y_yBar))
x_xBary_yBar=0

for i in range (0,10):

    x_xBary_yBar=x_xBar*y_yBar

print("(x-xbar)(y-ybar) is:-",x_xBary_yBar)

print("total is Σ:- ",sum(x_xBary_yBar))
x_xBar_squ=(x_xBar*x_xBar)

print("(x-xbar)^2 is :- ",x_xBar_squ)

print("total is :-",sum(x_xBar_squ))
b=sum(x_xBary_yBar)/sum(x_xBar_squ)

print(b)
a=yBar-b*xBar

print(a)
user=input("Enter value from x :-")

print("Entered value is",user)



t=float(user)

y1=a+(b*t)

print(y1)