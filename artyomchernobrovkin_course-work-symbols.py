import numpy as np

import sympy as sym
alpha,x,y,G1,G2,A1,B1,A2,B2,p_alpha,b,b1,a,n,xi,p_xi= sym.symbols('alpha,x,y,G1,G2,A1,B1,A2,B2,p_alpha,b,b1,a,n,xi,p_xi')
exp=sym.exp

sinh=sym.sinh

pi=sym.pi

alpha = (pi)/a

cos=sym.cos
W1=A1*exp(alpha*y)+B1*exp(-alpha*y)

W2=A2*exp(alpha*y)+B2*exp(-alpha*y)

W1_der=sym.diff(W1,y)

W2_der=sym.diff(W2,y)

print('W1_der',W1_der)

print('W2_der',W2_der)

Solve_pairing=sym.solve([(W1-W2).subs({y:b1}),(G1*W1_der-G2*W2_der).subs({y:b1})],[A2,B2])

print('solve',Solve_pairing)
A2=A1*G1/(2*G2) + A1/2 - B1*G1*exp(-2*alpha*b1)/(2*G2) + B1*exp(-2*alpha*b1)/2

B2=(-A1*G1*exp(2*alpha*b1) + B1*G1 + G2*(A1*exp(2*alpha*b1) + B1))/(2*G2)

print('A2',sym.latex(A2))

print('B2',sym.latex(B2))

W2=W2.subs({'A2':A2,'B2':B2})

print('W2',W2)
W1_der= sym.diff(W1,y).subs({y:0})

W2_der= sym.diff(W2,y).subs({y:b})

print('W1_der',W1_der)

print('W2_der',W2_der)
Solve=sym.solve([W1_der,W2_der-(p_alpha/G2)],[A1,B1])

print('solve',Solve)

print('solve',sym.latex(Solve))

Kramer=[[W1_der.subs({A1:1,B1:0}),W1_der.subs({A1:0,B1:1})],

         [W2_der.subs({A1:1,B1:0}),W2_der.subs({A1:0,B1:1})] ]

print('det',sym.simplify(sym.det(sym.Matrix(Kramer))))
A1=-2*p_alpha*exp(alpha*(b + 2*b1))/(alpha*(G1*exp(2*alpha*b) + G1*exp(2*alpha*b1) - G2*exp(2*alpha*b) + G2*exp(2*alpha*b1) - (G1*exp(2*alpha*b) + G1*exp(2*alpha*b1) + G2*exp(2*alpha*b) - G2*exp(2*alpha*b1))*exp(2*alpha*b1)))

B1=-2*p_alpha*exp(alpha*(b + 2*b1))/(alpha*(G1*exp(2*alpha*b) + G1*exp(2*alpha*b1) - G2*exp(2*alpha*b) + G2*exp(2*alpha*b1) - (G1*exp(2*alpha*b) + G1*exp(2*alpha*b1) + G2*exp(2*alpha*b) - G2*exp(2*alpha*b1))*exp(2*alpha*b1)))

W1=W1.subs({'A1':A1,'B1':B1})

W2=W2.subs({'A1':A1,'B1':B1})



W1=sym.simplify(W1)

W2=sym.simplify(W2)



print('!W1',sym.latex(W1))

print('!W2',sym.latex(W2))

print('!!!!!!W1',W1)

print('!!!!!!W2',W2)
'''

#GПроверка

W1=W1.subs({'G1':G2})

W2=W2.subs({'G1':G2})

print('GПрвоерка',sym.simplify(W2-W1))

'''
W1_checked=sym.sin(alpha*x)* 2*p_alpha*(exp(2*alpha*y) + 1)*exp(alpha*(b + 2*b1 - y))/(alpha*(-G1*exp(2*alpha*b) + G1*exp(4*alpha*b1) - G1*exp(2*alpha*b1) + G1*exp(2*alpha*(b + b1)) + G2*exp(2*alpha*b) - G2*exp(4*alpha*b1) - G2*exp(2*alpha*b1) + G2*exp(2*alpha*(b + b1))))
W1_checked_afterDecExp=(2*p_alpha*(exp(-alpha*(b-y))+exp(-alpha*(b+y)))*sym.sin(alpha *x))/(alpha*( (G1+G2)*(1-exp(-2*alpha*b))+(G1-G2)*(exp(-2*alpha*(b-b1))-exp(-2*alpha*b1))))

print('Check W1',sym.simplify(W1_checked-W1_checked_afterDecExp))

print('W1_checked_afterDecExp',sym.latex(W1_checked_afterDecExp))

W1_checked_der_y=sym.diff(W1_checked,y)

W1_checked_der_x=sym.diff(W1_checked,x)

W1_checked_der2_y=sym.diff(W1_checked_der_y,y)

W1_checked_der2_x=sym.diff(W1_checked_der_x,x)

print('W1_checked_der2_x + W1_checked_der2_y',sym.simplify(W1_checked_der2_x+W1_checked_der2_y))

print('W1_checked y=0',W1_checked_der_y.subs({y:0}))



print('W1_checked x=0',W1_checked.subs({x:0}))

print('W1_checked x=a',W1_checked.subs({x:a}))
W2_checked=-sym.sin(alpha*x)*p_alpha*(-G1*exp(4*alpha*b1) + G1*exp(2*alpha*b1) - G1*exp(2*alpha*y) + G1*exp(2*alpha*(b1 + y)) + G2*exp(4*alpha*b1) + G2*exp(2*alpha*b1) + G2*exp(2*alpha*y) + G2*exp(2*alpha*(b1 + y)))*exp(alpha*(b - y))/(G2*alpha*(G1*exp(2*alpha*b) - G1*exp(4*alpha*b1) + G1*exp(2*alpha*b1) - G1*exp(2*alpha*(b + b1)) - G2*exp(2*alpha*b) + G2*exp(4*alpha*b1) + G2*exp(2*alpha*b1) - G2*exp(2*alpha*(b + b1))))
W2_checked_afterDecExp=(p_alpha*sym.sin(alpha*x)*((G1-G2)*(exp(-alpha*(b-y+2*b1))+exp(-alpha*(b+y-2*b1)))-(G1+G2)*(exp(-alpha*(b-y))+exp(-alpha*(b+y)))))/(G2*alpha*((G1-G2)*(exp(-2*alpha*b1)-exp(-2*alpha*(b-b1)))+(G1+G2)*(exp(-2*alpha*b)-1)))

print('Check W2',sym.simplify(W2_checked-W2_checked_afterDecExp))

print('W2_checked_afterDecExp',sym.latex(W2_checked_afterDecExp))



W2_checked_der_y=sym.diff(W2_checked,y)

W2_checked_der_x=sym.diff(W2_checked,x)

W2_checked_der2_y=sym.diff(W2_checked_der_y,y)

W2_checked_der2_x=sym.diff(W2_checked_der_x,x)

print('W2_checked_der2_x + W2_checked_der2_y',sym.simplify(W2_checked_der2_x+W2_checked_der2_y))

print('W2_checked_der_y y=b',sym.latex(sym.simplify(W2_checked_der_y.subs({y:b}))))

print('W2_checked x=0',W2_checked.subs({x:0}))

print('W2_checked x=a',W2_checked.subs({x:a}))
print('ПРОВЕРКА',sym.simplify(W1_checked_afterDecExp.subs({y:b1})-W2_checked_afterDecExp.subs({y:b1})))



W1_C1,W1_C2,W1_C3,W1_C4,W1_t0,W1_t1,W1_x0,W1_x1,Q1,Q2= sym.symbols('W1_C1,W1_C2,W1_C3,W1_C4,W1_t0,W1_t1,W1_x0,W1_x1,Q1,Q2')
e=sym.exp

exp=sym.exp

sinh=sym.sinh

pi=sym.pi

#alpha = (pi)/a

cos=sym.cos

ln=sym.log

ch=sym.cosh
W1_t0=alpha*(b+y)

W1_t1=alpha*(b-y)

W1_x0=alpha*(xi-x)

W1_x1=alpha*(xi+x)
W1_C1=-ln(ch(W1_t1)-cos(W1_x0))

W1_C2=-ln(ch(W1_t0)-cos(W1_x0))

W1_C3=ln(ch(W1_t0)-cos(W1_x1))

W1_C4=ln(ch(W1_t1)-cos(W1_x1))
W1_C1_dery=W1_C1.diff(y)

W1_C2_dery=W1_C2.diff(y)

W1_C3_dery=W1_C3.diff(y)

W1_C4_dery=W1_C4.diff(y)
print(W1_C1_dery)

print(W1_C2_dery)

print(W1_C3_dery)

print(W1_C4_dery)
Q1=(e(-alpha*(b+y))+e(-alpha*(b-y)))/(alpha*((1-e(-2*alpha*b))*(G1+G2)+(e(-2*alpha*(b-b1))-e(-2*alpha*b1))*(G1-G2)))

Q2=(e(-alpha*(b+y))+e(-alpha*(b-y)))/(alpha*(G1+G2))
W1_series= (cos(xi-x)-cos(xi+x))*(Q1-Q2)
W1_series_dery=W1_series.diff(y)
sym.simplify(W1_series_dery)

W1_series_dery
print((W1_series_dery))
W1_C1_derx=W1_C1.diff(x)

W1_C2_derx=W1_C2.diff(x)

W1_C3_derx=W1_C3.diff(x)

W1_C4_derx=W1_C4.diff(x)
print(W1_C1_derx)

print(W1_C2_derx)

print(W1_C3_derx)

print(W1_C4_derx)
W1_series_derx=W1_series.diff(x)
sym.simplify(W1_series_derx)
print((W1_series_derx))
W2_C1,W2_C2,W2_C3,W2_C4,W2_C5,W2_C6,W2_C7,W2_C8,W2_t0,W2_t1,W2_t2,W2_t3,W2_x0,W2_x1,Q3,Q4=sym.symbols('W2_C1,W2_C2,W2_C3,W2_C4,W2_C5,W2_C6,W2_C7,W2_C8,W2_t0,W2_t1,W2_t2,W2_t3,W2_x0,W2_x1,Q3,Q4')
W2_t0=alpha*(b+2*b1-y)

W2_t1=alpha*(b-2*b1+y)

W2_t2=alpha*(b+y)

W2_t3=alpha*(b-y)

W2_x0=alpha*(xi-x)

W2_x1=alpha*(xi+x)
W2_C1=-ln(ch(W2_t0)-cos(W2_x0))

W2_C2=-ln(ch(W2_t1)-cos(W2_x0))

W2_C3=ln(ch(W2_t2)-cos(W2_x0))

W2_C4=ln(ch(W2_t3)-cos(W2_x0))

W2_C5=-ln(ch(W2_t2)-cos(W2_x1))

W2_C6=ln(ch(W2_t0)-cos(W2_x1))

W2_C7=ln(ch(W2_t1)-cos(W2_x1))

W2_C8=-ln(ch(W2_t3)-cos(W2_x1))
W2_C1_dery=W2_C1.diff(y)

W2_C2_dery=W2_C2.diff(y)

W2_C3_dery=W2_C3.diff(y)

W2_C4_dery=W2_C4.diff(y)

W2_C5_dery=W2_C5.diff(y)

W2_C6_dery=W2_C6.diff(y)

W2_C7_dery=W2_C7.diff(y)

W2_C8_dery=W2_C8.diff(y)
print(W2_C1_dery)

print(W2_C2_dery)

print(W2_C3_dery)

print(W2_C4_dery)

print(W2_C5_dery)

print(W2_C6_dery)

print(W2_C7_dery)

print(W2_C8_dery)
Q3=((G1-G2)*(e(-W2_t0)+e(-W2_t1))-(G1+G2)*(e(-W2_t2)+e(-W2_t3)))/(alpha*( (G1+G2)*(e(-2*alpha*b)-1)+(G1-G2)*(-e(-2*alpha*(b-b1))+e(-2*alpha*b1))))

Q4=((G1-G2)*(e(-W2_t0)+e(-W2_t1))-(G1+G2)*(e(-W2_t2)+e(-W2_t3)))/alpha*(G1+G2)
W2_series= (cos(xi-x)-cos(xi+x))*(Q3-Q4)
W2_series_dery=W2_series.diff(y)
W2_series_dery
print(W2_series_dery)
W2_C1_derx=W2_C1.diff(x)

W2_C2_derx=W2_C2.diff(x)

W2_C3_derx=W2_C3.diff(x)

W2_C4_derx=W2_C4.diff(x)

W2_C5_derx=W2_C5.diff(x)

W2_C6_derx=W2_C6.diff(x)

W2_C7_derx=W2_C7.diff(x)

W2_C8_derx=W2_C8.diff(x)
print(W2_C1_derx)

print(W2_C2_derx)

print(W2_C3_derx)

print(W2_C4_derx)

print(W2_C5_derx)

print(W2_C6_derx)

print(W2_C7_derx)

print(W2_C8_derx)
W2_series_derx=W2_series.diff(x)
W2_series_derx
print(W2_series_dery)