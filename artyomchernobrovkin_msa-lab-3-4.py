import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/Stat3.csv')
data.head(30)
lab_3_data = pd.DataFrame(data, columns = ['X12', 'X2','X3','X4','X6'])
lab_3_data.head(52)
#r_y,x2
lab_3_data.X2.corr(lab_3_data.X12)
#r_y,x3
lab_3_data.X3.corr(lab_3_data.X12)
#r_y,x4
lab_3_data.X4.corr(lab_3_data.X12)
#r_y,x6
lab_3_data.X6.corr(lab_3_data.X12)
import scipy.linalg as spla
lab_3_data.corr()
R=lab_3_data.corr()
import sympy as sym
print(sym.latex(R))
import matplotlib.pyplot as plt
fig = plt.figure()
cf = plt.matshow(lab_3_data.corr()) #отображает массив как матрицу на графике
plt.colorbar(cf, shrink=1)
plt.title('corr')
Z=spla.inv(lab_3_data.corr())
Z
fig = plt.figure()
cf = plt.matshow(Z) #отображает массив как матрицу на графике
plt.colorbar(cf, shrink=1)
plt.title('Z')
spla.det(Z)
r_y_x2= (-Z[0][1])/np.sqrt(Z[0][0]*Z[1][1])
print(r_y_x2)
r_y_x3= (-Z[0][2])/np.sqrt(Z[0][0]*Z[2][2])
print(r_y_x3)
r_y_x4= (-Z[0][3])/np.sqrt(Z[0][0]*Z[3][3])
print(r_y_x4)
r_y_x6= (-Z[0][4])/np.sqrt(Z[0][0]*Z[4][4])
print(r_y_x6)
r=[r_y_x2,r_y_x3,r_y_x4,r_y_x6]
r
n=52
k=4
t_ct_x2=np.abs(r_y_x2) * np.sqrt((n-k-2)/(1-r_y_x2**2))
t_ct_x2
t_ct_x3=np.abs(r_y_x3) * np.sqrt((n-k-2)/(1-r_y_x3**2))
t_ct_x3
t_ct_x4=np.abs(r_y_x4) * np.sqrt((n-k-2)/(1-r_y_x4**2))
t_ct_x4
t_ct_x6=np.abs(r_y_x6) * np.sqrt((n-k-2)/(1-r_y_x6**2))
t_ct_x6
alpha=0.1
n-k-2
t_kp=2.0129
t_ct=[t_ct_x2,t_ct_x3,t_ct_x4,t_ct_x6]
result=np.zeros(len(t_ct))
for i in range(len(t_ct)):
    if(t_ct[i]>t_kp):
        result[i]=1
t_ct
result
from scipy.stats import norm
q = 1 - alpha / 2
u_q=norm.ppf(q)
u_q
z1_yx2=(1/2)*  np.log((1+r_y_x2)/(1-r_y_x2))- ((r_y_x2)/(2*(n-k-1))) - ((u_q)/(np.sqrt(n-k-3))) 
z2_yx2=(1/2)*  np.log((1+r_y_x2)/(1-r_y_x2))- ((r_y_x2)/(2*(n-k-1))) + ((u_q)/(np.sqrt(n-k-3))) 
r=[r_y_x2,r_y_x3,r_y_x4,r_y_x6]
r
print(z1_yx2)
print(z2_yx2)
z1_yx3=(1/2)*  np.log((1+r_y_x3)/(1-r_y_x3))- ((r_y_x3)/(2*(n-k-1))) - ((u_q)/(np.sqrt(n-k-3))) 
z2_yx3=(1/2)*  np.log((1+r_y_x3)/(1-r_y_x3))- ((r_y_x3)/(2*(n-k-1))) + ((u_q)/(np.sqrt(n-k-3))) 
print(z1_yx3)
print(z2_yx3)
z1_yx4=(1/2)*  np.log((1+r_y_x4)/(1-r_y_x4))- ((r_y_x4)/(2*(n-k-1))) - ((u_q)/(np.sqrt(n-k-3))) 
z2_yx4=(1/2)*  np.log((1+r_y_x4)/(1-r_y_x4))- ((r_y_x4)/(2*(n-k-1))) + ((u_q)/(np.sqrt(n-k-3))) 
print(z1_yx4)
print(z2_yx4)
z1_yx6=(1/2)*  np.log((1+r_y_x6)/(1-r_y_x6))- ((r_y_x6)/(2*(n-k-1))) - ((u_q)/(np.sqrt(n-k-3))) 
z2_yx6=(1/2)*  np.log((1+r_y_x6)/(1-r_y_x6))- ((r_y_x6)/(2*(n-k-1))) + ((u_q)/(np.sqrt(n-k-3))) 
print(z1_yx6)
print(z2_yx6)
r1_yx2=(np.exp(2*z1_yx2)-1)/(np.exp(2*z1_yx2)+1)
r2_yx2=(np.exp(2*z2_yx2)-1)/(np.exp(2*z2_yx2)+1)

r1_yx3=(np.exp(2*z1_yx3)-1)/(np.exp(2*z1_yx3)+1)
r2_yx3=(np.exp(2*z2_yx3)-1)/(np.exp(2*z2_yx3)+1)

r1_yx4=(np.exp(2*z1_yx4)-1)/(np.exp(2*z1_yx4)+1)
r2_yx4=(np.exp(2*z2_yx4)-1)/(np.exp(2*z2_yx4)+1)

r1_yx6=(np.exp(2*z1_yx6)-1)/(np.exp(2*z1_yx6)+1)
r2_yx6=(np.exp(2*z2_yx6)-1)/(np.exp(2*z2_yx6)+1)
print(r1_yx2,'<r_yx2<',r2_yx2)
print(r1_yx3,'<r_yx3<',r2_yx3)
print(r1_yx4,'<r_yx4<',r2_yx4)
print(r1_yx6,'<r_yx6<',r2_yx6)
R=lab_3_data.corr()

R=np.matrix(R)
R
spla.det(R)
R_00=R[1:5, 1:5]
R_00
det_R=spla.det(R)
det_R_00=spla.det(R_00)
spla.det(R_00)
hat_R_yx_2=1- (det_R/det_R_00) 
hat_R_yx_2
hat_R_yx=np.sqrt(hat_R_yx_2)
hat_R_yx
n-52
p=4
if(n-p/p>=20):
    print('Треба коригувати')
r=R
R2_sqr_right=1
for i in range(2, len(r)):
    cur_r = r[:i, :i]
    cur_r_inv = np.linalg.inv(cur_r)
    cur_diag = np.diag(cur_r_inv).reshape((-1, 1))
    cur_r_part = -cur_r_inv / np.sqrt(np.dot(cur_diag, cur_diag.T))
    cur_part_y = cur_r_part[0, -1]
    R2_sqr_right *= (1 - cur_part_y ** 2)
R2_sqr = 1 - R2_sqr_right
R2 = np.sqrt(R2_sqr)
R2_sqr
R2
V=(n-1)/(n-p-1)
overline_R_yx=np.sqrt(1-(1-hat_R_yx_2)*V)
overline_R_yx
hat_R_yx_2
alpha
v=(n-p-1)/p
F_ct=hat_R_yx_2/(1-hat_R_yx_2)*v
print(F_ct)
F_kp=2.068464817
if(F_ct>F_kp):
    print('значно відрізняється')
nu2=n-p-1
nu1=(p+n*hat_R_yx_2)**2/(p+2*n*hat_R_yx_2)
print('nu1',np.trunc(nu1))
print('nu2',nu2)
F_ct=hat_R_yx_2/(1-hat_R_yx_2) * (n-p-1/p)
F_kp=1.527214469
if(F_ct>F_kp):
    print('значно від 0')
F_ct
F1=1.527214469
F2=0.629709179

print(hat_R_yx_2*(1-((p+1)/n))/( (1-hat_R_yx_2)*F1) - (p/n),'<','R2','<',hat_R_yx_2*(1-((p+1)/n))/( (1-hat_R_yx_2)*F2) - (p/n))
print(np.sqrt(hat_R_yx_2*(1-((p+1)/n))/( (1-hat_R_yx_2)*F1) - (p/n)),'<','R','<',np.sqrt(hat_R_yx_2*(1-((p+1)/n))/( (1-hat_R_yx_2)*F2) - (p/n)))
lab_4_data = pd.DataFrame(data, columns = ['X2','X3','X4','X6','X12','X11'])
lab_4_data.head(52)
x2=lab_4_data.iloc[:,0]
x2
n=lab_4_data.iloc[:,0].size
overlinex_i=[lab_4_data.iloc[:,0].sum()/n,
             lab_4_data.iloc[:,1].sum()/n,
             lab_4_data.iloc[:,2].sum()/n,
             lab_4_data.iloc[:,3].sum()/n]
overliney_i=[lab_4_data.iloc[:,4].sum()/n,
             lab_4_data.iloc[:,5].sum()/n]
sigma_x=[0,0,0,0]
sigma_y=[0,0]
for i in range(n):
    sigma_x[0]+= (lab_4_data.iloc[i,0]**2-overlinex_i[0])**2
    sigma_x[1]+= (lab_4_data.iloc[i,1]**2-overlinex_i[1])**2
    sigma_x[2]+= (lab_4_data.iloc[i,2]**2-overlinex_i[2])**2
    sigma_x[3]+= (lab_4_data.iloc[i,3]**2-overlinex_i[3])**2
    sigma_y[0]+= (lab_4_data.iloc[i,4]**2-overliney_i[0])**2
    sigma_y[1]+= (lab_4_data.iloc[i,5]**2-overliney_i[1])**2
for k in range (len(sigma_x)):
    sigma_x[k]=np.sqrt(sigma_x[k]/(n-1))
for k in range (len(sigma_y)):
    sigma_y[k]=np.sqrt(sigma_y[k]/(n-1))
Z_standart=np.zeros((52,6))
for i in range(n):
    Z_standart[i,0]= (lab_4_data.iloc[i,0]-overlinex_i[0])/sigma_x[0]
    Z_standart[i,1]= (lab_4_data.iloc[i,1]-overlinex_i[1])/sigma_x[1]
    Z_standart[i,2]= (lab_4_data.iloc[i,2]-overlinex_i[2])/sigma_x[2]
    Z_standart[i,3]= (lab_4_data.iloc[i,3]-overlinex_i[3])/sigma_x[3]
    Z_standart[i,4]= (lab_4_data.iloc[i,4]-overliney_i[0])/sigma_y[0]
    Z_standart[i,5]= (lab_4_data.iloc[i,5]-overliney_i[1])/sigma_y[1]
Standart = pd.DataFrame({'X2': Z_standart[:, 0], 'X3': Z_standart[:, 1],'X4': Z_standart[:, 2],'X6': Z_standart[:, 3],'X12': Z_standart[:, 4],'X11': Z_standart[:, 5], })
Standart
R=Standart.corr()
import sympy as sym
R
fig = plt.figure()
cf = plt.matshow(R) #отображает массив как матрицу на графике
plt.colorbar(cf, shrink=1)
plt.title('corr')
R11=R.iloc[0:4,0:4]
R12=R.iloc[0:4,4:6]
R21=R.iloc[4:6,0:4]
R22=R.iloc[4:6,4:6]

R11
R12
R21
R22
C=spla.inv(R22).dot(R21.dot(spla.inv(R11).dot(R12)))
C
np.linalg.eig(C)
np.linalg.eig(C)
lambda1_2=0.95845115
lambda2_2=0.23576997
B_1=np.matrix([[0.94088108],[0.33873707]])
B_2=np.matrix([[-0.70312797], [0.71106333]])
B_1,B_2
r1=np.sqrt(lambda1_2)
r2=np.sqrt(lambda2_2)
r1,r2
A1=(1/r1) * spla.inv(R11).dot(R12.dot(B_1))

A2=(1/r2) * spla.inv(R11).dot(R12.dot(B_2))
A1,A2
n=52
m=1
p=2
q=4
chi_ct_2_forboth=-(n-m-0.5*(q+p+1))*(np.log((1-r1**2)*(1-r2**2)))
m=2
chi_ct_2_onlyfor_r2=-(n-m-0.5*(q+p+1)+r1**2)*(np.log((1-r2**2)))
chi_ct_2_forboth,chi_ct_2_onlyfor_r2
alpha
chi_kp_2_forboth=13.36157
chi_kp_2_onlyfor_r2=6.25139
if(chi_ct_2_forboth>chi_kp_2_forboth):
    print('з ймовірністю 99\% r1=',r1,' значно відрізняється від 0')
else:
    print('з ймовірністю 99\% r1=',r1,' незначно відрізняється від 0')
if(chi_ct_2_onlyfor_r2>chi_kp_2_onlyfor_r2):
    print('з ймовірністю 99\% r2=',r2,' значно відрізняється від 0')
else:
    print('з ймовірністю 99\% r2=',r2,' незначно відрізняється від 0')
max(r1,r2)
import sympy as sym
x2,x3,x4,x6,y1,y2=sym.symbols('x2,x3,x4,x6,y1,y2')
U1=A1[0,0]*x2+A1[1,0]*x3+A1[2,0]*x4+A1[3,0]*x6
V1=B_1[0,0]*y1+B_1[1,0]*y2
U2=A2[0,0]*x2+A2[1,0]*x3+A2[2,0]*x4+A2[3,0]*x6
V2=B_2[0,0]*y1+B_2[1,0]*y2

print('U1=',U1)
print('V1=',V1)
print('r1=',r1)

print('U2=',U2)
print('V2=',V2)
print('r2=',r2)
Max_abs_A1=max(np.abs(A1))

Min_abs_A1=min(np.abs(A1))

Standart_without_x6 = pd.DataFrame({'X2': Z_standart[:, 0], 'X3': Z_standart[:, 1],'X4': Z_standart[:, 2],'X12': Z_standart[:, 4],'X11': Z_standart[:, 5] })
R_without_x6=Standart_without_x6.corr()
R_without_x6
fig = plt.figure()
cf = plt.matshow(R_without_x6) #отображает массив как матрицу на графике
plt.colorbar(cf, shrink=1)
plt.title('corr')
R11=R_without_x6.iloc[0:3,0:3]
R12=R_without_x6.iloc[0:3,3:5]
R21=R_without_x6.iloc[3:5,0:3]
R22=R_without_x6.iloc[3:5,3:5]
C=spla.inv(R22).dot(R21.dot(spla.inv(R11).dot(R12)))
np.linalg.eig(C)
lambda1_2=0.95257424
lambda2_2=0.14219666

B_1=np.matrix([[ 0.99017423],[0.13983919]])
B_2=np.matrix([[ -0.7007336], [0.71342304]])

r1=np.sqrt(lambda1_2)
r2=np.sqrt(lambda2_2)
r1,r2

A1=(1/r1) * spla.inv(R11).dot(R12.dot(B_1))
A2=(1/r2) * spla.inv(R11).dot(R12.dot(B_2))
A1,A2
n=52# розмір стовбчика
m=1
p=2#кількість r
q=3#кількість x
m=1
chi_ct_2_forboth=-(n-m-0.5*(q+p+1))*(np.log((1-r1**2)*(1-r2**2)))
chi_kp_2_forboth=[alpha,(q-m+1)*(p-m+1)]
m=2
chi_ct_2_onlyfor_r2=-(n-m-0.5*(q+p+1)+r1**2)*(np.log((1-r2**2)))
chi_kp_2_onlyfor_r2=[alpha,(q-m+1)*(p-m+1)]
chi_kp_2_forboth,chi_kp_2_onlyfor_r2
chi_kp_2_forboth=10.64464 
chi_kp_2_onlyfor_r2=4.60517 
if(chi_ct_2_forboth>chi_kp_2_forboth):
    print('з ймовірністю 99\% r1=',r1,' значно відрізняється від 0')
else:
    print('з ймовірністю 99\% r1=',r1,' незначно відрізняється від 0')
if(chi_ct_2_onlyfor_r2>chi_kp_2_onlyfor_r2):
    print('з ймовірністю 99\% r2=',r2,' значно відрізняється від 0')
else:
    print('з ймовірністю 99\% r2=',r2,' незначно відрізняється від 0')
max(r1,r2)
import sympy as sym
x2,x3,x4,x6,y1,y2=sym.symbols('x2,x3,x4,x6,y1,y2')
U1=A1[0,0]*x2+A1[1,0]*x3+A1[2,0]*x4
V1=B_1[0,0]*y1+B_1[1,0]*y2

U2=A2[0,0]*x2+A1[1,0]*x3+A2[2,0]*x4
V2=B_2[0,0]*y1+B_2[1,0]*y2
print('U1=',U1)
print('V1=',V1)
print('r1=',r1)
Max_abs_A1=max(np.abs(A1))
Max_abs_A1
Min_abs_A1=min(np.abs(A1))
Min_abs_A1
Standart_without_x6_x3 = pd.DataFrame({'X2': Z_standart[:, 0],'X4': Z_standart[:, 2],'X12': Z_standart[:, 4],'X11': Z_standart[:, 5] })
R_without_x6_x3=Standart_without_x6_x3.corr()

fig = plt.figure()
cf = plt.matshow(R_without_x6) #отображает массив как матрицу на графике
plt.colorbar(cf, shrink=1)
plt.title('corr')
R11=R_without_x6_x3.iloc[0:2,0:2]
R12=R_without_x6_x3.iloc[0:2,2:4]
R21=R_without_x6_x3.iloc[2:4,0:2]
R22=R_without_x6_x3.iloc[2:4,2:4]
C=spla.inv(R22).dot(R21.dot(spla.inv(R11).dot(R12)))
np.linalg.eig(C)
lambda1_2=0.82898601
lambda2_2=0.20110567

B_1=np.matrix([[0.99955413],[0.02985857]])
B_2=np.matrix([[0.56457045], [0.82538488]])

r1=np.sqrt(lambda1_2)
r2=np.sqrt(lambda2_2)
r1,r2
A1=(1/r1) * spla.inv(R11).dot(R12.dot(B_1))
A2=(1/r2) * spla.inv(R11).dot(R12.dot(B_2))
A1,A2
n=52# розмір стовбчика
m=1
p=2#кількість r
q=2#кількість x
m=1
chi_ct_2_forboth=-(n-m-0.5*(q+p+1))*(np.log((1-r1**2)*(1-r2**2)))
chi_kp_2_forboth=[alpha,(q-m+1)*(p-m+1)]
m=2
chi_ct_2_onlyfor_r2=-(n-m-0.5*(q+p+1)+r1**2)*(np.log((1-r2**2)))
chi_kp_2_onlyfor_r2=[alpha,(q-m+1)*(p-m+1)]
chi_kp_2_forboth,chi_kp_2_onlyfor_r2
chi_kp_2_forboth=7.77944
chi_kp_2_onlyfor_r2=2.70554
if(chi_ct_2_forboth>chi_kp_2_forboth):
    print('з ймовірністю 99\% r1=',r1,' значно відрізняється від 0')
else:
    print('з ймовірністю 99\% r1=',r1,' незначно відрізняється від 0')
if(chi_ct_2_onlyfor_r2>chi_kp_2_onlyfor_r2):
    print('з ймовірністю 99\% r2=',r2,' значно відрізняється від 0')
else:
    print('з ймовірністю 99\% r2=',r2,' незначно відрізняється від 0')

max(r1,r2)
import sympy as sym
x2,x3,x4,x6,y1,y2=sym.symbols('x2,x3,x4,x6,y1,y2')
U1=A1[0,0]*x2+A1[1,0]*x4
V1=B_1[0,0]*y1+B_1[1,0]*y2

U2=A2[0,0]*x2+A1[1,0]*x4
V2=B_2[0,0]*y1+B_2[1,0]*y2
print('U1=',U1)
print('V1=',V1)
print('r1=',r1)
r1=0.9790051838473584
r2=0.910486688535313
z1=0.5*np.log( (1+r1)/(1-r1)   )
z2=0.5*np.log( (1+r2)/(1-r2)   )
t_ct=(z1-z2)*np.sqrt((n-3)/2)
t_kr=[alpha,n-3]
t_kr
t_kr=62.03753663
if(t_ct<t_kr):
    print('r1 та r2 незначимо відрізняються один від одного з ймовірністю 99\%- тому чинники х3 та х6 можна не брати до уваги')
else:
    print('r1 та r2 значимо відрізняються один від одного з ймовірністю 99\%')