import numpy as np

import pandas as pd
vals=[1,0,0,0,-0.5,0,0,0,-0.5]
s=np.array(vals).reshape(3,3)
s.shape
s
temp=[1/1.414,-1/1.414,0,1/1.414,1/1.414,0,0,0,1]

g=np.array(temp).reshape(3,3)
g
gs=np.dot(g,s)

gs
e=np.dot(gs,g.T)

e
term_a=((e[0][0])**2 + (e[1][1])**2 + (e[2][2])**2)*1.5

term_a
term_b=(3/4)*((2*e[0][1])**2 + ((2*e[1][2]**2)) + ((2*e[2][0])**2))

term_b
term_c=(term_a+term_b)**0.5

term_c
vms=(2/3)*term_c

vms
col_a=[1,0,-1,0,0,0,0.5,0.5,-1,-1,0.5,0.5,0.5,0.5,0.5,0.5,0,0,0,0,-0.5,-0.5,-0.5,-0.5,0,0,0,0]

col_b=[-1,1,0,0,0,0,-1,-1,0.5,0.5,0.5,0.5,0,0,0,0,-0.5,-0.5,-0.5,-0.5,0.5,.5,.5,.5,0,0,0,0]

col_c=[0,-1,1,0,0,0,.5,.5,.5,.5,-1,-1,-.5,-.5,-.5,-.5,.5,.5,.5,.5,0,0,0,0,0,0,0,0]

col_f=[0,0,0,1,0,0,0,0,.5,-.5,0,0,.5,-.5,.5,-.5,0,0,0,0,.5,-.5,.5,-.5,.5,.5,-.5,.5]

col_g=[0,0,0,0,1,0,.5,-.5,0,0,0,0,0,0,0,0,.5,-.5,.5,-.5,.5,.5,-.5,-.5,.5,-.5,.5,.5]

col_h=[0,0,0,0,0,1,0,0,0,0,.5,-.5,.5,.5,-.5,-.5,.5,.5,-.5,-.5,0,0,0,0,-.5,.5,.5,.5]
A=np.array(col_a)* 6**.5

B=np.array(col_b)* 6**.5

C=np.array(col_c)* 6**.5

F=np.array(col_f)* 6**.5

G=np.array(col_g)* 6**.5

H=np.array(col_h)* 6**.5
dW= (-B*e[0][0] + A*e[1][1] + 2*F*e[1][2] + 2*G*e[2][0] + 2*H*e[0][1] )
df=pd.DataFrame(columns=['Stress state','Work done'])
states=np.arange(1,29)

df['Stress state']=states

df.index=df['Stress state']

df['Work done']=dW

df.drop('Stress state',axis=1,inplace=True)
df
max_work= max(abs(dW))

print('Maximum work done:{0:1f}'.format(max_work))
taylor_factor= max_work/vms

print('The Taylor Factor for the above given strain matrix is: {0:1f}'.format(taylor_factor))