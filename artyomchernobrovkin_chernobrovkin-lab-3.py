import numpy as np
m=0

n=9

M=np.matrix([[7,9,m+1,m+2],

             [8+n,9,11,3+n],

             [12,14,m+n+2,13],

             [m+3,2+n,9,12]])
M
q=np.matrix([[0.3],[0.2],[0.4],[0.1]])
q
Min_krt=M.min(axis=1)

print(r'min e_{ir}:')

print(Min_krt)
Max_krt=Min_krt.max(axis=0)

print('max e_i:')

print(Max_krt)

print(Min_krt.argmax(axis=0)+1)
BL=np.zeros((4,1))

for i in range(4):

    for j in range(4):

        BL[i,0]+=M[i,j]*q[j,0]

print('e_ir')

print(BL)
Max_e_ir=BL.max(axis=0)

print('max e_ir:')

print(Max_e_ir)

print(BL.argmax(axis=0)+1)
print(M)
Max_e_ij=M.max(axis=0)

print(Max_e_ij)
A=np.zeros((4,4))

for i in range(4):

    for j in range(4):

        A[i,j]=Max_e_ij[0,j]-M[i,j]

print('A:')

print(A)
Max_aij=A.max(axis=1)

print('a_ir')

print(Max_aij)

Min_e_ir=min(Max_aij)

print('Min_e_ir')

print(Min_e_ir)
C=1

Max_e_ir=(C*M.min(axis=1)+(1-C)*M.max(axis=1)).max(axis=0)

print(Max_e_ir)

print((C*M.min(axis=1)+(1-C)*M.max(axis=1)).argmax(axis=0)+1)
C=0

Max_e_ir=(C*M.min(axis=1)+(1-C)*M.max(axis=1)).max(axis=0)

print(Max_e_ir)

print((C*M.min(axis=1)+(1-C)*M.max(axis=1)).argmax(axis=0)+1)
C=0.5

Max_e_ir=(C*M.min(axis=1)+(1-C)*M.max(axis=1)).max(axis=0)

print(Max_e_ir)

print((C*M.min(axis=1)+(1-C)*M.max(axis=1)).argmax(axis=0)+1)
nu=0

XL=np.zeros((4,1))

for i in range(4):

    for j in range(4):

        XL[i,0]+=M[i,j]*q[j,0]

resXL=(nu*XL+(1-nu)* M.min(axis=1)).max(axis=0)

print(resXL)

print((nu*XL+(1-nu)* M.min(axis=1)).argmax(axis=0)+1)
nu=1

XL=np.zeros((4,1))

for i in range(4):

    for j in range(4):

        XL[i,0]+=M[i,j]*q[j,0]

resXL=(nu*XL+(1-nu)* M.min(axis=1)).max(axis=0)

print(resXL)

print((nu*XL+(1-nu)* M.min(axis=1)).argmax(axis=0)+1)
nu=0.5

XL=np.zeros((4,1))

for i in range(4):

    for j in range(4):

        XL[i,0]+=M[i,j]*q[j,0]

resXL=(nu*XL+(1-nu)* M.min(axis=1)).max(axis=0)

print(resXL)

print((nu*XL+(1-nu)* M.min(axis=1)).argmax(axis=0)+1)
G=np.zeros((4,4))

K=M-(M.max()+1)

for i in range(4):

    for j in range(4):

        G[i,j]=K[i,j]*q[j,0]

resG=(G.min(axis=1)).max(axis=0)

print(resG)

print((G.min(axis=1)).argmax(axis=0)+1)
q_rav=np.matrix([[1/4],[1/4],[1/4],[1/4]])

G=np.zeros((4,4))

K=M-(M.max()+1)

for i in range(4):

    for j in range(4):

        G[i,j]=K[i,j]*q[j,0]

print(G)

resG=(G.min(axis=1)).max(axis=0)

print(resG)

print((G.min(axis=1)).argmax(axis=0)+1)
MO=np.zeros((4,1))#1

for i in range(4):

    for j in range(4):

        MO[i,0]+=M[i,j]*q[j,0]

#опорное значение

print('MатOжидание',MO)

opor=(M.min(axis=1)).max(axis=0)

print('Опорное значение',opor)

#print('исходная матрица',M)

position_opr=(M.min(axis=1)).argmax(axis=0)

print('Номер строки опроного значения',position_opr[0,0])



Difference_1=np.zeros((4,1))#2

Difference_1=opor-M.min(axis=1)

print('второй столбец',Difference_1)

Difference_2=np.zeros((4,1))#2

Difference_2=M.max(axis=1)-M[position_opr[0,0]].max(axis=1)

print('третий столбец',Difference_2)
P=np.ones((4,1))

for i in range(4):

    for j in range(4):

        P[i,0]*=M[i,j]

resP=P.max(axis=0)

print(resP)

print(P.argmax(axis=0)+1)