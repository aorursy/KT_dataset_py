N=10000#N=S+E+I+R
S=[-1 for i in range(0,200)]
E=[-1 for i in range(0,200)]
I=[-1 for i in range(0,200)]
R=[-1 for i in range(0,200)]
S[0]=10000
E[0]=0
I[0]=1
R[0]=0

r=20#感染者接触易感者的人数
B=0.03#传染概率
a=0.1#潜伏者转化为感染者概率(潜伏期的倒数)
r2=20#潜伏者接触易感者的人数
B2=0.03#潜伏者传染正常人的概率
y=0.1#康复概率(感染期的倒数)
p=0.5#免疫比例

for i in range(0,199):
    S[i+1]=S[i]-r*B*I[i]*S[i]/N+(1-p)*y*I[i]-r2*B2*E[i]*S[i]/N
    E[i+1]=E[i]+r*B*I[i]*S[i]/N+r2*B2*E[i]*S[i]/N-a*E[i]
    I[i+1]=I[i]+a*E[i]-y*I[i]
    R[i+1]=R[i]+p*y*I[i]
print(S)
print(E)
print(I)
print(R)
    