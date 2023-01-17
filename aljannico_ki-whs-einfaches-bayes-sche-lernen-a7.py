p_h1, p_h2, p_h3 = 0.5,0.2,0.3

p_D_h1 = 0.2**3
p_D_h2 = 0.5**3
p_D_h3 = 0.9**3
print("P(D|h1) =",p_D_h1,"P(D|h2) =",p_D_h2,"P(D|h3) =",p_D_h3)

print("P(D|h1)*P(h1) =",p_D_h1*p_h1,"P(D|h2)*P(h2) =",p_D_h2*p_h2,"P(D|h3)*P(h3) =",p_D_h3*p_h3)

p_D = p_D_h1*p_h1+p_D_h2*p_h2+p_D_h3*p_h3
p_h1_D=p_D_h1*p_h1/p_D
p_h2_D=p_D_h2*p_h2/p_D
p_h3_D=p_D_h3*p_h3/p_D
print("P(h1|D)=",p_h1_D,"P(h2|D)=",p_h2_D,"P(h3|D) =",p_h3_D)

p_X_D = 0.8*p_h1_D+0.5*p_h2_D+0.1*p_h3_D
print("Die gesuchte Wahrscheinlichkeit: ",p_X_D)
print("Wahrscheinlichkeit für ein brasilianisches Lakritz: ",1-p_X_D)


