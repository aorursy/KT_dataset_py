#1.



print("digite k");

k = float(input());

control = 1;

S = 1;

den = 2;

termo_passado = 1

termo_atual = 1/2

erro = termo_passado - termo_atual;

while(erro >= k):    

    S += termo_atual

    den += 2;

    erro = (termo_atual - termo_passado)*-1;

    termo_passado = termo_atual;

    termo_atual = (1/den)*control;

    control = control * -1;

    print(erro)

print(S);
# 2. Desenvolver um algoritmo para calcular a soma:

## 1 − 1/2 + 1/4 - 1/6 + 1/8... + 1/200 



control = -1

n = 2;

S = 1;

while(n != 200):

    S = S + (1/n)*control;

    n = n + 2;

    control = control * -1;

print(S);
# 3. Faça um algoritmo que gere os primeiros n termos da seqüência de Fibonacci, que é definida

# por recorrência por

# F(1) = 1

# F(2) = 1

# F(n) = F(n − 2) + F(n − 1)



## estruturado



print("Digite um número maior que 2:")

n = int(input());

print("\n");

if(n >2):

    ant1 = 1;

    ant2 = 1;

    cont = 2; 

    print(ant1);

    print(ant2);

    while(cont < n):

        fib = ant1 + ant2;

        print(fib);        

        cont = cont + 1;

        ant1 = ant2;

        ant2 = fib;

else:

    print("Erro. número não é maior que 2");





# 3. Faça um algoritmo que gere os primeiros n termos da seqüência de Fibonacci, que é definida

# por recorrência por

# F(1) = 1

# F(2) = 1

# F(n) = F(n − 2) + F(n − 1)



## com função

        

print("Escreva um número maior que 2: ");

n = int(input());

if(n > 2): 

    def fun(numero):    

        if((numero == 1) or (numero == 2)):

            return 1;

        else:             

            i = numero-2;

            j = numero-1;

            return fun(i) + fun(j); 

    cont = 1;

    print("\n Sequencia: \n")

    print("1");

    while(cont < n):    

        cont = cont + 1;

        print(fun(cont));

        

else:    

    print("Erro. o número não é maior que 2");



    



# 4. O fatorial de um número pode ser expresso pela seguinte relação: n! = 1 × 2 × 3 × . . . × n ,

# para n ≥ 0. Além disso, podemos definí-lo também utilizando a relação de recorrência abaixo:

# f(i) = i × f(i − 1), i > 0

# f(0) = 1



n = int(input(print("Digite um número")));

tot = 1;

aux = n;

while(aux > 1):    

    tot = tot*aux;

    aux = aux -1;

print("fatorial de ",n," = ",tot);

# 7. Fazer um algoritmo que calcule e escreva o valor

# de S:

# 1 + 3/2 + 5/3 + 7/4 + ... + 99/50



num = 3

den = 2

S = 1

while((num !=99) and (den != 50)):

    S = S + num/den;

    num += 2;

    den += 1;

print(S);
# 8. Fazer um algoritmo que calcule e escreva a seguinte soma:

# (2^1)/50 + (2^2)/49 + (2^3)/48 +...+(2^50)/1



S = 0;

exp = 1;

den = 50;

while((exp != 50) and (den != 1)):

    S = S + (2^exp)/den;

    exp = exp + 1;

    den = den - 1;

print(S);
# 9. Fazer um algoritmo para calcular e escrever a

# seguinte soma:

# S = (37*38)/1 + (36*37)/2 + (35*36)/3 +...+ (1*2)/37



num1 = 37;

num2 = 38;

den = 1;

S = 0;



while((num1 !=1) and (num2 != 2) and (den != 37)):

    S = S + num1*num2/den;

    num1 = num1 - 1;

    num2 = num2 - 1;

    den = den +1;

print(S);

    

# 10. Fazer um algoritmo que calcule e escreva o valor

# de S onde: S =

# 1/1 − 2/4 + 3/9 − 4/16 + 5/25 − 6/36 +... − 10/100



# 11. 



Tot = 0

a = 1000

b = 1

while (b < 50):

    Tot = Tot + a / b * - 1

    a = a - 3

    b = b + 1;

print ( Tot );
# 12



control = 1;

s= 0;

cont = 0;

a = 480;

b= 10;

while(cont <= 30):

    s += (a/b) * control;

    control *= -1;

    a-= 5;

    b += 1;

    cont += 1;

print(s)
# 13



pi=0;

s = 0;

coef = 1

control = 1;

cont = 0;

while(cont != 51):

    s += control * (1/(coef**3));

    cont += 1;

    coef += 2;

    control *=-1;

pi = (s*32)**(1/3);

print(pi);
#14

pi = 0;

precisao = 0.0001;

div = 1;

coef = 1;

control = 1;

while( ((div**2)**1/2) >= precisao ):

    div = (4/coef)*control;

    control = control * -1;

    pi = pi + div;

    coef = coef + 2;

print(pi);
#15



print("digite um numero");

s = 0;

x =  int(input());

den = 1;

control = 1

exp = 25;

while(exp != 1):

    s += control*((x**exp)/den);

    control *= -1;

    exp -= 1;

    den += 1;

print("somatória = ", s);
#16



S = 0;

control = 1;

num = 1

dn = 15;

while(dn != 0):

    S += control*(num/(dn**2));

    control = control * -1;

    num *= 2;

    dn -= 1;

    

print(S);