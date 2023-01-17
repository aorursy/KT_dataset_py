soma=0

quant=int(input('Quantos numeros vão ser inseridos? '))

guard=[None]*quant

for i in range(quant):

    guard[i]=int(input("Informe um numero: "))

    soma+=guard[i]

media=soma/quant

print('Media:',media)

menor=maior=0

for j in range(quant):

    if (guard[j]>media):

        maior+=1

    if (guard[j]<media):

        menor+=1

print('Existem', menor, 'numero(s) abaixo da media e', maior,'numero(s) acima da media')
quant=int(input('Quantos numeros vão ser inseridos? '))

vet=[None]*quant

for i in range(quant):

    vet[i]=float(input('Informe um numero: '))

for j in range(-1,-quant-1,-1):

    print(vet[j])
control=0

exodia=0

a=[]

while(control>=0) and (exodia<=500):

    control=float(input('Insira um numero, ou um numero negativo para parar: '))

    if(control>=0):

        a.append(control)

        exodia+=1

for i in range(-1,-exodia-1,-1):

    print(a[i])

    
maior=0

A=[]

for i in range (100):

    A.append(float(input('Insira um valor para a lista: ')))

for j in range (100):

    temp=A[j]+A[-j-1]

    if temp>maior:

        maior=temp

        jm=j

print('Maior soma:', maior, 'Posições:',jm+1, 'e seu simetrico', 101-(jm+1))
senha=[]

confirm=[]

vf=1

for i in range(25):

    senha.append(float(input('Insira um numero: ')))

for i in range(25):

    confirm.append(float(input('Repita o numero: ')))

for i in range(25):

    if(senha[i] != confirm[i]):

        vf=vf*0

if(vf==0):

    print('Fracasso!')

else:

    print('Sucesso!')
vf=0

c1=[]

c2=[]

for i in range(15):

    c1.append(float(input('Insira um numero: ')))

for i in range(15):

    c2.append(float(input('Repita o numero: ')))

for i in range(15):

    for j in range(15):

        if(c1[i]==c2[j]):

            c1[i]=c2[j]=None

for i in range(15):

    if (c1[i]!=None):

        vf+=1

if(vf==0):

    print('Sucesso!')

else:

    print('Fracasso!')

            

            
vet=[]

for i in range(21):

    vet.append(float(input('Insira um numero: ')))

for j in range(21):

    for i in range(20):

        if (vet[i]>vet[i+1]):

            temp=vet[i]

            vet[i]=vet[i+1]

            vet[i+1]=temp

print(vet[10])
n=int(input('Insira um valor n entre 1 e 30: '))

while(n>30) or (n<1):

    n=int(input('Insira um valor n entre 1 e 30: '))

save=[None]*n

for i in range (n):

    controle=0

    while (controle!=1):

        save[i]=float(input('Insira um numero real diferente dos anteriores: '))

        controle=0

        for j in range(n):

            if (save[i]==save[j]):

                controle+=1

    print(save[i])

            
n=int(input('Insira um valor inteiro entre 1 e 750: '))

lugar=0

while (n>750) or (n<1):

    n=int(input('Insira um valor inteiro entre 1 e 750: '))

vet =[None]*n

moda =[None]*n #PROPRIEDADE DESGRAÇADA DO PYTHON SE EU FIZER vet=moda=[None]*n os 2 ficam interligados e se eu mudo 1 individualmente ja muda o outro

mq=[0]*n

for i in range(n):

    flag=0

    vet[i]=float(input('Insira um valor real: '))

    for j in range(n):

        if(vet[i]==moda[j]):

            mq[j]+=1

            flag+=1

    if(flag==0):

        moda[lugar]=vet[i]

        mq[lugar]+=1

        lugar+=1

#print(vet)

#print(moda) tabela que foi montada acima

#print(mq)

maiorq=0

qualq=[]

for i in range(n):

    if(mq[i]>maiorq):

        maiorq=mq[i]

for i in range(n):

    if(mq[i]==maiorq):

        qualq.append(moda[i])

    

print('Moda:',qualq, 'Apareceu:', maiorq,'vezes.')
n=int(input('Insira um valor inteiro entre 1 e 380: '))

lugar=0

while (n>380) or (n<1):

    n=int(input('Insira um valor inteiro entre 1 e 380: '))

vet =[None]*n

moda =[None]*n #PROPRIEDADE DESGRAÇADA DO PYTHON SE EU FIZER vet=moda=[None]*n os 2 ficam interligados e se eu mudo 1 individualmente ja muda o outro

mq=[0]*n

for i in range(n):

    flag=0

    vet[i]=float(input('Insira um valor real: '))

    for j in range(n):

        if(vet[i]==moda[j]):

            mq[j]+=1

            flag+=1

    if(flag==0):

        moda[lugar]=vet[i]

        mq[lugar]+=1

        lugar+=1

#print(vet)

#print(moda) tabela que foi montada acima

#print(mq)

maiorq=0

qualq=[]

for i in range(n):

    if(mq[i]>maiorq):

        maiorq=mq[i]

for i in range(n):

    if(mq[i]==maiorq):

        qualq.append(moda[i])

        menor=moda[i]

for i in range(len(qualq)):

    if(qualq[i]<menor):

        menor=qualq[i]

print('Moda:',menor, 'Apareceu:', maiorq,'vezes.')
a=[None]*3

print(len(a))