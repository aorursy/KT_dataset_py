a=input('Enter a number for X:')
b=input('Enter a number for Y:')
c=input('Enter a number for Z:')
d=int(a)+int(b)+int(c)
if int(a)%2==0 and int(b)%2==0 and int(c)%2==0:
    print('the maximum even number is:', max(int(a),int(b),int(c)))
elif int(a)%2==1 and int(b)%2==0 and int(c)%2==1:
    print('the maximum even number is:', int(b))
elif int(a)%2==1 and int(b)%2==1 and int(c)%2==0:
    print('the maximum even number is:', int(c))
elif int(a)%2==0 and int(b)%2==1 and int(c)%2==1:
    print('the maximum even number is:', int(a))
elif int(a)%2==0 and int(b)%2==0 and int(c)%2==1:
    print('the maximum even number is:', max(int(a),int(b)))
elif int(a)%2==1 and int(b)%2==0 and int(c)%2==0:
    print('the maximum even number is:', max(int(b),int(c)))
elif int(a)%2==0 and int(b)%2==1 and int(c)%2==0:
    print('the maximum even number is:', max(int(a),int(c)))
elif int(a)%2==1 and int(b)%2==1 and int(c)%2==1:
    print('all the entered number are odd')
num=int(input('How many Xs do you want?'))
print('X'*int(num))