CHOICE=input('which type of convert? A.convert mil to km B.convert km to mil C.convert meter to mil D.convert mil to meter Please Enter Your Choice in CAPITAL LETTER:') 
if CHOICE==str('A'):
    DIS1=input('enter the distance by mile:')
    print(round(float(DIS1)/0.62137,2))
elif CHOICE==str('B'):
    DIS2=input('enter the distance by km:')
    print(round(float(DIS2)*0.62137,2))
elif CHOICE==str('C'):
    DIS3=input('enter the distance by meter:')
    print(round(float(DIS3)/1000*0.62137,2))
elif CHOICE==str('D'):
    DIS4=input('enter the distance by mile:')
    print(round(float(DIS4)/0.62137*1000,2))
a=input('Please enter your name:')
b=input('please enter your age:')
print('Hi',str(a),'your age in 2047 will be:',int(2047-(2013-int(b))))