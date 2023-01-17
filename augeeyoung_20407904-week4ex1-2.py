#exercise 1
miles=float(input('enter the whole miles:'))

print(str(miles)+'miles is equivalent to '+str(round(miles/0.62137,2))+' km / '
      +str(round(miles/0.61237*1000,2))+' meters')
#exercise 2
name=input('Enter your name:')
age=int(input('Enter your age:'))
name = str(name)
age= int(age)
age_in_year_2047=age+27
print('Hi',name,'! In 2047 you will be',age_in_year_2047,'!')