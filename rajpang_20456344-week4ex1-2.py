#Finger Exercise 1
#km=miles/0.62137
#km=1000meters
print ('Enter miles:')
mile = input("Enter miles:")
mile = float(mile)
km = mile/0.62137
meters = 1000 * km
mile_km = mile/0.62137
mile_meter = 1000*mile_km
print('{:.2f}miles is equivalent to '.format(mile))
print('{:.4f}km / {:.2f}meters' .format(mile_km , mile_meter))
name = input ('What is your name? ')
age = input ('What is your age? ')
age = int (age)
age_in_year_2047=age+2047-2020
print ('Hi',name,'! In 2047 you will be' ,age_in_year_2047,'!')