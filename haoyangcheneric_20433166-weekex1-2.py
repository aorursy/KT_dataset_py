#EX1
mile = input("Enter miles:")
mile = float(mile)
km = mile/0.62137
meters = 1000 * km
mile_km = mile / 0.62137
mile_meter = 1000* mile_km
print('{} miles is equivalent to'.format(mile))
print('{:6.4f} km / {:6.4f} meters'.format(mile_km,mile_meter))


#Ex2
name = input("what's your name:")
age = input("How old are you:")
name = str(name)
age = int(age)
age_in_year_2047 = age +2047-2020
print('Hi',name,'! In 2047 you will be ',age_in_year_2047,'!')
