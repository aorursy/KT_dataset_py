mile = input("Enter miles:")
mile = float(mile)
km = mile/0.62137
meters = 1000 * km
mile_km = mile/0.62137                                         #mile converted
mile_meter = 1000*mile_km                                     #mile converted
print('{:.2f} miles is euqivalent to '.format(mile))
print('{:6.4f}km / {:6.1f}meters'.format(mile_km , mile_meter))
#print(str(km) + "km/" + str(meters),"meters\n")
name = input ('Enter your name:')
age = input ('Ebter your age:')
#name = str(name)
age = int(age)
age_in_year_2047=age+2047-2020
print('Hi',name,'! In 2047 you will be ', age_in_year_2047, '!')