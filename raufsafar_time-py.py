"""Let's take a look at how we can extract time information from the datetime module. We can create a timestamp by specifying datetime.time(hour,minute,second,microsecond)"""



import datetime



t = datetime.time(4, 20, 1)



# Let's show the different components

print(t)

print('hour  :', t.hour)

print('minute:', t.minute)

print('second:', t.second)

print('microsecond:', t.microsecond)

print('tzinfo:', t.tzinfo)



print('='*50)



"""We can also check the min and max values a time of day can have in the module:"""



print('Earliest  :', datetime.time.min)

print('Latest    :', datetime.time.max)

print('Resolution:', datetime.time.resolution)



print('='*50)



today = datetime.date.today()

print(today)

print('ctime:', today.ctime())

print('tuple:', today.timetuple())

print('ordinal:', today.toordinal())

print('Year :', today.year)

print('Month:', today.month)

print('Day  :', today.day)



print('='*50)



d1 = datetime.date(2015, 3, 11)

print('d1:', d1)



d2 = d1.replace(year=1990)

print('d2:', d2)






