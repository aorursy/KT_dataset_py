import datetime as dt



x = dt.datetime.now()

print("a. Current date and time: ",x)

print("b. Current year: ",x.year)

print("c. Month of year: ",x.month)

print("d. Week number of the year: ",x.isocalendar()[1])

print("e. Weekday of the week: ",x.strftime("%A"))

print("f. Day of year: ",x.timetuple().tm_yday)

print("g. Day of the month ",x.day)

print("h. Day of week: ",x.isocalendar()[2])
sample_date = dt.datetime.strptime('Jul 1 2014  2:43PM', '%b %d %Y %I:%M%p')

print(sample_date)
y = x - dt.timedelta(5)

print(y)
temp = int("1284105682")



print(dt.datetime.utcfromtimestamp(temp).strftime('%Y-%m-%d %H:%M:%S'))
import pandas as pd

DoB = pd.Series(["07Sep59","01Jan55","15Dec47","11Jul42"])

dob = DoB.apply(lambda x: dt.datetime.strptime(x,"%d%b%y"))

dob = dob - pd.offsets.DateOffset(years=100)

print(dob)
date1 = dt.date(2020, 2, 25)

date2 = dt.date(2019, 8, 25)

diffrnce = date1 - date2

print(diffrnce.days)
date1 = "15Dec1989"

b_date = dt.datetime.strptime(date1,"%d%b%Y")

f_date = dt.datetime.strftime(b_date,"%A, %d %b %y")

f_date