year = input('What "Year" do you want to check ? ')
if ((int(year)%4) == 0):
    a = int(year)+28
    print(f'The year {a} has the same calendar as {year}!')
elif ((int(year)%4) == 1):
    b = int(year)+6
    print(f'The year {b} has the same calendar as {year}!')
elif ((int(year)%4) == 2 or 3):
    c = int(year)+11
    print(f'The year {c} has the same calendar as {year}!')
else :
    print('Invalid Input!')
ref_date  = 1     #1st
ref_month = 1     #January
ref_year  = 2000  #2000
ref_day   = 7     #Saturday

date = input("Enter Date : ")
month = input('Month (January to December: 1 to 12) : ')
year = input('Year : ')

m_diff = 0

if int(month) == 1:
        m_diff = 0
elif int(month) == 2:
        m_diff = 31
elif int(month) == 3:
        m_diff = 60
elif int(month) == 4:
        m_diff = 91
elif int(month) == 5:
        m_diff = 121
elif int(month) == 6:
        m_diff = 152
elif int(month) == 7:
        m_diff = 182
elif int(month) == 8:
        m_diff = 213
elif int(month) == 9:
        m_diff = 244
elif int(month) == 10:
        m_diff = 274
elif int(month) == 11:
        m_diff = 305
elif int(month) == 12:
        m_diff = 335

leap_diff = ((int(year)-2000)//4)*3

diff = ((int(date)-1) + m_diff + (((int(year)-2000)*366)-leap_diff))

day = ((diff-7)%7)

print('\n')
def day_switch(day):
    switcher = {
        1 : 'Sunday',
        2 : 'Monday',
        3 : 'Tuesday',
        4 : 'Wednesday',
        5 : 'Thursday',
        6 : 'Friday',
        0 : 'Saturday',
    }
    return switcher.get(day,' ')
    
print(day_switch(day))
import datetime 
import calendar 
  
def findDay(date): 
    day = datetime.datetime.strptime(date, '%d %m %Y').weekday() 
    return (calendar.day_name[day]) 

date = input('Date (dd mm yyyy) : ')
print(findDay(date))
