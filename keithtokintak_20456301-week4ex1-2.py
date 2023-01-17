mile = float (input ("Enter miles:"))
km = mile / 0.62137
meters = 1000 * km
print ('%0.0f mile(s) is equivalent to %0.4f km / %0.1f meters' %(mile, km, meters))
name = input('What is your name? ')
#ask the user to input the name 

age_now = int(input('What is your age? '))
#ask the user to input age for now

age_2047 = age_now + 27
#caculate age in 2047

print('Hi {}! In 2047 you will be {:d}!'.format(name ,age_2047 ))