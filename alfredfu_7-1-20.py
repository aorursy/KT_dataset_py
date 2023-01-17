height = str(input('what is your height? Answer in short,average, or tall.'))
if height == str('short'):
    print('Do you like plant?')
    length = 6
    plant = input('[Yes] or [No]?')
    if plant == 'Yes':
        hair = 'Phoenix feather'
        print(hair)
    if plant == 'No':
        hair = 'Unicorn hair'
        print(hair)
elif height == str('average'):
    print('Are you brave or smart?')
    length = 9
    CHR = input('[brave] or [smart]?')
    if CHR == 'brave':
        hair = 'Phoenix feather'
        print(hair)
    if CHR == 'smart':
        hair = 'Unicorn hair'
        print(hair)
elif height == str('tall'):
    print('Which one do you prefer? TV or movies.')
    length = 12
    video = input('[TV] or [movies]?')
    if video == 'TV':
        hair = 'Phoenix feather'
        print(hair)
    if video == 'movies':
        hair = 'Unicorn hair'
        print(hair)
month = str(input('Which month were you born in? Answer in [even] or [odd].'))
if month == str('even'):
    print('Redwood wand')
elif month == str('odd'):
    print('Elm wood wand')
num = str(input('the number customer gave me'))
Totalnum = int(num)%6
if Totalnum == 0:
    print('you got the power stone, loser.')
if Totalnum == 1:
    print('you got the space stone, loser.')
if Totalnum == 2:
    print('you got the reality stone, loser.')
if Totalnum == 3:
    print('you got the soul stone, loser.')
if Totalnum == 4:
    print('you got the time stone, loser.')
if Totalnum == 5:
    print('you got the mind stone, winner.')