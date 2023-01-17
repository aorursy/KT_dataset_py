while True:
    try:
        age=int(input('Please Enter your age: '))
    except Exception:
        print('Please Enter your age in numbers')
    else:
        break
if age>18:
    print('You can vote')
else:
    print("You can't, grow up!")