# assign total zombies to 1

zombies = 1

# set our while loop to run until we are less than or equal to 9

while zombies <= 9:

    print(zombies)

    # adds 1 to the value of zombies until reaches 9

    # this reads like: zombies = (zombies + 1)

    zombies += 1
n = 10

while n > 0:

    # subtract one from n which is 10

    n -= 1

    # run until n is equal to 2

    if n == 2:

        break

    print(n)

print('Loop has completed')
spain = "\nWhat is your favorite city in Spain?"

spain += "\n(enter 'stop' when you are finished.) "



while True:

    city = input(spain)

    if city == 'stop':

        break

else:

    print(f"I see your favorite city in Spain is {city.title()}")
number = 0

while number < 10:

    number += 1

    # % 2 means modulo which is divisible by 2

    if number % 2 == 0:

        continue

    # we should see every number from 0 to 10 that is not divisible by 2

    print(number)
dogs = ['dog1', 'dog2', 'cat']

print(dogs)



# we can go through a while loop and remove unwanted list items

while 'cat' in dogs:

    dogs.remove('cat')

    

print(dogs)