dogs = ["Oliver", "Duncan", "sushi", "ZEUS"]
print(dogs)
for dog in dogs:

    if dog == 'ZEUS':

        print(dog.title())

    else:

        print(dog.lower())
dog = 'oliver'
# == means is the value equal to 'oliver'?

dog == 'oliver'
dog2 = 'Zeus'
dog2 == 'Yues'
# but what if it is upper case? We can ignore that if we like

dog3 = 'SuSHi'
dog3.lower() == 'sushi'
# What if you want to know how to handle something when it is false



computer = 'macbook'



if computer != 'insurance':

    print("No coverage")
# numbers can be used as well

gpu = 4

gpu == 4
# now see if they are equal



if gpu != 8:

    print("You are using a slow computer")
gpu2 = 8

gpu2 < 12
gpu2 > 10
gpu2 <=9
gpu2 >= 6
# check multiple numeric values

gpu > 3 and gpu2 < 6
gpu > 3 or gpu2 < 6