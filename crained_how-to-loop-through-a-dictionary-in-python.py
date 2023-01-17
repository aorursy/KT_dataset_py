car_1 = {'color': 'red', 'make': 'tesla', 'year': 2020}



# we must create variables to hold the key and value

# this example may not be best to show this. 

# items() returns a list of key-value pairs

for key, value in car_1.items():

    print(f"\nKey: {key}")

    print(f"Value: {value}")
for info, data in car_1.items():

    print(f"\nInfo: {info}")

    print(f"Data: {data}")
# Now let's make this useful

gamers = {'tim': 'Mario Bros', 'sara': 'FIFA', 'benny': 'oblivion'}





for name, game in gamers.items():

    print(f"{name.title()}'s favorite game is {game.title()}")
# loop through just the names

for name in gamers.keys():

    print(name.title())
# change the order

for name in sorted(gamers.keys()):

    print(f"{name.title()}, is an awesome gamer")
# we can also loop through the values

for game in gamers.values():

    print(game.title())