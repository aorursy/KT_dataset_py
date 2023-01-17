# We will create multiple dictionaries of cars

car1 = {'color': 'red', 'year': 2019}

car2 = {'color': 'blue', 'year': 2020}

car3 = {'color': 'gray', 'year': 2018}
# now we will create a list of these dictionaries

cars = [car2, car2, car3]
# Now we can loop through the list

for car in cars:

    print(car)
# start with empty inventory of cars

cars = []



# create 10 new cars for inventory

for car_builder in range(10):

    new_cars = {'color': 'gray', 'year': 2021}

    cars.append(new_cars)

    

for car in cars:

    print(car)
# now show the total of the new cars created

print(f"We added {len(cars)} cars to our inventory")
# adding a list of colors

carz = {'color': ['red', 'gray'], 

        'year': 2019}



print(f"Your car is a {carz['year']}")
print(f"Your car is a {carz['year']} and it has the following colors: ")

for colors in carz['color']:

    print(f"{colors}")
logins = {

    'atari23': {

        'fname': 'tim',

        'cars': 'tesla'

    },

    

    'twinn': {

        'fname': 'bill',

        'cars': 'ford'        

    },

}



for username, users_data in logins.items():

    print(f"\nYour Username: {username}")

    print(f"\tYour First name: {users_data['fname'].title()}")

    print(f"\tYour Car: {users_data['cars'].title()}")