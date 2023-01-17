car_1 = {'color': 'red', 'make': 'tesla', 'year': 2020}

print(car_1['color'])

print(car_1['make'])
# you can now assign variables to values

awesome_car = car_1['make']

print(f"The most awesomest car is a {awesome_car}")
car_1['tires'] = 'michelin'

print(car_1)
car_2 = {}



car_2['make'] = 'audi'

car_2['year'] = 2019



print(car_2)
car_2['year'] = 2020

print(car_2)
# sometimes you don't need information and want to remove it as it gets old

print(f"Original car: {car_2}")



del car_2['year']

print(f"Car without year: {car_2}")
fav_food = {

    'john': 'lasagna',

    'terry': 'pizza',

    'oliver': 'burger'

    }



food = fav_food['oliver'].title()

print(f"Oliver's favorite food is clearly a {food}")