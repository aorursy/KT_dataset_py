def car_info(car_brand, car_model):

    '''Information about car'''

    #note the space between car_brand and car_model for later

    full_car_info = f"{car_brand} {car_model}"

    return full_car_info.title()



# here we are passing our return values to the function

car = car_info('audi', 'q5')

print(car)
def car_info(car_brand, car_model, car_year=''):

    if car_year:

        full_car_info = f"{car_brand} {car_model} {car_year} "

    else:

        full_car_info = f"{car_brand} {car_model}"

    return full_car_info.title()

        

car = car_info('audi', 'q5')

print(car)
# now with the year

car = car_info('toyota', 'rav4', '2019')

print(car)
def build_car(car_brand, car_model):

    '''Return a dictionary about a car'''

    car = {'brand': car_brand, 'model': car_model}

    return car



car_info = build_car('audi', 'q5')

print(car_info)
def chatbot(names):

    '''Print greeting'''

    for name in names:

        ello = f"Hello there {name}"

        print(ello)

        

login = ['Tipsy', 'KillerQueeN']

chatbot(login)

# * tells Python to make an empty tuple and allow all the values it gets

def salad(*ingredients):

    '''list all the salad ingredients'''

    print(ingredients)

    

salad('lettuce')

salad('tomatoes', 'cheese')
def salads(lettuce, veg, **extras):

    '''Build a dictionary with salads'''

    extras['lettuce_type'] = lettuce

    extras['veg_type'] = veg

    return extras



full_salad = salads('romaine', 'tomato',

                   cheese='cheddar',

                   topper='pepperoni')



print(full_salad)