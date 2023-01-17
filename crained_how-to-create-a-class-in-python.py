# capitalizing is the way to define a class in Python

class Car:

    '''building out a car and model'''

    

    def __init__(self, brand, year):

        '''Initialize car brand and year'''

        self.brand = brand

        self.year = year

        

    def drive(self):

        '''Car is now driving'''

        print(f"{self.brand} is now driving")

        

    def insane_mode(self):

        '''Car is now using insane mode'''

        print(f"{self.brand} can now go insanely fast!")
my_car = Car('Tesla', 2020)

# we can access attributes of Car instance with dot notation

# which is my_car.brand which is the same as self.brand

print(f"My car is a {my_car.brand}.")

print(f"My car is a {my_car.year} {my_car.brand}.")
your_car = Car('BMW', 2020)

print(f"Your car is a {your_car.year} {your_car.brand}.")
# capitalizing is the way to define a class in Python

class Car:

    '''building out a car and model'''

    

    def __init__(self, brand, model, year):

        '''Initialize car brand and year'''

        self.brand = brand

        self.model = model

        self.year = year

        

    def car_description(self):

        '''Car description'''

        full_car = f"{self.year} {self.brand} {self.model}"

        

        return full_car.title()

        

total_car = Car('BMW', 'series 4', 2019)

print(total_car.car_description())
# capitalizing is the way to define a class in Python

class Car:

    '''building out a car and model'''

    

    def __init__(self, brand, model):

        '''Initialize car brand and year'''

        self.brand = brand

        self.model = model

        # set the default year

        self.year = 2020

        

    def car_description(self):

        '''Car description'''

        full_car = f"{self.brand} {self.model}"

        

        return full_car.title()

        

    def car_year(self):

        '''print car year'''

        print(f"{self.year}")

        

total_car = Car('BMW', 'series 4')

print(total_car.car_description())

total_car.car_year()
total_car.year = 2021

total_car.car_year()
# capitalizing is the way to define a class in Python

class Car:

    '''building out a car and model'''

    

    def __init__(self, brand, model):

        '''Initialize car brand and year'''

        self.brand = brand

        self.model = model

        # set the default year

        self.year = 2020

        

    def car_description(self):

        '''Car description'''

        full_car = f"{self.brand} {self.model}"

        

        return full_car.title()

        

    def car_year(self):

        '''print car year'''

        print(f"{self.year}")

        

    def update_year(self, new_year):

        '''set new year'''

        self.car_year = new_year

        

total_car = Car('BMW', 'series 4')

print(total_car.car_description())



total_car.update_year(2022)

total_car.car_year