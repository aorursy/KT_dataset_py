# capitalizing is the way to define a class in Python

class Car:

    '''building out a car and model'''

    

    def __init__(self, brand, year):

        '''Initialize car brand and year'''

        self.brand = brand

        self.year = year

        

    def carData(self):

        '''Car information'''

        car_info = f"{self.year} {self.brand}"

        return car_info.title()

        

class hydrogenCar(Car):

    '''details on a hydrogen car'''

    def __init__(self, brand, year):

        '''initialize attributes of the parent class'''

        super().__init__(brand, year)

        

hy_car = hydrogenCar('Rictor', '2025')

print(hy_car.carData())



        
class Car:

    '''building out a car and model'''

    

    def __init__(self, brand, year):

        '''Initialize car brand and year'''

        self.brand = brand

        self.year = year

        

    def carData(self):

        '''Car information'''

        car_info = f"{self.year} {self.brand}"

        return car_info.title()

        

class hydrogenCar(Car):

    '''details on a hydrogen car'''

    def __init__(self, brand, year):

        '''initialize attributes of the parent class'''

        super().__init__(brand, year)

        self.cellSize = "Large"

    

    def hyCell(self):

        '''information on car cell'''

        print(f"This car is equipped with a {self.cellSize} hydrogen cell")

        

hy_car = hydrogenCar('Rictor', '2025')

print(hy_car.carData())

hy_car.hyCell()
class Car:

    '''building out a car and model'''

    

    def __init__(self, brand, year):

        '''Initialize car brand and year'''

        self.brand = brand

        self.year = year

        

    def carData(self):

        '''Car information'''

        car_info = f"{self.year} {self.brand}"

        return car_info.title()

    

class Cell:

    '''define your cell'''

    def __init__(self, cellSize = "Large"):

        '''initialize attributes of Cell'''

        self.cellSize = cellSize

        

    def hyCell_size(self):

        '''information on car cell'''

        print(f"This car is equipped with a {self.cellSize} hydrogen cell")

        

class hydrogenCar(Car):

    '''details on a hydrogen car'''

    def __init__(self, brand, year):

        '''initialize attributes of the parent class'''

        super().__init__(brand, year)

        self.cell = Cell()



        

hy_car = hydrogenCar('Rictor', '2025')

print(hy_car.carData())

hy_car.cell.hyCell_size()