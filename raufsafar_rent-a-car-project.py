"""RENT A CAR project """
import datetime


# parent class

class VehilcleRent():

    def __init__(self, stock):
        self.stock = stock
        self.now = 0

    def displaystock(self):
        # display stock
        print('{} vehicle available to rent'.format(self.stock))
        return self.stock

    def renthourly(self, n):
        # rent hourly
        self.n = n
        if self.n <= 0:
            print('Number should be positive')
            return None
        elif n > self.stock:
            print("sorry, {} vehicle available to rent".format(self.stock))
            return None
        else:
            self.now = datetime.datetime.now()
            print('Rented a {} vehicle for hourly at {} hours'.format(n, self.now.hour))
            self.stock -= n
            return self.now

    def rentdaily(self, n):
        # rent daily
        self.n = n
        if self.n <= 0:
            print('Number should be positive')
            return None
        elif n > self.stock:
            print("sorry, {} vehicle available to rent".format(self.stock))
            return None
        else:
            self.now = datetime.datetime.now()
            print('Rented a {} vehicle for daily at {} hours'.format(n, self.now.hour))
            self.stock -= n
            return self.now

    def returnVeh(self, request, brand):
        # return a bill
        car_h_price = 10
        car_d_price = car_h_price * 8 / 10 * 24
        bike_h_price = 5
        bike_d_price = bike_h_price * 7 / 10 * 24

        rentalTime, rentalBasis, numOfVehicle = request
        bill = 0
        if brand == 'car':
            if rentalTime and rentalBasis and numOfVehicle:
                self.stock += numOfVehicle
                now = datetime.datetime.now()
                rentalPeriod = now - rentalTime
                if rentalBasis == 1:  # hourly
                    bill = rentalPeriod.seconds / 3600 * car_h_price * numOfVehicle
                elif rentalBasis == 2:  # daily
                    bill = rentalPeriod.seconds / (3600 * 24) * car_d_price * numOfVehicle
                if (2 <= numOfVehicle):
                    print('you have  extra 20% discount')
                    bill = bill * 0.8
                print('thank you for returing your car')
                print('Price: ${}'.format(bill))
                return bill

        elif brand == 'bike':
            if rentalTime and rentalBasis and numOfVehicle:
                self.stock += numOfVehicle
                now = datetime.datetime.now()
                rentalPeriod = now - rentalTime
                if rentalBasis == 1:  # hourly
                    bill = rentalPeriod.seconds / 3600 * bike_h_price * numOfVehicle
                elif rentalBasis == 2:  # daily
                    bill = rentalPeriod.seconds / (3600 * 24) * bike_d_price * numOfVehicle
                if (4 <= numOfVehicle):
                    print('you have  extra 20% discount')
                    bill = bill * 0.8
                print('thank you for returing your bike')
                print('Price: ${}'.format(bill))
                return bill
        else:
            print('you do not a vehcile')


"""Child class 1"""


class Carrent(VehilcleRent):
    global discount_rate
    discount_rate = 15

    def __init__(self, stock):
        super().__init__(stock)

    def discount(self, b):
        # discount
        bill = b - (b * discount_rate) / 100
        return bill


"""Child class 2"""


class BikeRent(VehilcleRent):

    def __init__(self, stock):
        super().__init__(stock)


"""Customer"""


class Customer():
    def __init__(self):
        self.bike = 0
        self.rentalBasis_b = 0
        self.rentalTime_b = 0

        self.cars = 0
        self.rentalBasis_c = 0
        self.rentalTime_c = 0

    def requestVeh(self, brand):
        # request a vehicle
        """"" Take a request bike or car from customer"""""
        if brand == 'bike':
            bikes = input('How many bikes would you like to rent?')

            try:
                bikes = int(bikes)
            except ValueError:
                print('Number should be Number')
                return -1
            if bikes < 1:
                print('Number of bikes showl be greather than zero')
                return -1
            else:
                self.bikes = bikes
            return self.bikes

        elif brand == 'car':
            cars = input('How many cars would you like to rent?')
            try:
                cars = int(cars)
            except ValueError:
                print('Number should be Number')
                return -1
            if cars < 1:
                print('Number of cars showl be greather than zero')
                return -1
            else:
                self.bikes = cars
            return self.cars

    def returnVeh(self, brand):
        # return a bikes or cars
        if brand == 'bike':
            if self.rentalTime_b and self.rentalBasis_b and self.bikes:
                return self.rentalTime_b and self.rentalBasis_b and self.bikes
            else:
                return 0, 0, 0
        elif brand == 'car':
            if self.rentalTime_c and self.rentalBasis_c and self.cars:
                return self.rentalTime_c, self.rentalBasis_c, self.cars
            else:
                return 0, 0, 0
        else:
            print('Return vehcile error')


# from List import Carrent,BikeRent,Customer

bike = BikeRent(100)
car = Carrent(10)
customer = Customer()

main_menu = True
while True:
    if main_menu:
        print("""
        ********* Vehcile Rental Shop******
        A. Bike Menu
        B. Car Menu
        Q. Exit 
        """)
        main_menu = False

        choice = input('Enter choice: ')
    if choice == 'A' or choice == 'a':
        print("""
        ******* Bike Menu******
        1.Display available bikes
        2.Request a bike on hourly basis $5
        3.Request bike on a daily basis $84
        4.Return a bike
        5.Main Menu
        6.Exit
        """)
        choice = input('Enter choice')

        try:
            choice = int(choice)
        except ValueError:
            print('it is not a integer')
            continue
        if choice == 1:
            bike.displaystock()
            choice = 'A'
        elif choice == 2:
            customer.rentalTime_b = bike.renthourly(customer.requestVeh('bike'))
            customer.rentalBasis_b = 1
            main_menu = True
            print("----------------")
        elif choice == 3:
            customer.rentalTime_b = bike.rentdaily(customer.requestVeh('bike'))
            customer.rentalBasis_b = 2
            main_menu = True
            print("----------------")
        elif choice == 4:
            customer.bil = bike.returnVeh(customer.returnVeh('bike'), 'bike')
            customer.rentalBasis_b, customer.rentalTime_b, customer.bikes = 0, 0, 0
            main_menu = True
        elif choice == 5:
            main_menu = True
        elif choice == 6:
            break
        else:
            print("invalid input,please right number betwwen 1 or 6")
            main_menu = True

    elif choice == 'B' or choice == 'b':
        print("""
        ******* Car Menu******
        1.Display available cars
        2.Request a car on hourly basis $10
        3.Request car on a daily basis $192
        4.Return a car
        5.Main Menu
        6.Exit
        """)

        choice = input('Enter choice')

        try:
            choice = int(choice)
        except ValueError:
            print('it is not a integer')
            continue

        if choice == 1:
            car.displaystock()
            choice = 'A'
        elif choice == 2:
            customer.rentalTime_c = car.renthourly(customer.requestVeh('car'))
            customer.rentalBasis_c = 1
            main_menu = True
            print("----------------")
        elif choice == 3:
            customer.rentalTime_c = car.rentdaily(customer.requestVeh('car'))
            customer.rentalBasis_c = 2
            main_menu = True
            print("----------------")
        elif choice == 4:
            customer.bil = car.returnVeh(customer.returnVeh('car'), 'car')
            customer.rentalBasis_c, customer.rentalTime_c, customer.cars = 0, 0, 0
            main_menu = True
        elif choice == 5:
            main_menu = True
        elif choice == 6:
            break

        else:
            print("invalid input,please enter number betwwen 1-6")
            main_menu = True


    elif choice == 'Q' or choice == 'q':
        break
    else:
        print('Invalid input,pleaee enter A -B-Q')
        main_menu = True
    print('Thank you for using Rental shop')