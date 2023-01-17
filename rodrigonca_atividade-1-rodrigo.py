def test_for_prime_for_loop(x):

    if isinstance(x, int) and x > 0:

        if x == 1: 

            is_prime = (True, f'it is divisible by 1 and {x}')

        elif x == 2:

            is_prime = (False, f'it is divisible by 1 and {x}')

        else:

            i = 2

            for i in range(2, abs(x), 1):

                if x % i == 0:

                    is_prime = (False, f'it is divisible by 1 and {i}')

                    break

                else:

                    is_prime = (True, f'it is divisible by 1 and {x}')

        return is_prime

    else:

        return (False, 'non integer input or not greater than 0')
for test in range(1, 50, 1):

    is_prime = test_for_prime_for_loop(test)[0]

    if is_prime:

        print(f'number: {test} is prime? {test_for_prime_for_loop(test)}')
def test_for_prime_while_loop(x):

    if isinstance(x, int) and x > 0:

        if x == 1: 

            is_prime = (True, f'it is divisible by 1 and {x}')

        elif x == 2:

            is_prime = (False, f'it is divisible by 1 and {x}')

        else:

            i = 2

            while i < x:

                if x % i == 0:

                    is_prime = (False, f'it is divisible by 1 and {i}')

                    break

                else:

                    is_prime = (True, f'it is divisible by 1 and {x}')

                i += 1

        return is_prime

    else:

        return (False, 'non integer input or not greater than 0')
for test in range(1, 50, 1):

    is_prime = test_for_prime_while_loop(test)[0]

    if is_prime:

        print(f'number: {test} is prime? {test_for_prime_while_loop(test)}')
def is_cpf(cpf):

    # checks if it is an string and then converts to integer

    # if the conversion fails it means that the input is not numeric.

    if isinstance(cpf, str):

        try:

            cpf = int(cpf)

        except ValueError:

            return (False, 'non valid input')

        

    # convert the result from above operation to a string with 11 characters and leading zeros. 

    if isinstance(cpf, int):

        if len(str(cpf)) < 12 and int(cpf) > 0:

            cpf = "{:0>11d}".format(cpf)

            # checks if there is repetition in the input data

            if (str(cpf) + str(cpf)).find(str(cpf), 1, -1)  != -1: 

                return (False, 'repeated numbers')



            # extract cpf root for check digit calculation

            root_cpf = cpf[0:9]

            

            # calculates the first check digit

            digt_1 = 0

            for c, i in zip(root_cpf, list(range(10, 1, -1))):

                digt_1 = int(c) * i + digt_1

            digt_1 = digt_1 * 10 % 11

            if digt_1 == 10: digt_1 = 0

            

            # concatenates the first check digit to cpf root to calculate the second check digit

            root_cpf = cpf[0:9] + str(digt_1)

            

            # calculates the second check digit

            digt_2 = 0

            for c, i in zip(root_cpf, list(range(11, 1, -1))):

                digt_2 = int(c) * i + digt_2

            digt_2 = digt_2 * 10 % 11

            if digt_2 == 10: digt_2 = 0

            

            # concatenates first and second digits to compare to the digits informed in the input data.

            digt = str(digt_1) + str(digt_2)

            if(digt == cpf[9:11]):

                return (True, 'valid check digits')

            else:

                return (False, 'invalid check digits')

        else:

            if int(cpf) < 0:

                return (False, 'input is negatuve')

            return (False, 'input greater than 11 digits')

    else:

        return (False, "can't convert to integer")
cpfs_to_test = ['04697398947', 4697398947, 

                1720217904, '017202179045',

                4697398948, '44444444444', 

                '40469739894z', -4697398947, 

                ['04697398947', 1720217904],

               True, False, 1, 0]



for cpf in cpfs_to_test:

    print(f'is cpf\t {cpf} \tvalid? {is_cpf(cpf)}')
from math import sqrt



class Rocket():

     

    def __init__(self, x=0, y=0):

        self.x = x

        self.y = y

        

    def move_rocket(self, x_increment=0, y_increment=1):

        self.x += x_increment

        self.y += y_increment

        

    def print_rocket(self):

        print(self.x, self.y)
roc1 = Rocket(x = 5, y = 10)
roc1.print_rocket()

print(f'x = {roc1.x} \t y = {roc1.y}')
roc1.move_rocket(x_increment = 2, y_increment = 20)
roc1.print_rocket()

print(f'x = {roc1.x} \t y = {roc1.y}')
import re
class Person():

    

    def __init__ (self, name, city, phone, email):

        self.name = name

        self.city = city

        self.phone = phone

        self.email = email

    

    # this method just prints a contact card

    def print_contact_card(self):

        print(f'Name:\t {self.name}')

        print(f'City:\t {self.city}')

        print(f'Phone:\t {self.phone}')

        print(f'e-mail:\t {self.email}')

    

    # this method splits name and phone number in its components using some regular expressions.

    def get_components(self):

        self.first_name, self.last_name = self.name.split(' ')

        self.phone_contry_code = re.findall('\+\d+', self.phone)[0]

        self.ddd = re.findall('\(\d+\)', self.phone)[0]

        self.phone_number = re.findall('\d \d{4}-\d{4}', self.phone)[0]
Rodrigo = Person('Rodrigo Goncalves', 'São Paulo', '+55 (11) 9 2035-3045', 'user.name@mail.com')
Rodrigo.print_contact_card()
Rodrigo.get_components()
print(Rodrigo.first_name)

print(Rodrigo.last_name)

print(Rodrigo.phone_contry_code)

print(Rodrigo.ddd)

print(Rodrigo.phone_number)
class SmartPhone():



    def __init__ (self, size = '', interface = ''):

        self.size = size

        self.interface = interface

    

    def call(self, number):

        print(f'calling {number} ...')

    

    def send_msg(self, number, msg):

        print(f"message '{msg[0:10]}...' sent to {number}")
class MP3Player(SmartPhone):

    

    def __init__ (self, mp3_name, size, interface):

        SmartPhone.__init__(self, size, interface)

        self.mp3_name = mp3_name

        

    def play_song(self, song):

        print(f"playing '{song}'")
My_Phone = SmartPhone(interface = 'Android', size = '5 by 7')
print(f'Phone: \t {My_Phone.interface} \t Size: \t {My_Phone.size}')
My_Phone.call('+55 (11) 9 3256-8956')

My_Phone.send_msg(msg = 'hey, how r u?', number = '+55 (11) 9 5689-2356')
My_MP3 = MP3Player(mp3_name = 'google music', size = '5 by 7', interface = 'Android')
print(f'Phone: \t {My_MP3.interface} \t Size: \t {My_MP3.size} \t Name: \t {My_MP3.mp3_name}')
My_MP3.call('+55 (11) 9 5689-4578')

My_MP3.send_msg(msg = 'hey, how r u?', number = '+55 (11) 9 2356-4512')
My_MP3.play_song(song = 'My way')