def print_hello():

    print('Hello')





def print_greetings(name):

    print('Ola %s' % (name))

    



def print_numbers(n1, n2):

    print('Número 1: %d, Número 2: %d' % (n1, n2))
def square_with_ret(n):

    result = n * n

    print('Square value from %d is %d' % (n, result))

    return result



def square_without_ret(n):

    result = n * n

    print('Square value from %d is %d' % (n, result))
print_hello()

print_greetings('Matheus')

number1 = 3

other_number = 5

print_numbers(number1, other_number)

square_return_1 = square_with_ret(4)

square_return_2 = square_without_ret(4)

print(f'result 1: {square_return_1}, result 2: {square_return_2}')
def nothing():

    pass



nothing_result = nothing()

print(f'Nothing result: {nothing_result}')
def square(n, m, o):

    print(f'Primeiro argumento (n): {n}')

    print(f'Segundo argumento (m): {m}')

    print(f'Terceiro argumento (o): {o}')

    return [n * n, m * m, o * o]



square(3, 2, 55)
def square(n, **kwargs):

    print(f'Primeiro argumento (n): {n}')

    print(f'Argumento keywrod (kwargs): {kwargs}')

    

    print(f'Vamos calcular o quadrado de {kwargs["name"]}')

    return n * n



square(5, name="cinco") # O valor passado em name é uma string por escolha do desenvolvedor e não por ser um argumento keyword. O argumento keyword, não restrige os valores a uma string
def square(n, is_upper_case=False, **kwargs):

    print(f'Primeiro argumento (n): {n}')

    print(f'Argumento keywrod (kwargs): {kwargs}')

    print(f'Argumento default (is_upper_case): {is_upper_case}')

    

    quote = f'Vamos calcular o quadrado de {kwargs["name"]}'

    if is_upper_case:

        quote = quote.upper()

    print(quote)

    return n * n



square(5, name="cinco")

square(5, name='cinco', is_upper_case=True)
def square(n, *num_list, is_upper_case=False, **kwargs):

    print(f'Primeiro argumento (n): {n}')

    print(f'Argumento keywrod (kwargs): {kwargs}')

    print(f'Argumento default (is_upper_case): {is_upper_case}')

    print(f'Argumento arbitrário (num_list): {num_list}')

    

    quote = f'Vamos calcular o quadrado de: {kwargs["name"]}'

    if is_upper_case:

        quote = quote.upper()

    print(quote)

    result = [n * n]

    for x in num_list:

        result.append(x * x)

    return result



res1 = square(5, name="cinco")

res2 = square(5, 6, 7, name=["cinco", "seis", "sete"])

res1, res2
global_var = 2



def func1():

    local_var = 3

    

func1()
global_var = 2

var = 4

var2 = 6



def func1():

    print(f'-----start func1-----')

    local_var = 3

    var = 1

    global var2

    var2 = 32

    print(f'global_var: {global_var}')

    print(f'local_var: {local_var}')

    print(f'var: {var}')

    print(f'var2: {var2}')

    print(f'-----end func1-----')



func1()

print(f'global_var: {global_var}')

print(f'var: {var}')

print(f'var2: {var2}')
def do_math_op(n1, n2, op):

    result = op(n1, n2)

    print(f'Result of op({n1}, {n2}): {result}')

    

do_math_op(2, 4, lambda x, y: x + y)

do_math_op(2, 4, lambda x, y: x ** y)
# 1ª forma

def map_on_list(a, func):

    res = []

    for a_i in a:

        res.append(func(a_i))

    return res





def pow_(x):

    return x ** expoent





def run():

    target = [1, 2, 3]

    expoent = 2

    new_target = map_on_list(target, pow_)

    print(target)

    print(new_target)



    

run()
# 2ª forma

def map_on_list(a, func):

    res = []

    for a_i in a:

        res.append(func(a_i))

    return res





def run():

    target = [1, 2, 3]

    expoent = 2

    new_target = map_on_list(target, lambda x: x ** expoent)

    print(target)

    print(new_target)



    

run()
class Guest:

    pass
class VipGuest:

    pass
g = Guest()

g
class Guest:

    

    def __init__(self):

        print('Chamando o Construtor')



g = Guest()

g
class Guest:

    

    def __init__(self, name, age=1):

        print(f'Convidado {name} ({age})')

    

    def hello(self):

        print(f'Hello everyone')

        

g = Guest('Matheus', age=24)

g2 = Guest('Marcos')

g.hello()

print(g)

g, g2
class Guest:

    

    def __init__(self, name, age=1):

        self.name = name

        print(f'Convidado {name} ({age})')

        

    def __str__(self):

        return f'Convidado {self.name}'

    

    def hello(self):

        print(f'Hello everyone')

        

g = Guest('Matheus', age=24)

g2 = Guest('Marcos')

g.hello()

print(g)

g, g2
class Guest:

    

    def __init__(self, name, age=1):

        self.name = name

        self.age = age

        print(f'Convidado {name} ({age})')

        

    def __str__(self):

        return f'Convidado {self.name}'

    

    def __repr__(self):

        return f'Convidado: nome={self.name}, idade={self.age}'

    

    def hello(self):

        print(f'Hello everyone')

        

g = Guest('Matheus', age=24)

g2 = Guest('Marcos')

g.hello()

print(g)

g, repr(g2)
class Guest:

    

    def __init__(self, name, age=1):

        self.name = name

        self.age = age

        print(f'Convidado {name} ({age})')

        

    def __str__(self):

        return f'Convidado {self.name}'

    

    def __repr__(self):

        return f'Convidado: nome={self.name}, idade={self.age}'

    

    def __eq__(self, other):

        return self.age == other.age

    

    def __lt__(self, other):

        return self.age < other.age

    

    def hello(self):

        print(f'Hello everyone')

        

g = Guest('Matheus', age=24)

g2 = Guest('Marcos')

g.hello()

print(g)

print(repr(g2))

print(g == g2)

print(g != g2)

print(g < g2)

print(g > g2)
class Guest:

    

    def __init__(self, name, age=1):

        self.name = name

        self.age = age

        self.invite_card = None  # Estamos criando um atributo sem valor

        print(f'Convidado {name} ({age})')

        

    def __str__(self):

        return f'Convidado {self.name}'

    

    def __repr__(self):

        return f'Convidado: nome={self.name}, idade={self.age}, convite={self.invite_card}'

    

    def __eq__(self, other):

        return self.age == other.age

    

    def __lt__(self, other):

        return self.age < other.age

    

    def hello(self):

        print(f'Hello everyone')

        

g = Guest('Matheus', age=24)

g2 = Guest('Marcos')

g.hello()

print(g)

print(repr(g2))

print(g == g2)

print(g != g2)

print(g < g2)

print(g > g2)

print(g2.age)
class Guest:

    

    def __init__(self, name, age=1):

        self.name = name

        self.age = age

        self.invite_card = None  # Estamos criando um atributo sem valor

        print(f'Convidado {name} ({age})')

        

    def __str__(self):

        return f'Convidado {self.name}'

    

    def __repr__(self):

        return f'Convidado: nome={self.name}, idade={self.age}, convite={self.invite_card}'

    

    def __eq__(self, other):

        return self.age == other.age

    

    def __lt__(self, other):

        return self.age < other.age

    

    def hello(self):

        print(f'Hello everyone')

        

    @staticmethod

    def minimum_age():

        print(f'A idade mínima é de 13 anos')

        

g = Guest('Matheus', age=24)

g2 = Guest('Marcos')

g.hello()

print(g)

print(repr(g2))

print(g == g2)

print(g != g2)

print(g < g2)

print(g > g2)

print(g2.age)

Guest.minimum_age()
class Guest:

    

    total = 0

    

    def __init__(self, name, age=1):

        Guest.total += 1

        self.name = name

        self.age = age

        self.invite_card = None  # Estamos criando um atributo sem valor

        print(f'Convidado {name} ({age})')

        

    def __str__(self):

        return f'Convidado {self.name}'

    

    def __repr__(self):

        return f'Convidado: nome={self.name}, idade={self.age}, convite={self.invite_card}'

    

    def __eq__(self, other):

        return self.age == other.age

    

    def __lt__(self, other):

        return self.age < other.age

    

    def hello(self):

        print(f'Hello everyone')

        

    @staticmethod

    def minimum_age():

        print(f'A idade mínima é de 13 anos')

        

    @staticmethod

    def total_guest_created():

        print(f'O total de convidados é {Guest.total}')

        

g = Guest('Matheus', age=24)

g2 = Guest('Marcos')

g.hello()

print(g)

print(repr(g2))

print(g == g2)

print(g != g2)

print(g < g2)

print(g > g2)

print(g2.age)

Guest.minimum_age()

print(Guest.total)

Guest.total_guest_created()
# Bibliotecas de data e tempo

import datetime

import time



# Principais subclasses do datetime



# Date Class

# Classe para gerenciar datas em Python.

d = datetime.date.today()

d.year

d.month

d.day



# Timedate Class

# Classe que combina as data e hora.

dt = datetime.datetime.today()

print(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)



# Timedelta Class

# Classe que gerencia diferença de date e tempo

d = datetime.date.today()

print(d.year, d.month, d.day)

d = d + datetime.timedelta(days=7)

print(d.year, d.month, d.day)



## Time Class

print(time.time()) # Unix Epoch Time