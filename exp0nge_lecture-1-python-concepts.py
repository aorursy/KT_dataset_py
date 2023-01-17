this_is_a_string = "Hello, world!"



this_is_a_number = 1



this_is_a_boolean = True
print(this_is_a_number)
2 + 2
"a" + "b"
2 + "a"
2 * "a"
2.1 * "a"
1 / 2
1 // 2
1 % 2
long_number_string = "1.2324234324932483290489023843294732984623046324932649"

print(long_number_string)

print(type(long_number_string))

print(len(long_number_string))

print(len(123))
float(long_number_string)
from decimal import Decimal

Decimal(long_number_string)
zero = 0

one = 1

cat = True

hello = "world"

hello2 = ""
print(bool(hello) is True)

if hello2:

    print("🐱")

else:

    print("🥺")
if 1 > 2:

    print("1 > 2")

elif 2 > 3:

    print("2 > 3")

elif 2 == 2:

    print("2 == 2")

else:

    # do this if none is True

    pass
numbers = {1: "one", 2: "two", 1: "three"}

print(numbers)
numbers[1] = "Bob"

numbers
animals = {"dog": "bark", "cat": "meow"}

print(animals)

dynamic_animals = {"dog": "meow" }

print(dynamic_animals)

animals_2 = animals.copy()

print(animals_2)

animals_2["dog"] = "meow"

print(animals_2)

print(animals)

print(animals["dog"])
dynamic_animals = {"dog": { "small": "chihuahua" } }





dynamic_animals["dog"]["small"] # chihuahua
# animal -> [small, large]

dynamic_animals = {"dog": ["chihuahua", "great dane"] }





dynamic_animals["dog"][0]
a_list = list(range(10))

a_list
b = [0, 1, 2, 3, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 5, 6, 7, 8, 9]

print(b[4][5])



c = [0, 1, 2, 3, {"dog": ["chihuahua", "great dane"] }, 5, 6, 7, 8, 9]

print(c[4]["dog"])
[[i for i in range(10)] for j in range(20)]
numbers = [1, 2, 3, 3, 2]

numbers.append(5 * "a")

numbers2 = [8, 9, 10]



numbers +  numbers2
numbers = [1, 2, 3, 3, 2]

numbers2 = [2, 3]



set(numbers)
set(numbers2)
set(numbers) - set(numbers2)
numbers = [1, 2, 3, 3, 5]

if len(numbers) > 5:

    print(numbers)

else:

    print("Less than 5")
numbers = [1, 2, 3, 3, 5]

if numbers:

    print(numbers)

else:

    print("Else!")

print(bool(numbers))
for i in [1,2,3]:

    print(i)
for i in [1,2,3]:

    if i > 2:

#         break

        continue

    print(i)
for idx, number in enumerate([5,6,7], start=9):

    print(idx, number)
# tuple

colors = ('red', 'blue', 1, 22)

print(colors)
i = 0

while i < 11:

    print("i", i)

    i += 1

j = 6

while j < 11 and j > 5:

    print("j", j)

    j += 1

# while 1 == 1:

#     print("1 == 1")
def my_first_function():

    print('Hello world!')



print('type: {}'.format(my_first_function))



my_first_function()  # Calling a function

my_first_function
print('%s' % 'blue')

print('%i' % 1)
def greet_us(name1, name2):

    print('Hello {} and {}!'.format(name1, name2))



greet_us('John Doe', 'Superman')



def greet_us_2(*args):

    print("Hello " + ", ".join(args))

    

greet_us_2("Ron", "Conrad")
def create_person_info(name, age, job=None, salary=300):

    info = {'name': name, 'age': age, 'salary': salary, 'job': job}

    print(info)

    

create_person_info("Rassu", 90, salary=1)
def even_numbers(number):

    x = number % 2

    if x == 0:

        print("Even")

    else:

        print("Odd")





even_numbers(2)

even_numbers(5)


def odd_number(number):

    if number % 2 == 0:

        return False

    else:

        return True

    

def print_odd_numbers(number_list):

    for number in number_list:

        is_odd = odd_number(number)

        if is_odd:

            print(number)

            



print_odd_numbers([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print_odd_numbers(range(100))
def compute_patterns(inputs=[]):

    inputs.append('some stuff')

    patterns = ['a list based on'] + inputs

    return patterns
compute_patterns()
compute_patterns()
def compute_patterns_2(inputs=[]):

    if not 'some_stuff' in inputs:

        inputs.append('some stuff')

    patterns = ['a list based on'] + inputs

    return patterns



def compute_patterns_3(inputs=None):

    my_list = []

    patterns = ['a list based on'] + inputs

    my_list.append(patterns)

    return my_list



def compute_patterns_4(inputs=None):

    if inputs is None:

        inputs = []

        inputs.append("some stuff")

    else:

        inputs.append("it's a list")

    patterns = ['a list based on'] + inputs

    return patterns
compute_patterns_2()
compute_patterns_3([1, 2, 3])
compute_patterns_4(["other stuff"])
class Dog(object):

    def __init__(self, name, weight):

        self.weight = weight

        self.name = name

        

    def speak(self):

        print("Bark! %s" % self.name)



class Person(object):

    def __init__(self, name, gender, height):

        self.name = name

        self.gender = gender

        self.height = height
a_person = Person("George", "male", "5'11")

a_person.gender
# An instance of Dog

# self is _the_ instance

great_dane = Dog("Great Dane", 100)

golden_retriever = Dog("Golden Retriever", 80)

print(great_dane.name)

print(great_dane.weight)

print(golden_retriever.name)

print(great_dane.speak())
!pip install requests
# import requests

# from requests import get