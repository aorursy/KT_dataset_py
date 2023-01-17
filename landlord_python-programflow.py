1 + 1
2 * 4.4
1 + 1

2 * 4.4
print(1 + 1)

print(2 * 4.4)
# this is a comment and it will not be executed

# print 1 + 1

print(f"1 + 1 is {1+1}")

# print 2 * 4.4

print(f"2 * 4.4 is {2*4.4}")

# print 'completed'

print('completed')
first = 1 + 1

final = first * 4.4



print(final)
my_name = 'Vishwas'

my_age = 22

my_favourite_number = 34

has_pet = False



print(f"My name is {my_name}")

print(f"My age is {my_age}")

print(f"My favourite number is {my_favourite_number}")

print(f"Do I have a Pet: {has_pet}")
print(type(my_name))

print(type(my_age))

print(type(my_favourite_number))

print(type(has_pet))
def cube(number):

    return number**3
# we can store the function output in variables

cube_6 = cube(6)



print(cube_6)



my_input = 3.4

# we can also use variables as function input

print(cube(my_input))
print(cube('apple'))
assert cube(3) == 23
1**2

2**2

3**2

4**2

5**2

6**2

7**2

8**2

9**2
print(1**2)

print(2**2)

print(3**2)

print(4**2)

print(5**2)

print(6**2)

print(7**2)

print(8**2)

print(9**2)
def my_func(x):

    print(x**2 + 3)
my_func(1)

my_func(2)

my_func(3)

my_func(4)

my_func(5)

my_func(6)

my_func(7)

my_func(8)

my_func(9)
def ex_1():

    print('ex_1 is running')

    print('returning value "a"')

    return 'a'



ex_1()
def ex_2():

    print('ex_2 is running')

    print('exiting without returing a value')

    

ex_2()
def test_high_score(player_score, high_score):

    if player_score > high_score:

        print('High score!')

        high_score = player_score

        

    return high_score
print(test_high_score(83, 98))
print(test_high_score(95, 93))
def nested_example(x):

    if x < 50:

        if x % 2 == 0:

            return 'branch_a'

        else:

            return 'branch_b'

    else:

        return 'branch_c'

    

print(nested_example(42))

print(nested_example(51))

print(nested_example(37))
print(50 > 10)

print(2 + 2 == 4)

print(-3 > 2)
print(True and True)

print(True and False)

print(False and True)

print(False and False)
print(True or True)

print(True or False)

print(False or True)

print(False or False)
x = 5

y = 3



print(x > 4 and y > 2)

print(x > 7 and y > 2)

print(x > 7 or y > 2)
print(not True)

print(not False)
x = 10

y = 8



print(x > 7 or y < 7)

print(not x > 7 or y < 7)

print(not x > 7 or not y < 7)

print(not (x > 7 or y < 7))
x = 0

while x < 5:

    print(x)

    x = x + 1
def first_n_fibonacci(n):

    prev_num = 0

    curr_num = 1

    count = 2

    

    print(prev_num)

    print(curr_num)

    

    while count <= n:

        next_num = curr_num + prev_num

        print(next_num)

        prev_num = curr_num

        curr_num = next_num

        count += 1



def below_x_fibonacci(x):

    prev_num = 0

    curr_num = 1

    

    if curr_num < x:

        print(prev_num)

        print(curr_num)

    elif prev_num < x:

        print(prev_num)

        

    while curr_num + prev_num < x:

        next_num = curr_num + prev_num

        print(next_num)

        prev_num = curr_num

        curr_num = next_num
m = 7

print(f'First {m} Fibonacci numbers')

first_n_fibonacci(m)
y = 40

print(f'Fibonacci numbers below {y}')

below_x_fibonacci(y)
bread_recipe = ['Dissolve salt in water', 'Mix yeast into water', 'Mix water with flour to form dough', 

                'Knead dough', 'Let dough rise', 'Shape dough', 'Bake']
soup_recipe = ['Dissolve salt in water', 'Boil  water', 'Add bones to boiling water', 'Chop onions', 

               'Chop garlic', 'Chop carrot', 'Chop celery', 'Remove bones from water', 

               'Add vegetables to boiling water', 'Add meat to boiling water']



beans_recipe = ['Soak beans in water', 'Dissolve salt in water', 'Heat water and beans to boil', 

                'Drain beans when done cooking']
def print_recipe(instructions):

    count = 1

    for step in instructions:

        print(f'{count}. {step}')

        count += 1
print_recipe(soup_recipe)
print_recipe(bread_recipe)
print_recipe(beans_recipe)
def first_n_fibonacci_while(n):

    prev_num = 0

    curr_num = 1

    count = 2

    

    print(prev_num)

    print(curr_num)

    

    while count <= n:

        next_num = curr_num + prev_num

        print(next_num)

        prev_num = curr_num

        curr_num = next_num

        count += 1



def first_n_fibonacci_for(n):

    prev_num = 0

    curr_num = 1

        

    print(prev_num)

    print(curr_num)

    

    for count in range(2, n+1):

        next_num = curr_num + prev_num

        print(next_num)

        prev_num = curr_num

        curr_num = next_num
first_n_fibonacci_while(7)
first_n_fibonacci_for(7)
def fibonacci_recursive(n):

    if n == 0:

        return 0

    elif n == 1:

        return 1

    else:

        return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)
fibonacci_recursive(4)
def is_prime(number):

    if number <= 1:

        return False

    

    for factor in range(2, number):

        if number % factor == 0:

            return False

    

    return True



def print_prime(n):

    for number in range(1, n):

        if is_prime(number):

            print(f'{number} is prime')
print_prime(10)
list_of_numbers = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

prime_list = []

for number in list_of_numbers:

    if is_prime(number):

        prime_list.append(number)

prime_list
[number for number in list_of_numbers if is_prime(number)]
def print_this(a):

    print(f'inside print_this: {a}')



a = 5

print_this(2)

print(f'a = {a}')
def print_it():

    print(f'inside print_it: {a}')

    

a = 5

print_it()

print('a = ', a)
def some_exponent(exponent):

    def func(x):

        return x**exponent

    return func
some_exponent(2)(2), some_exponent(3)(2)
def print_todo(watch_tv, read, eat, sleep):

    print('I need to:')

    if watch_tv:

        print('  watch_tv')

    if read:

        print('  read')

    if eat:

        print('  eat')

    if sleep:

        print('  sleep')

print_todo(True, True, True, True)
def print_todo_default(watch_tv, read, eat=True, sleep=True):

    print('I need to:')

    if watch_tv:

        print('  watch_tv')

    if read:

        print('  read')

    if eat:

        print('  eat')

    if sleep:

        print('  sleep')

print_todo_default(True, True)
def print_todo_args(*args):

    print('I need to:')

    for arg in args:

        print(f'{arg}')

        

print_todo_args('watch_tv', 'read', 'eat', 'sleep')