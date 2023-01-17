class Robot():

  # Essentially a blank template since we never defined any attributes

  pass

print(Robot, Robot())
# Give it life!

my_robot = Robot()

my_robot.name = 'Wall-E'

my_robot.height = 100  # cm



your_robot = Robot()

your_robot.name = 'Rob'

your_robot.height = 200 # cm
# They live!!!!!

print(my_robot.name, my_robot.height)

print(your_robot.name, your_robot.height)
# Uh oh, we didn't give it this attribute

print(my_robot.purpose)
class Robot():

  # All robots should love humans

  purpose = 'To love humans'
# Give it life!

my_robot = Robot()

my_robot.name = 'Wall-E'

my_robot.height = 100  # cm



your_robot = Robot()

your_robot.name = 'Rob'

your_robot.height = 200 # cm
print('What is your purpose?\n')

print(my_robot.purpose)
# Rogue robot!!!

evil_robot = Robot()

evil_robot.name = 'Bender'

evil_robot.purpose = 'TO KILL ALL HUMANS!!!'



print('What is your name and your purpose?\n')

print(f'My name is {evil_robot.name} and my purpose is {evil_robot.purpose}')


my_robot = Robot()

my_robot.name = 'Wall-E'

my_robot.height = 100  # cm



your_robot = Robot()

your_robot.name = 'Rob'

your_robot.height = 200 # cm
# Who's taller?



# Tie defaults to my bot 😁

tall_bot = my_robot if my_robot.height >= your_robot.height else your_robot



# Alternative code

## if my_robot.height >= your_robot.height:

##     tall_bot = my_robot

## else:

##     tall_bot = your_robot



print(f'{tall_bot.name} is the tallest bot at {tall_bot.height} cm')
# You guys taking up my (memory) space

print('Where are you (in memory)?')

print(my_robot)

print(your_robot)
# Are you the same..?

print(f'Are you the same (using ==)? {my_robot == your_robot}') # FALSE

print(f'Are you the same (using is)? {my_robot is your_robot}') # FALSE

print(f'Are you yourself? {my_robot == my_robot}') # TRUE
generic_robot0 = Robot()

generic_robot1 = Robot()



# Are you the same..?

print(f'Are you the same (using ==)? {generic_robot0 == generic_robot1}')

print(f'Are you the same (using is)? {generic_robot0 is generic_robot1}')



print(generic_robot0)

print(generic_robot1)

# You didn't make a copy

# same_robot = generic_robot0



print(f'Are you the same (using ==)? {generic_robot0 == same_robot}') # TRUE

print(f'Are you the same (using is)? {generic_robot0 is same_robot}') # MAYBE



# def some_func(a, arr = {}):

#     arr[a] = 1

#     return arr



# a = 1

# b = 2



# print(some_func(a))
print(same_robot)

print(generic_robot0)
same_robot.name = '0001'



print(same_robot.name, generic_robot0.name)
class Robot():

    name = None

    material = 'Metal'

    is_electric = True

    num_of_arms = 2
walle = Robot()



print(f'''

name: {walle.name}

material: {walle.material}

is_electric: {walle.is_electric}

num_of_arms: {walle.num_of_arms}

''')

  
# Changing an attribute

walle.name = 'Wall-E'

# Adding a new attribute

walle.is_solar = True



print(f'''

name: {walle.name}

material: {walle.material}

is_electric: {walle.is_electric}

num_of_arms: {walle.num_of_arms}

''')

  

print(f'is_solar: {walle.is_solar}')
class Robot(object):

    

    def __init__(self, name, material, is_electric, num_of_arms, height):

        self.name = name

        self.material = material

        self.is_electric = is_electric

        self.num_of_arms = num_of_arms

        self.height = height

    

    def __repr__(self):

        return f'''

            name: {walle.name}

            material: {walle.material}

            is_electric: {walle.is_electric}

            num_of_arms: {walle.num_of_arms}

            '''



    def __gt__(self, other):

        return self.height > other.height



walle = Robot('Wall~E', 'wood', False, 2, 10)

bender = Robot('Bender', 'wood', False, 2, 30)

print(walle > bender)
class Robot():



    laws_of_robotics = [

        '1. First Law:	A robot may not injure a human being or, through inaction, allow a human being to come to harm.',

        '2. Second Law:	A robot must obey the orders given it by human beings except where such orders would conflict with the First Law.',

        '3. Third Law:	A robot must protect its own existence as long as such protection does not conflict with the First or Second Laws.'

    ]

  

    

    def print_laws(self):

        for law in self.laws_of_robotics:

            print(law)



      

    def print_n_law(n: int):

        # Check the law exists

        if n < 1 or n > 3:

            print('The #{n} law doesn\'t exist')

            return



        print(Robot.laws_of_robotics[n-1])

    
Robot.laws_of_robotics
Robot.print_laws()
Robot.print_n_law(2)
# Note what happens with Wall-e

walle = Robot()
# Has the laws built in 

walle.laws_of_robotics
# Let's have Wall-E print out those laws too! (Wait, can he do that...?)

walle.print_laws()
class Robot():

    name = None

    material = 'Metal'

    is_electric = True

    num_of_arms = 2



    # These methods belong to the Object (its self)

    def speak(self):

        print(f'I am {self.name}!')

    

    @staticmethod

    def sayHello():

        print('Hello!')



    def add_numbers(self, num0, num1):

        total = num0 + num1

        return total
walle = Robot()



print(f'''

name: {walle.name}

material: {walle.material}

is_electric: {walle.is_electric}

num_of_arms: {walle.num_of_arms}

''')



walle.speak()

walle.add_numbers(100,1)

walle.sayHello()
# Changing an attribute

walle.name = 'Wall-E'

walle.speak()
# Changing how Wall-E talks (a little more advanced)

walle.speak = lambda : print('Wwaaaalllll-eeeee!!!')

walle.speak()