def recursive_sum(n):

    if n == 1:

        return 1

    return n + recursive_sum(n - 1)



recursive_sum(10)
class Animal(object):

    def __init__(self, name, age):

        self.name = name

        self.age = age

    

    def description(self):

        print('%s is %s years old' % (self.name, self.age))

        

    def speak(self, sound):

        print('%s says %s' % (self.name, sound))



class Dog(Animal):

    def __init__(self, name, age):

        super(Dog, self).__init__(name, age)

        self._speed = 10

    

    @property

    def speed(self):

        return self._speed

    

    @speed.setter

    def speed(self, speed):

        assert isinstance(speed, (int, float))

        assert speed > 0 and speed <= 40

        self._speed = speed

    

    def description(self):

        print('%s is %s years old and runs at %s mph' % (self.name, self.age, self._speed))

    

class Fish(Animal):

    def __init__(self, name, age):

        super(Fish, self).__init__(name, age)

        

    @property

    def habitat(self):

        return self._habitat

    

    @habitat.setter

    def habitat(self, habitat):

        assert habitat in ['freshwater', 'saltwater'], 'Fish can either live in freshwater or saltwater'

        self._habitat = habitat

    

    def swim(self):

        print('%s is swimming in %s' % (self.name, self._habitat))  

        

dog = Dog('wang cai', 3)

dog.description()

dog.speed = 37

dog.description()

dog.speak('wang')



fish = Fish('Nemo', 1)

fish.habitat = 'saltwater'

fish.swim()

fish.speak('blue')
# write to file

file = open('test.txt', 'w+')

for i in range(10):

    file.write('This is line %s.\n' % i)

file.close()
# append to the end of file

file = open('test.txt', 'a')

file.write('Append this line to the end.')

file.close()
# read file

file = open('test.txt', 'r')

lines = file.readlines()

for l in lines:

    print(l)

file.close()
import json



data = {

    'name': 'Disco',

    'birthday': {

        'year': 1998,

        'month': 12,

        'day': 15

    },

    'pets': ['cat', 'dog'],

    'location': 'US'

}



response = json.dumps(data)

print(type(response))



load_data = json.loads(response)

print(load_data.keys())