def sumOfn(n):

    if n==0 or n==1:

        return n

    return n+sumOfn(n-1)



print(sumOfn(5))
class Animal(object):

    def __init__(self, name):

        self.name=name

        self.__color = 'red'

        

    def get_name(self):

        return self.name

    

    def get_color(self):

        return self.__color

    

    def speak(self):

        return 'LOL'

    

animal = Animal('Ruhua')

animal.speak()
class Cat(Animal):

    def __init__(self, name):

        Animal.__init__(self, name)

        

    def speak(self):

        return self.name + ': Mow'

    

cat = Cat('Wang cai')

print(cat.get_name())

print(cat.get_color())

print(cat.speak())
class BritishShortHair(Cat):

    def __init__(self, name):

        Cat.__init__(self, name)

        

    def get_hair(self):

        return self.name + ': ShortHair'

    

britishorthair = BritishShortHair('british shorthair cat')

print(britishorthair.get_name())

print(britishorthair.get_color())

print(britishorthair.speak())

print(britishorthair.get_hair())
f = open('result.csv', 'w')

f.write('Label, 10\n')

f.close()
f = open('result.csv', 'r')

print(f.read())

f.close()
import json

data = {

    'icecream': 6,

    'lolipop': 5,

    'candy': 9

}

print(type(data))

response = json.dumps(data)

print(type(response))
toText = json.loads(response)

print(type(toText))