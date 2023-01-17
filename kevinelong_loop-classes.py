
class Dog:
    def __init__(self, voice="woof"):
        self.voice = voice
    def speak(self):
        print(self.voice)
    
class Cat:
    def __init__(self, voice="meow"):
        self.voice = voice
    def speak(self):
        print(self.voice)
    

animals = [ Dog("woof") , Cat("meow") ]

for a in animals:
    a.speak()
    
animals = [ Dog , Cat ]

for a in animals:
    i = a()
    i.speak()
    
animals = [ Dog , Cat ]

for a in animals:
    i = a("quack")
    i.speak()