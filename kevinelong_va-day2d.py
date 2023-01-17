class Animal: #NOUN
    def __init__(self, name, word = "squeak"): # CONSTRUCTOR
        #PROPERTIES/ATTRIBUTES - AJECTIVES
        self.name = name
        self.word = word

    def show_kind(self): #METHOD - VERB
        print(self.name)
   
    def speak(self): #METHOD - VERB
        print(self.word)
        
p = Animal("penguin")
p.show_kind()
p.speak()

d = Animal("duck", "quack")
d.show_kind()
d.speak()

p.speak()
print(p.name)
class Dog(Animal):
    
    def __init__(self, breed = "mutt"):
        super().__init__("dog", "woof")
        self.breed = breed
        
    def fetch():
        pass
    
d = Dog()
        
d.speak()
class Pug(Dog):
    
    def __init__(self):
        super().__init__(breed="pug")
        self.word = "whine"
d = Pug()
        
d.speak()
class Lab(Dog):
    
    def __init__(self):
        super().__init__(breed="lab")
        self.word = "beg"
d = Lab()
        
d.speak()
animals = [ Pug(), Dog(), Animal("penguin")]

for a in animals:
    a.speak()