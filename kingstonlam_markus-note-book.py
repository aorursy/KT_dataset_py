class Person:
    
    distance = 0
    
    
    def __init__(self, name, age, height):
        self.name = name
        self.age = age
        self.height = height
        
        
    def walk(self, miles):
        self.distance += miles
        print(self.name + " walked " + str(miles) + " miles.", flush=True)
        
    def ate (self, food):
        self.distance += food
        print(self.name + " ate " + str(food) + " food.", flush=True)
         
        
        
markus = Person("Markus",7,6.7)
print(markus.height,flush=True)

markus.walk(14)
class Person:
    
    distance = 34
    
    foodAte = 42
    
    
    def __init__(self, name, age, height, favoriteFood):
        self.name = name
        self.age = age
        self.height = height
        self.favoriteFood = favoriteFood
        
    def walk(self, miles):
        self.distance += miles
        print(self.name + " walked " + str(self.distance) + " miles.", flush=True)
        
    def ate (self, food):
        self.foodAte += food
        print(self.name + " ate " + str(self.foodAte) + " food.", flush=True)
        print("{} walked {} miles.".format(self.name, str(self.distance)), flush=True) 
        
        
markus = Person("Markus" ,7 ,6.7 ,"chocolate")
print(markus.height,flush=True)

markus.walk(3)
markus.ate(3)
print(markus.favoriteFood)
class Person:
    
    distance = 34
    
    foodAte = 42
    
    def __init__(self, name, age, height, favoriteFood):
        self.name = name
        self.age = age
        self.height = height
        self.favoriteFood = favoriteFood
        
    def fly(self, miles):
        self.distance += miles
        print(self.name + " flyed " + str(self.distance) + " miles.", flush=True)
        
    def ate (self, food):
        self.foodAte += food
        print(self.name + " ate " + str(self.foodAte) + " food.", flush=True)
        
        
markus = Person("a bird" ,7 ,6.7 ,"fish")

markus.fly(3)
markus.ate(3)
print(markus.favoriteFood,flush=True)
