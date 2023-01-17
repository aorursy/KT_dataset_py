class Employee:
    raise_salary = 1.04

    def __init__(self, first, last, payment):
        self.first = first
        self.last = last
        self.payment = payment
        self.email = first + '.' + last + '@email.com'
        
    def fullname(self):
        return '{} {}'.format(self.first, self.last)
    
    def salary_up(self):
        self.payment = int(self.payment * self.raise_salary)
        
    def __repr__(self):
        return 'Employee("{}", "{}", "{}")'.format(self.first, self.last, self.payment)

    def __str__(self):
        return '{}-{}'.format(self.fullname(), self.email)
ins = Employee('hello', 'Kitty', 5000)
print(ins)
print(1 + 2)
print(int.__add__(1, 2))
print(str.__add__('a', 'b'))
class starbucks:
    discount = 0.8
    
    def __init__(self, coffee, time, costs):
        self.coffee = coffee
        self.time = time
        self.costs = costs
        
    def cook(self):
        return '{} takes {} minutes to finish.'.format(self.coffee, self.time)
    
    def cost_down(self):
        self.costs = self.costs * self.discount
        return '{} is cheap, taking only ${}!'.format(self.coffee, self.costs)
    
    def __add__(self, other):
        return self.costs + other.costs
    
    def __len__(self):
        return len(self.cook())
store1 = starbucks('mocha', 4, 30)
store2 = starbucks('chocolate', 2, 50)
print(store1.cook())
print(store2.cost_down())
print(store1 + store2)
print(len(store1))
print(len(store2))
