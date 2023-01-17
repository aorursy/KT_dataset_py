class calculator:
    def __init__(self, pi=3.14):    # special method 特殊方法
        self.pi = pi    # attribute 属性
    
    def add(self, x, y):    # method 方法
        return x + y
    
    def minus(self, x, y):    # method 方法
        return x - y
        
    def multiply(self, x, y):    # method 方法
        return x * y
        
    def divide(self, x, y):    # method 方法
        return x / y
    
    def area(self, r):
        return self.multiply(self.pi, r * r)
# instance 实例
machine1 = calculator()
print(machine1.add(20, 30))
print(machine1.pi)
machine2 = calculator(pi=5.14)
print(machine2.pi)
machine1.multiply(3, 5)

class starbucks:
    pass
shenzhen = starbucks()
shenzhen.address = 'U know it'
shenzhen.owner = 'ME'
shenzhen.invest = 10000
print(shenzhen.invest)
import time

start_time = time.clock()

class Greeting(object):
    def __init__(self, greeting='hello'):
        self.greeting = greeting
        
    def greet(self, name):
        return '%s! %s' % (self.greeting, name)
    
greeting = Greeting('hola')
print(greeting.greet('bob'))
end_time = time.clock()
time_consume = end_time - start_time
print(time_consume)
start_time = time.clock()

def greet(greeting, target):
    return '%s! %s' % (greeting, target)

print(greet('hola', 'bob'))
print(time.clock() - start_time)
class computer(calculator):
    pass
cmp = computer(pi=4.14)
print(cmp.area(5))
print(cmp.pi)
