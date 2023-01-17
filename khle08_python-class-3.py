@property    # a decorator
def a_function(argument):
    # do something here and the content would be compacted as a function
    return None
def external():
    text = 'hello world'
    
    def intrinsic():
        print(text)
        
    return intrinsic
a = external
print(a())
def decorator_func(original_func):
    def wrapper_func(*args, **kwargs):
        print('wrapper is ahead of {}'.format(original_func.__name__))
        return original_func(*args, **kwargs)
    return wrapper_func
@decorator_func
def display():
    print('run this "desplay" function')
    
@decorator_func
def display2(name, age):
    print('2nd gen func with arguments {}, {}'.format(name, age))
    
display()
display2('kitty', 20)
def decorator_var(var1, var2):
    def decorator_func(original_func):
        def wrapper_func(*args, **kwargs):
            sums = var1 + var2
            print('wrapper is ahead of {}'.format(original_func.__name__))
            print('the sum of 2 vars is {}'.format(sums))
            return original_func(*args, **kwargs)
        return wrapper_func
    return decorator_func
@decorator_var(4, 5)
def testing(name='testing'):
    print('I am %s' % name)
    
testing()

class shop(object):
    @staticmethod
    def product1(price, tax):
        return price * tax
    
    def product2(self):
        return self.product1(self.a, self.b)
print(shop().product1 is shop().product1)
print(shop().product2 is shop().product2)
print(shop().product1 is shop.product1)
class shop(object):
    staffs = 50
    
    @classmethod
    def func2(cls):
        print('this is func2')
        print('calling {}'.format(cls.staffs))
        cls().func1()
        
    def func1(self):
        print('this is func1')
shop.func2()
shop.staffs
class starbucks:
    shop_num = 0
    extra_invest = 2000
    
    def __init__(self, owner, invest, address):
        self.o = owner
        self.i = invest
        self.address = address
        
        starbucks.shop_num += 1
        
    @property
    def email(self):
        name_list = self.o.split(' ')
        emailAcc = name_list[0] + '_' + name_list[1] + '@starbucks.com'
        return emailAcc
    
    def invest_raise(self):
        self.i = int(self.i + self.extra_invest)
print(starbucks.shop_num)
ft = starbucks('hello kitty', 10000, 'it is my home')
print(ft.email)
class starbucks:
    shop_num = 0
    extra_invest = 2000
    
    def __init__(self, owner, invest, address):
        self.o = owner
        self.i = invest
        self.address = address
        
        starbucks.shop_num += 1
        
    @property
    def email(self):
        name_list = self.o.split(' ')
        emailAcc = name_list[0] + '_' + name_list[1] + '@starbucks.com'
        return emailAcc
        
    @email.setter
    def email(self, address):
        name, tail = address.split('@')
        first, last = name.split('_')
        self.first = first
        self.last = last
       
    @email.deleter
    def email(self):
        print('Delete Email name!')
        self.first = None
        self.last = None
shop = starbucks('hello kitty', 1000, 'China')
print(shop.email)
shop.email = 'define_name@starbucks.com'
print(shop.first)
print(shop.last)
del shop.email
print(shop.first)
print(shop.last)
def time_counter(original_func):
    import time
    
    def wrapper_func(*args, **kwargs):
        t1 = time.time()
        result = original_func(*args, **kwargs)
        t2 = time.time() - t1
        print('took {:.2} sec on {} func'.format(t2, original_func.__name__))
        return result
    
    return wrapper_func
