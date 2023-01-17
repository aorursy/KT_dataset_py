# Depth-First Serach (Left to Right) Algo used

# C3 Linearization Algo used



class A:

    def sam(self):

        print('A')

        

class B(A):

    def sam(self):

        print('B')

        

class C(B):

    pass



a, b, c = A(), B(), C()



C().sam()



# Method Resolution Order

print(C.mro())

print(C.__mro__)
from collections import OrderedDict



o_dict = OrderedDict()

o_dict['name'] = 'pankaj'

o_dict['l_name'] = 'goyal'

o_dict['nationality'] = 'indian'

o_dict['age'] = '30'

type(o_dict)
strng = 'This may sound like an easy job if you have a collection based on a list. You just loop over all of the elements. But how do you sequentially traverse elements of a complex data structure, such as a tree? For example, one day you might be just fine with depth-first traversal of a tree. Yet the next day you might require breadth-first traversal. And the next week, you might need something else, like random access to the tree elements'

l = strng.split()



from collections import Counter



# Word Count / Element Count / Character Count (for string)

cnt = Counter(l)



# Top 2 words

cnt.most_common()[:2]
# Create Iterators



favorite_numbers = [2,4,6,8]



itrtr = iter(favorite_numbers)     # Creates Iterator over the List which itself is iterable

print(type(itrtr))

print(next(itrtr))
# Create Generators



favorite_numbers = [2,4,6,8]



#----------------------------------------------------

# Method 1

#----------------------------------------------------

squares = (n**2 for n in favorite_numbers)

print(type(squares))

print(next(squares))





#----------------------------------------------------

# Method 2

#----------------------------------------------------

def gen():

    for n in favorite_numbers:

        yield n**2

        

g = gen()

print(type(g))

print(next(g))
a = 1

b = 1







print(a is b)

print(a == b)



print(id(a))

print(id(b))
name = """w'o"w"""

str(name)
repr(name)
class Example(object): 

  

    # Initializing 

    def __init__(self): 

        self.value = '' 

  

    # deletes an attribute 

    def __delete__(self, instance): 

        print ("Inside __delete__") 

          

    # Destructor 

    def __del__(self): 

        print("Inside __del__") 

      

      

class Foo(object): 

    exp = Example() 

  

# Driver's code 

f = Foo() 

del f.exp
import sys



try:

    #pass

    sys.exit(0)

    raise Exception

    

except:

    sys.exit(0)

    print('Exception Raised')

    

else:

    print('No Exception')



finally:

    print('Finally Block')

def gen(text):

    try:

        for line in text:

            try:

                yield int(line)

            except:

                # Ignore blank lines - but catch too much!

                pass

    finally:

        print('Doing important cleanup')



text = ['1', '', '2', '', '3']



if any(n > 1 for n in gen(text)):

    print('Found a number')



print('Oops, no cleanup.')