a = 6
a
a == 7
a == 6
i = 101
i > 100
i = 99
i > 100
"AC/DC" == "Micheal Jackson"
"AC/DC" != "Micheal Jackson"
age=17
if age>=18:
    print("Enter")
print("Move")
age=17
if age>=18:
    print("Enter")
else:
    print("Meat Loaf")
print("Move")
age=18
if age>18:
    print("Enter")
elif age==18:
    print("Pink Floyd")
else:
    print("Meat Loaf")
print("Move")
not (True)
not (False)
A = False
B = True
A or B
A = False
B = False
A or B
album_year = 1990
if (album_year<1980) or (album_year>1989):
    print("This 70's or 90's")
else:
    print("This 80's")
A = False
B = True
A and B
A = True
B = True
A and B
album_year = 1983
if (album_year>1979) and (album_year<1990):
    print("This 80's")
range(3)
list(range(3))
list(range(10,15))
squares_indexes = range(5)
list(squares_indexes)
squares = ['red','yellow','green','purple','blue']
squares
print(f'Before squares {squares}')

for i in range(5):
    
    print(f'Before square {i} is {squares[i]}')
    
    squares[i]="white"
    
    print(f'After square {i} is {squares[i]}')

    print(f'After squares {squares}')    
squares = ['red','yellow','green']
squares
for square in squares:
    print(square)
for i,square in enumerate(squares):
    print(f'index {i},square {square}')
squares = ['orange','orange','purple','blue']
squares
newsquares =[]
newsquares
i=0
i
while squares[i]=='orange':
    newsquares.append(squares[i])
    i+=1
newsquares
def function(a):
    """add 1 to a"""
    b = a + 1
    print(f'a + 1 = {b}')
    return b
function(3)
def f1(input):
    """add 1 to input"""
    output=input+1
    return output
def f2(input):
    """add 2 to input"""
    output=input+2
    return output
f1(1)
f2(f1(1))
f2(f2(f1(1)))
f1(f2(f2(f1(1))))
album_ratings = [10.0,8.5,9.5]
album_ratings
Length=len(album_ratings)
Length
Sum=sum(album_ratings)
Sum
print(f'Before album_ratings {album_ratings}')
sorted_album_ratings=sorted(album_ratings)
print(f'sorted_album_ratings {sorted_album_ratings}')
print(f'After album_ratings {album_ratings}')
print(f'Before album_ratings {album_ratings}')
album_ratings.sort()
print(f'After album_ratings {album_ratings}')
def add1(a):
    """
    add 1 to a
    """
    b=a+1
    return b
help(add1)
add1(5)
c=add1(10)
c
def Mult(a,b):
    c=a*b
    return c
Mult(2,3)
Mult(2,'Micheal Jackson ')
def MJ():
    print('Micheal Jackson')
MJ()
def NoWork():
    pass
NoWork()
print(NoWork())
def NoWork():
    pass
    return None
NoWork()
print(NoWork())
def add1(a):
    b=a+1
    print(f'{a} plus 1 equals {b}')
    return b
add1(2)
def printStuff(Stuff):
    for i,s in enumerate(Stuff):
        print(f'Album {i} Rating is {s}')
album_ratings
printStuff(album_ratings)
def ArtistNames(*names):
    for name in names:
        print(f'Name {name}')
ArtistNames("Micheal Jackson","AC/DC","Pink Floyd")
ArtistNames("Micheal Jackson","AC/DC")
def AddDC(y):
    x =y+"DC"
    print(f'Local x {x}')
    return x
x="AC"
print(f'Global x {x}')
z=AddDC(x)
print(f'Global z {x}')
def Thriller():
    Date=1982
    return Date
Thriller()
# Date
# NameError: name 'Date' is not defined
Date = 2017
print(Thriller())
print(Date)
def ACDC(y):
    print(f'Rating {Rating}')
    return Rating+y
Rating=9
Rating
z=ACDC(1)
print(f'z {z}')
print(f'Rating {Rating}')
def PinkFloyd():
    global ClaimedSales
    ClaimedSales = '45 million'
    return ClaimedSales
PinkFloyd()
print(f'ClaimedSales {ClaimedSales}')
def type_of_album(artist, album, year_released):
    
    print(artist, album, year_released)
    if year_released > 1980:
        return "Modern"
    else:
        return "Oldie"
    
x = type_of_album("Michael Jackson", "Thriller", 1980)
print(x)
type([1,34,3])
type(1)
type("yellow")
type({"dog":1,"cat":2})
Ratings = [10,9,6,5]
Ratings
Ratings.sort()
Ratings
Ratings.reverse()
Ratings
# Class Circle
# Data Attributes radius,color

class Circle(object):
    pass
# Object 1: instance of type Circle

# Data Attributes 
# radius=4
# color='red'
# Object 2: instance of type Circle

# Data Attributes 
# radius=2
# color='green'
# Class Rectangle
# Data Attributes width,height,color

class Rectangle(object):
    pass
# Object 1: instance of type Rectangle

# Data Attributes 
# widrh=2
# height=2
# color='blue'
# Object 2: instance of type Rectangle

# Data Attributes 
# widrh=3
# height=1
# color='yellow'
class Circle(object):
    def __init__(self,radius,color):
        self.radius = radius
        self.color = color
class Rectangle(object):
    def __init__(self,height,width,color):
        self.height = height
        self.width = width
        self.color = color
RedCircle = Circle(10,"red")
print(f'RedCircle radius {RedCircle.radius}')
print(f'RedCircle color {RedCircle.color}')
C1 = Circle(10,"blue")
print(f'C1 radius {C1.radius}')
print(f'C1 color {C1.color}')
C1.color = 'yellow'
print(f'C1 radius {C1.radius}')
print(f'C1 color {C1.color}')
C1.radius = 25
C1.color = 'green'

print(f'C1 radius {C1.radius}')
print(f'C1 color {C1.color}')
# Method add_radius to change Circle size
class Circle(object):
    def __init__(self,radius,color):
        self.radius = radius
        self.color = color
        
    def add_radius(self,r):
        self.radius = self.radius + r
        return self.radius
    
    def change_color(self,c):
        self.color = c
        return self.color
    
    def draw_circle():
        pass
C1=Circle(2,'red')
print(f'C1 radius {C1.radius}')
print(f'C1 color {C1.color}')
C1.add_radius(8)
print(f'C1 radius {C1.radius}')
print(f'C1 color {C1.color}')
C1.change_color('blue')
print(f'C1 radius {C1.radius}')
print(f'C1 color {C1.color}')
dir(Circle)
import matplotlib.pyplot as plt
%matplotlib inline  
class Circle(object):
    def __init__(self,radius,color):
        self.radius = radius
        self.color = color
        
    def add_radius(self,r):
        self.radius = self.radius + r
        return self.radius
    
    def change_color(self,c):
        self.color = c
        return self.color
    
    def draw_circle(self):
        plt.gca().add_patch(plt.Circle((0, 0), radius=self.radius, fc=self.color))
        plt.axis('scaled')
        plt.show() 
RedCircle = Circle(1,'red')

print(f'RedCircle radius {RedCircle.radius}')
print(f'RedCircle color {RedCircle.color}')
RedCircle.draw_circle()
# Create a new Rectangle class for creating a rectangle object

class Rectangle(object):
    
    # Constructor
    def __init__(self, width=2, height=3, color='r'):
        self.height = height 
        self.width = width
        self.color = color
    
    # Method
    def draw_rectangle(self):
        plt.gca().add_patch(plt.Rectangle((0, 0), self.width, self.height ,fc=self.color))
        plt.axis('scaled')
        plt.show()
SkinnyBlueRectangle = Rectangle(2, 10, 'blue')
print(f'SkinnyBlueRectangle height {SkinnyBlueRectangle.height}')
print(f'SkinnyBlueRectangle width {SkinnyBlueRectangle.width}')
print(f'SkinnyBlueRectangle color {SkinnyBlueRectangle.color}')
SkinnyBlueRectangle.draw_rectangle()