

def welcome_to_blog():

  # print welcome message

  print("Welcome to my blog.")

  

welcome_to_blog()



# output

# Welcome to my blog.
def welcome_to_blog(name):

  # print welcome message

  message = "Hello " + name.title() + ", Welcome to blog."

  print(message)

  

welcome_to_blog('Durgesh')

def welcome_to_blog(name):

  # print welcome message

  message = "Hello " + name.title() + ", Welcome to blog."

  print(message)

  

welcome_to_blog('Durgesh')

welcome_to_blog('John')

welcome_to_blog('Jessica')


class users():

  def __init__(self, name):

    self.name = name

  

  def greetings(self):

    print("Hello ", self.name.title())


class users():

  def __init__(self, name):

    self.name = name

  

  def greetings(self):

    print("Hello ", self.name.title())

    

user = users('Durgesh')

user.greetings()



# output

# Hello, Durgesh


class users():

  def __init__(self, name):

    self.name = name

  

  def greetings(self):

    print("Hello ", self.name.title())

    

user1 = users('James')

user1.greetings()



# output

# Hello, James



user1 = users('Jessica')

user1.greetings()



# output

# Hello, Jessica
class users():

  def __init__(self, name):

    self.name = name

  

  def greetings(self):

    print("Hello ", self.name.title())



class admin(users):

  def __init__(self, name):

    super().__init__(name)

  

  def describe_admin(self):

    print(self.name.title() + " is an admin.")

  

admin1 = admin('James')

admin1.greetings()

admin1.describe_admin()



# output

# Hello, James

# James is admin.



user1 = users('Jessica')

user1.greetings()



# output

# Hello, Jessica