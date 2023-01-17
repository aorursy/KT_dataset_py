# Lets look at the example discussed 

# in the description of Advance counting

# tilt.So how many ways are their to 

# select 5 students from class of 15 in order

import math



n = 15 # Total Number of students

k = 5 # selection_of_students

print("Combination of 5 students ", (math.factorial(n))/(math.factorial(n-k)))



# which is equal to (15 X 14 X 13 X 12 x 11)

# Now we have to remove repetation 

# e.g (Sohaib, Anwaar, Awais, Uzma, Afifa)

# are same as (Afifa, Uzma, Anwaar, Sohaib, Awais)

# So we have to remove it.

# Now how many ways are their to arrange these 5 

# student i.e (5! = 5 X 4 X 3 X 2 X 1)



print("Combination to Arrange Students ",math.factorial(k))

# For this 1 possible combination their are 120 

# ways to change place of every student.

# Now their are other combination to so we have

# to do it for every combination. i.e simple multiply 

# 5! to the denominator 

# So number of combination of chosing 5 students from class of 15 is



print("Combination of chosing 5 students from class of 15 is ",math.factorial(n)/(math.factorial(n-k) * math.factorial(k)))



# Now thats going to be amazing

# formula for this is  (n! /n - k! * k! )  
# Python 3 program to print terms of binomial 

# series and also calculate sum of series. 

  

# Function to print the series 

def Binomial_theorm(A, X, n): 

    sum_ = []

    # Calculating and printing first term 

    term = pow(A, n) 

    sum_.append(term)

  

    # Computing and printing remaining terms 

    for i in range(1, n+1):  

  

        # Find current term using previous terms 

        # We increment power of X by 1, decrement 

        # power of A by 1 and compute nCi using  

        # previous term by multiplying previous 

        # term with (n - i + 1)/i 

        term = int(term * X * (n - i + 1)/(i * A)) 

        sum_.append(term)

    return sum(sum_) 

      

# Driver Code 

A = 3; B = 3; n = 2

vals = []

A_B = []



for i in range(-22,22):

    if i != 0:

        A_B.append(f"({i} ,{i})^{n}")

        vals.append(Binomial_theorm(i, i, n))

    else:

        A_B.append(f"({i} ,{i})^{n}")

        vals.append(0)

    



import plotly.express as px



df = px.data.gapminder().query("country=='Canada'")

fig = px.line( x=A_B, y=vals, title='(a + b)^2')

fig.show()

 

n = 3

vals = []

A_B = []



for i in range(-22,22):

    if i != 0:

        A_B.append(f"({i} ,{i})^{n}")

        vals.append(Binomial_theorm(i, i, n))

    else:

        A_B.append(f"({i} ,{i})^{n}")

        vals.append(0)

    



import plotly.express as px



df = px.data.gapminder().query("country=='Canada'")

fig = px.line(x=A_B, y=vals, title='(a + b)^3')

fig.show()

 

n = 4

vals = []

A_B = []



for i in range(-22,22):

    if i != 0:

        A_B.append(f"({i} ,{i})^{n}")

        vals.append(Binomial_theorm(i, i, n))

    else:

        A_B.append(f"({i} ,{i})^{n}")

        vals.append(0)

    



import plotly.express as px



df = px.data.gapminder().query("country=='Canada'")

fig = px.line(x=A_B, y=vals, title='(a + b)^4')

fig.show()

 

n = 5

vals = []

A_B = []



for i in range(-22,22):

    if i != 0:

        A_B.append(f"({i} ,{i})^{n}")

        vals.append(Binomial_theorm(i, i, n))

    else:

        A_B.append(f"({i} ,{i})^{n}")

        vals.append(0)

    



import plotly.express as px



fig = px.line( x=A_B, y=vals, title='(a + b)^5')

fig.show()

 

n = 6

vals = []

A_B = []



for i in range(-22,22):

    if i != 0:

        A_B.append(f"({i} ,{i})^{n}")

        vals.append(Binomial_theorm(i, i, n))

    else:

        A_B.append(f"({i} ,{i})^{n}")

        vals.append(0)

    



import plotly.express as px



fig = px.line( x=A_B, y=vals, title='(a + b)^6')

fig.show()

# lets add 5 to A and see the curves

n = 2

vals = []

A_B = []



for i in range(-22,22):

    if i != -5:

        A_B.append(f"({i} ,{i})^{n}")

        vals.append(Binomial_theorm(i+5, i, n))

    else:

        A_B.append(f"({i} ,{i})^{n}")

        vals.append(0)

    



import plotly.express as px



fig = px.line( x=A_B, y=vals, title='(a + b)^7')

fig.show()
