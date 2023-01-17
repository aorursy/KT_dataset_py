import pandas as pd
df = pd.read_csv("../input/telugu-movies-released-in-2019/Telugu Movies 2019.csv")
tsd = dict(zip(df['Title'],df['Story']))
import random 

n = random. randint(0,26)

i = int(n)
t = df['Title'][i]
ts = tsd[t]

ts

input("Here is your question. Are you ready? (Press Enter)")

print("Here we go : \n{}".format(ts))
a = input("Enter the movie name: ")

a = str(a)

if(a == t):

    print("Congratulations! That is correct")

else:

    print("Sorry! You have lost, better luck next time")

    print("The title is: \n{}".format(t))

