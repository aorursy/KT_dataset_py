import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/Thesavagecoder7784/images/master/netflix_titles.csv")
df.shape
tsd = dict(zip(df['title'],df['description']))
import random 
n = random. randint(0,6234)
i = int(n)
t = df['title'][i]
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
