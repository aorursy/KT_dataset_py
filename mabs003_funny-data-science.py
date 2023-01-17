# few libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
episode = np.array([1,2,3,4,5])
R = np.array([33318, 12033, 8509, 6532, 5697])
# for ease of comparison let's rescale the number of viewers between 0 and 100
r = (R-R.min())/(R.max()-R.min())*100

sns.pointplot(x=episode, y = r)
plt.ylabel("viewership")

S = np.array([865016, 489785, 243792, 177522, 130342])
s = (S-S.min())/(S.max()-S.min())*100
sns.pointplot(x=episode, y = s)
plt.ylabel("viewership")
G = np.array([1303448, 517419, 284472 , 262147 , 200109])
g = (G-G.min())/(G.max()-G.min())*100
sns.pointplot(x=episode, y = g)
plt.ylabel("viewership")
# Nando: https://www.youtube.com/watch?v=w2OtwL5T1ow&list=PLzVF1nAqI9VmKRcgzZX0L0diFoApovY88
N = np.array([140197 , 47640 , 56207, 28616  , 17062])
n = (N-N.min())/(N.max()-N.min())*100
sns.pointplot(x=episode, y = n)
plt.ylabel("viewership")
A = np.array([67161, 29365 , 24237, 19821, 16963])
a = (A-A.min())/(A.max()-A.min())*100
sns.pointplot(x=episode, y = a)
plt.ylabel("viewership")
U = np.array([217040, 80335, 67417, 64909, 81149 ])
u = (U-U.min())/(U.max()-U.min())*100
sns.pointplot(x=episode, y = u)
plt.ylabel("viewership")
D = np.array([85615 , 25523 , 18772 , 26155 , 19539])
d = (D-D.min())/(D.max()-D.min())*100
sns.pointplot(x=episode, y = d)
plt.ylabel("viewership")
