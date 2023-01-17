def pstate(x,y,stored,verbose=False):

    if x==0 and y==0:p=1

    elif x<0 or y<0:p=0

    elif (x,y) in stored.keys():p=stored[(x,y)]

    else:

        p=((365-(x-1+y))/365)*pstate(x-1,y,stored)[0]+((x+1)/365)*pstate(x+1,y-1,stored)[0]

        if verbose:print(f"terms a {((365-(x-1+y)))} b {pstate(x-1,y)} c {((x+1))} d {pstate(x+1,y-1)}")

        stored[(x,y)]=p

    return(p,stored)

stored={}
print(pstate(2,0,stored)[0])

print(364/365)

print(pstate(0,1,stored)[0])

print(1/365)

print(pstate(3,0,stored)[0])

print((364/365)*(363/365))

print(pstate(1,1,stored)[0])

print((1/365)*(364/365)+(364/365)*(2/365))

stored
print(pstate(23,0,stored)[0])

print(pstate(22,0,stored)[0])

stored
def triplicate_at_N(n,stored):

    x=n-1

    y=(n-1)-x

    p_trip=0

    while x>=0:

        p_trip+=(y/365)*pstate(x,y,stored)[0]

        #print(f"({x},{y})  P {pstate(x,y)}")

        x-=2

        y+=1

    return(p_trip,stored)
print(triplicate_at_N(3,stored)[0])

print((1/365)*(1/365))

print(triplicate_at_N(4,stored)[0])

print(pstate(1,1,stored)[0]*(1/365))

print(triplicate_at_N(5,stored)[0])

print(pstate(2,1,stored)[0]*(1/365)+pstate(0,2,stored)[0]*(2/365))

stored
def triplicate_by_N(n,stored):

    p=0

    x=3

    while x<=n:

        p+=triplicate_at_N(x,stored)[0]

        x+=1

    return(p,stored)
print(triplicate_by_N(3,stored)[0])

print(triplicate_at_N(3,stored)[0])

print(triplicate_by_N(4,stored)[0])

print(triplicate_at_N(3,stored)[0]+triplicate_at_N(4,stored)[0])

stored
def find_50():

    p=0

    x=3

    while p<.5:

        p+=triplicate_at_N(x)

        print(f"{x}  {p}")

        x+=1

    print(f"The {x-1}th person brings the cumulative probability up to {p}.")

    return(p)
#find_50()
def find_50_fast(stored):

    p=0

    x=3

    while p<.5:

        p+=triplicate_at_N(x,stored)[0]

        print(f"{x}  {p}")

        x+=1

    print(f"The {x-1}th person brings the cumulative probability up to {p}.")

    return(p,stored)
find_50_fast(stored)
import random, collections



def generate_birthdays(n):

    daylist=[]

    for x in range(n):

        daylist.append(random.random()*365.25//1)

    c=collections.Counter(daylist)

    print(c)

    if 3 in c:return True

    else:return False



        
generate_birthdays(88)

print(triplicate_by_N(100,stored)[0])