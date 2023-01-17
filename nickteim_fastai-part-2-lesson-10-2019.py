%load_ext autoreload

%autoreload 2dne dejdeosnd djd sj



%matplotlib inline
import torch

import matplotlib.pyplot as plt
import ipywidgets as widgets #is to make buttoms and stuff
def f(o): print('hi')
w = widgets.Button(description='Click me')
w
w.on_click(f)
from time import sleep
def slow_calculation():

    res = 0

    for i in range(5):

        res += i*i

        sleep(1) #wait a secound 

    return res
slow_calculation() #so it will take 5 seound to mak the calulation 
def slow_calculation(cb=None): #cb=None callback we accept that it can take a parameter like funciton and let call it cb 

    res = 0

    for i in range(5):

        res += i*i

        sleep(1) #wait a sec.

        if cb: cb(i) #if there is a callback (cb) call it and pass in i #it could be the epoch number 

    return res
def show_progress(epoch):

    print(f"Awesome! We've finished epoch {epoch}!")
#so we take one function and pass it into another

slow_calculation(show_progress) #it count from 0 to 4 since the res+=i*i dosent influence i only range does that in slow_calculation

#do also note that it takes 5 sec. 
slow_calculation() #lambda o: print(f"Awesome! We've finished epoch {o}!")
def show_progress(exclamation, epoch):

    print(f"{exclamation}! We've finished epoch {epoch}!")
#but since we above wrote a function we only used ones, we can rewrite it and use lambda

slow_calculation(lambda o: show_progress("OK I guess", o)) # #lambda notation is just another way of whriting a function but we only uses it one

#so insted of 'def' we say 'lambda' and insted of parentheses we write the argument (o) and then what we want it to do. 

#note here that 'show_progress("OK I guess"' is the exclamation and 'o 'is epoch in the show_progress function above. 
#so lets say we want to make a function just takes exclamation we can do the below and we dont want to write the lambda in slow_calculation



def make_show_progress(exclamation):

    _inner = lambda epoch: print(f"{exclamation}! We've finished epoch {epoch}!") 

    return _inner
slow_calculation(make_show_progress("Nice!"))


def make_show_progress(exclamation):

    # Leading "_" is generally understood to be "private"

    def _inner(epoch): print(f"{exclamation}! We've finished epoch {epoch}!")

    return _inner
slow_calculation(make_show_progress("Nice!"))
f2 = make_show_progress("Terrific")
slow_calculation(f2)
slow_calculation(make_show_progress("Amazing"))
from functools import partial
slow_calculation(partial(show_progress, "OK I guess")) #so here we just pass in the function show_progress

#as the one parameter and then use "OK I guess" as an argument for the next parameter  

#so this now return a new function that just takes one parameter where the secound parameter is always given 
f3 = partial(show_progress, "OK I guess")
# f3() #now this function just takes one parameter which is epoch. since show_progress took to parameters one for epoch and the 

#secound was exclamation. but now exclamation is alwas "OK I guess". so the f2 only take the one parameter (epoch)
f3(1)
class ProgressShowingCallback():

    def __init__(self, exclamation="Awesome"): self.exclamation = exclamation #same a last lecture but we just store 

        #the exclamation value in a function

    def __call__(self, epoch): print(f"{self.exclamation}! We've finished epoch {epoch}!") #__call__ taking a objeckt(class-->ProgressShowingCallback) 

        #and treat it as if it was a function 
cb = ProgressShowingCallback("Just super")
#so we can call ProgressShowingCallback as if it was a function with paratenthess

# cb('hi')
slow_calculation(cb)
def f(*args, **kwargs): print(f"args: {args}; kwargs: {kwargs}")
f(3, 'a', thing1="hello")
f(3, 'a','b',9, thing1="hello", nine=9) #do remember as it is right here, the position of which you pass the argument er importen
def slow_calculation(cb=None):

    res = 0

    for i in range(5):

        if cb: cb.before_calc(i) #i we use it in PrintStatusCallback but not in PrintStepCallback

        res += i*i

        sleep(1)

        if cb: cb.after_calc(i, val=res) #i, val=res we use them in PrintStatusCallback but not in PrintStepCallback

    return res
class PrintStepCallback():

    def __init__(self): pass

    def before_calc(self, *args, **kwargs): print(f"About to start") #even though we dont use the arguments given from slow_calculation

        # we still have to make place for them and we do that with *args, **kwargs. 

    def after_calc (self, *args, **kwargs): print(f"Done step")
slow_calculation(PrintStepCallback())
class PrintStatusCallback():

    def __init__(self): pass

    def before_calc(self, epoch, **kwargs): print(f"About to start: {epoch}") #here we add **kwargs to make sure the code doesnt break if another argument are add to the function in the fucture

    def after_calc (self, epoch, val, **kwargs): print(f"After {epoch}: {val}") #but note we still use the arguments given from the function (slow_calculation)
slow_calculation(PrintStatusCallback())
#early stopping 

def slow_calculation(cb=None):

    res = 0

    for i in range(5):

        if cb and hasattr(cb,'before_calc'): cb.before_calc(i) #check if 'before_calc' exsist and call it if it is

        res += i*i

        sleep(1)

        if cb and hasattr(cb,'after_calc'): #chack if there is callback(cb) called 'after_calc' and only call it is it is 

            if cb.after_calc(i, res): #check the return value 

                print("stopping early") #and do something based on the returned value 

                break

    return res
class PrintAfterCallback():

    def after_calc (self, epoch, val):

        print(f"After {epoch}: {val}")

        if val>10: return True #cancel if our loop if the 'val' is greater then 10  

slow_calculation(PrintAfterCallback()) #and here we see that it stops at 14, since 14 is greater then 10 
#change the calulation 

class SlowCalculator():

    def __init__(self, cb=None): self.cb,self.res = cb,0 #defines what we need 

    

    #to use calc(function in this class) in ModifyingCallback we have to make the below function 

    def callback(self, cb_name, *args): #here u can also use __call__ and in the calc function u can just use self insted of self.callback

        if not self.cb: return #check to see if the given callback is defined 

        cb = getattr(self.cb,cb_name, None) #if it is it will grap it ...

        if cb: return cb(self, *args) #... and pass it into the calulator object itself (self) 

    

    #so we take our calulation function and putting it into a class so now the value it is calulation(res) is a attribute of the class 

    def calc(self):

        for i in range(5):

            self.callback('before_calc', i)

            self.res += i*i

            sleep(1)

            if self.callback('after_calc', i):

                print("stopping early")

                break
class ModifyingCallback():

    def after_calc (self, calc, epoch): #note the calculator (calc) functions calls on this funtions 

        print(f"After {epoch}: {calc.res}")

        if calc.res>10: return True #so we can now go into the calulator function and stop if the result gets greater then 10

        if calc.res<3: calc.res = calc.res*2 #and we can double the result by multipying with 2 if it is less then 3 
calculator = SlowCalculator(ModifyingCallback()) #for the changes from ModifyingCallback to be valid we pass it into the SlowCalculator class
calculator.calc() #have to call it like it is a class

calculator.res
# __dunder__ thingles
#exsampel

class SloppyAdder():

    def __init__(self,o): self.o=o #construck o 

    def __add__(self,b): return SloppyAdder(self.o + b.o + 0.01) #add o with b + 0.01 since it is sloppy

    def __repr__(self): return str(self.o)#printing 
a = SloppyAdder(1)

b = SloppyAdder(2)

a+b
t = torch.tensor([1.,2.,4.,18])
m = t.mean(); m
(t-m).mean()
(t-m).pow(2).mean() #taking the power of 2 to the difference for a veribel and the mean(m) to the mean of it all 
(t-m).abs().mean() #taking the absolut value of the  difference for a veribel and the mean(m) to the mean of it all 
(t-m).pow(2).mean().sqrt() #we have to take the "kvadratrod" to get to the right scale again
(t-m).pow(2).mean(), (t*t).mean() - (m*m)
t #we use same data from variance 
#exsampel

# `u` is twice `t`, plus a bit of randomness

u = t*2 #multiply t with 2 and set it = to u

u *= torch.randn_like(t)/10+0.95 #and put in some random noise 



plt.scatter(t, u); #plot
prod = (t-t.mean())*(u-u.mean()); prod #so now let compare the difference from 't' and its mean. With the difference in u and its mean and let multiply them together
prod.mean() #and let take the mean of that 
#so now lets take some random in a new verible and call it 'v'

v = torch.randn_like(t)

plt.scatter(t, v); #plot 't' and 'v'
((t-t.mean())*(v-v.mean())).mean() #and let us calulate the same product as before and take the mean of it 

#we see that this new number is much smaller then before (tensor(105.0522))
cov = (t*v).mean() - t.mean()*v.mean(); cov
cov / (t.std() * v.std())
def log_softmax(x): return x - x.exp().sum(-1,keepdim=True).log()
import numpy

#n : int or array_like of ints

#p : float or array_like of floats



def binomial(n,p):numpy.random.binomial(n, p, size=None)

    