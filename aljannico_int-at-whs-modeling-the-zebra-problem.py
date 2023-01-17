## Lists of Constraint variables



color       = [ 'red', 'green', 'yellow', 'blue', 'ivory' ]

nationality = [ 'Norwegian', 'Japanese', 'Englishman', 'Spaniard', 'Ukranian' ]

pets        = [ 'zebra', 'horse', 'snails', 'dog', 'fox' ]

drinks      = [ 'water', 'orange juice', 'milk', 'coffee', 'tea' ]

cigarette   = [ 'Lucky Strike', 'Parliaments', 'Old Gold', 'Kools', 'Chesterfields']



# Variable

X = color + nationality + pets + drinks + cigarette 

# Domains

D = { x: {1,2,3,4,5} for x in X}



# Constraints:

class C:

    def __init__(self):

        self.rules = []

        

    # Methods to define constraints

    def alldiff(self,s):

        self.rules.append(('alldiff',s))

    def eq(self,l,r):

        self.rules.append(('eq',l,r))

    def right(self,l,r):

        self.rules.append(('right',l,r))        

    def next(self,l,r):

        self.rules.append(('next',l,r))

        

    def _v(self,k,a):

        if type(k) is int: return k

        return a[k]

        

    def check_rule(self,rule,a):

        op = rule[0]

        if op != 'alldiff':

            l = self._v(rule[1],a)

            r = self._v(rule[2],a)

            if op == 'eq':

                return l == r

            if op == 'next':

                return l == r + 1 or l == r - 1

            if op == 'right':

                return l == r + 1

        else:

            # Check for alldiff

            var = rule[1]

            values = set()

            for x in var:

                val = a[x]

                if val in values:

                    # print("All diff failure for val ",val)

                    return False

                values.add(val)

            return True

                                        

    def check_assignment(self,a,debug=False):

        rule_id = 1

        for rule in self.rules:

            if not self.check_rule(rule,a):

                if debug: print("Found failure in Rule ",rule_id)

                return False

            rule_id += 1

        return True

    



c = C()



# Create Constraints

c.alldiff(color) # 1

c.alldiff(nationality) # 2

c.alldiff(pets) # 3

c.alldiff(drinks) # 4

c.alldiff(cigarette) # 5

c.eq('Englishman','red') # 6

c.eq('Spaniard','dog') # 7

c.eq('coffee','green') # 8

c.eq('Ukranian','tea') # 9

c.right('green','ivory') # 10

c.eq('Old Gold','snails') # 11

c.eq('Kools','yellow') # 12

c.eq('milk',3) # 13

c.eq('Norwegian',1) # 14

c.next('Chesterfields','fox') # 15

c.next('Kools','horse') # 16

c.eq('Lucky Strike','orange juice') # 17

c.eq('Japanese','Parliaments') # 18

c.next('Norwegian','blue') # 19
from random import choice



## Create an assignment and check it

ra = { x : choice([1,2,3,4,5]) for x in X }

print(ra)

c.check_assignment(ra,debug=True)
## Now let do something more or less stupid and generate all more or less plausible assignments

## ... in groups of 5 (could be done with a generator!)



def perm(s):

    if len(s)==1: 

        return [s]

    result = []

    for v in s:

        ns = s[:]

        ns.remove(v)

        l = perm(ns)

        for el in l:

            result.append([v] + el)

    return result



len(perm([1,2,3,4,5])) # This creates the right 120 elements

# print(perm[1,2,3,4,5]) # uncomment to show the permutations
## Now, we will use it

## X is color | nationality | pets | drinks | cigarette 



def build_assignment(var,values):

    return { k:v for (k,v) in zip(var,values) }



# Test it:

a = build_assignment(X,[1,2,3,4,5]+[1,2,3,4,5]+[1,2,3,4,5]+[1,2,3,4,5]+[1,2,3,4,5])

print(a)



# Brute force!

def try_all():

    count = 0

    for cos in perm([1,2,3,4,5]):

        for nas in perm([1,2,3,4,5]):

            for pes in perm([1,2,3,4,5]):

                for drs in perm([1,2,3,4,5]):

                    for cis in perm([1,2,3,4,5]):

                        a = build_assignment(X,cos+nas+pes+drs+cis)

                        res = c.check_assignment(a)

                        if res:

                            print("Solution found! ",a)

                            return True

                        count += 1

                        if not count % 10000:

                            print(count)

    return False



## Don't do it at home, your CPU will burn ;)

# try_all() # will try 24,883,200,000 assignments AT MOST...well, well
from copy import deepcopy



## We need some more stuff in our constraint class:

## Lists of Constraint variables



color       = [ 'red', 'green', 'yellow', 'blue', 'ivory' ]

nationality = [ 'Norwegian', 'Japanese', 'Englishman', 'Spaniard', 'Ukranian' ]

pets        = [ 'zebra', 'horse', 'snails', 'dog', 'fox' ]

drinks      = [ 'water', 'orange juice', 'milk', 'coffee', 'tea' ]

cigarette   = [ 'Lucky Strike', 'Parliaments', 'Old Gold', 'Kools', 'Chesterfields']



# Variable

X = color + nationality + pets + drinks + cigarette 

# Domains

D = { x: [1,2,3,4,5] for x in X}



# Constraints:

class C:

    def __init__(self):

        self.rules = []

        

    # Methods to define constraints

    def alldiff(self,s):

        self.rules.append(('alldiff',s))

    def eq(self,l,r):

        self.rules.append(('eq',l,r))

    def right(self,l,r):

        self.rules.append(('right',l,r))        

    def next(self,l,r):

        self.rules.append(('next',l,r))

                

    def check_rule(self,rule,d):

        op = rule[0]

        if op != 'alldiff':

            l = set(d[rule[1]])

            # Left domain empty!

            if len(l)==0: return False

            if op == 'eq':

                if type(rule[2]) is int:

                    return rule[2] in l

                else:

                    l = set(l)

                    r = set(d[rule[2]])

                    return not l.isdisjoint(r)

            # Right domain empty

            r = set(d[rule[2]])

            if len(r)==0: return False

            # Weak checking

            if len(l) > 1: return True

            # For next and right, only single values are checked on the left side 

            # (more consistency problems could be found!)

            l = l.pop()

            if op == 'next':    

                return l-1 in r or l+1 in r

            if op == 'right':

                return l-1 in r

        else:

            # Check for alldiff in the simplest possible way

            # leaving out some intricate problems

            var = rule[1]

            values = set()

            for x in var:

                # Any participating domain empty?

                if len(d[x]) == 0: return False

                values = values.union(set(d[x]))

            # print("Check values against max domain for ",var,":",values)

            return values == set([1,2,3,4,5])



                                        

    def partially_consistent(self,d,debug=False):

        rule_id = 1

        for rule in self.rules:

            # print("Check rule",rule)

            if not self.check_rule(rule,d):

                if debug: print("Found failure in Rule ",rule_id)

                return False

            rule_id += 1

        return True



    def _propagate_alldiff(self,var,nd):

        changed = True

        while changed:

            changed = False

            for x in var:

                # print("Check",x,"=",nd[x],len(nd[x]),nd[x][0])

                if len(nd[x]) == 1:

                    for y in var:

                        if x != y:

                            if nd[x][0] in nd[y]:

                                print("Alldiff: Remove",nd[x][0],"from",y)

                                nd[y].remove(nd[x][0])

                                changed = True                    

    

    def propagate(self,d):

        nd = deepcopy(d)

        for rule in self.rules:

            op = rule[0]

            if op == 'alldiff':

                # Use only direct propagation for now                

                var = rule[1]

                self._propagate_alldiff(var,nd)                 

            elif op == 'eq':

                l = rule[1]

                r = rule[2]

                if type(r) is int:

                    if r in nd[l]:

                        # We have checked consistency before

                        # ... but to be on the safe side

                        if nd[l] != [r]:

                            print("Equate",l,"with fixed value",r)

                            nd[l] = [r]

                    else:

                        nd[l] = []

                        print("ERROR IN PROPAGATING EQUALITY: ",l,r)

                else:

                    if nd[l] != nd[r]:

                        print(nd[l],"<->",nd[r])

                        nd[l] = list(set(nd[l]).intersection(set(nd[r])))

                        nd[r] = nd[l][:]

                        print("Equating",l,"and",r,"to",nd[r])

            else: # next or right

                l,r = nd[rule[1]],nd[rule[2]] 

                res_l,res_r = [],[]

                if op == 'next':

                    for e in l:

                        if e+1 in r or e-1 in r:

                            res_l.append(e)

                    for e in r:

                        if e+1 in l or e-1 in l:

                            res_r.append(e)

                if op == 'right':

                    for e in l:

                        if e-1 in r:

                            res_l.append(e)

                    for e in r:

                        if e+1 in l:

                            res_r.append(e)

                if nd[rule[1]] != res_l:

                    nd[rule[1]] = res_l

                    print("Neighborhood: change",rule[1],"to",res_l)

                if nd[rule[2]] != res_r:

                    nd[rule[2]] = res_r

                    print("Neighborhood: change",rule[2],"to",res_r)

        return nd # return the new domains    

        

    # Pick the variable with the smallest domain next

    def pick_variable(self,d,a_set):

        min_value = 6

        min_var = None

        for x in X:

            if x not in a_set and len(d[x]) > 1 and len(d[x]) < min_value:

                min_var = x

                min_value = len(d[x])

        print("PICKED: ",min_var)

        return min_var



c = C()



# Create Constraints

c.alldiff(color) # 1

c.alldiff(nationality) # 2

c.alldiff(pets) # 3

c.alldiff(drinks) # 4

c.alldiff(cigarette) # 5

c.eq('Englishman','red') # 6

c.eq('Spaniard','dog') # 7

c.eq('coffee','green') # 8

c.eq('Ukranian','tea') # 9

c.right('green','ivory') # 10

c.eq('Old Gold','snails') # 11

c.eq('Kools','yellow') # 12

c.eq('milk',3) # 13

c.eq('Norwegian',1) # 14

c.next('Chesterfields','fox') # 15

c.next('Kools','horse') # 16

c.eq('Lucky Strike','orange juice') # 17

c.eq('Japanese','Parliaments') # 18

c.next('Norwegian','blue') # 19



def done(d):

    for x in X:

        if len(d[x]) != 1:

            return False

    return True



def solvable(c,d,a_set,debug=True,interactive=False):

    if debug: print("Solvable?")

    search = False

    cnt = 0

    while not search:

        # if debug: print("Done? ",cnt)

        # Are we done?

        if done(d): return True,d

        # Check for partial consistency

        cons = c.partially_consistent(d)

        if debug: print("Partially consistent? ",cons)

        if cons:  

            # Propagate!

            if debug: print("Propagate",cnt)

            nd = c.propagate(d)

            print(nd)

            if interactive: input()

            if nd == d:

                search = True

            d = nd

            cnt += 1

        else:

            # Not consistent anymore

            if debug: print("--inconsistent--/n")

            return False,None

    # we need to search and we are still partially consistent

    x = c.pick_variable(d,a_set)

    if x==None: 

        # This should have been recognized as Done already!

        print("Problem!",a_set,d)

        return False,None

    a_set.add(x)

    values = d[x]

    for v in values:

        print("Try value",v,"of Variable",x)

        dd = deepcopy(d)

        dd[x] = [v]

        res,a = solvable(c,dd,a_set)

        if res: return True,a

    # No value for x worked!

    return False,None                



def try_straight(csp,debug=True,interactive=True):

    count = 0

    # One of the following candidates must give a solution!

    d = deepcopy(D)

    a_set = set()

    res,a = solvable(csp,d,a_set,debug=debug,interactive=interactive)

    if res: 

        print("Solution found! ",a)

        for i in range(1,6):

            print("\nHaus",i,": ",end='')

            for x in X:

                if a[x] == [i]: print(x,end=' ')

        return True

    return False



    

try_straight(c,interactive=False)    
# Some magic to extend classes on the fly

# adapted from: https://medium.com/@mgarod/dynamically-add-a-method-to-a-class-in-python-c49204b85bd6

# extended to being able to use self or cls, could be done nicer, I am sure

from functools import wraps 

def add_method(cls,type='o'):

    def decorator(func):

        @wraps(func) 

        def wrapper(self, *args, **kwargs): 

            if type == 'o':

                return func(self, *args, **kwargs)

            if type == 'c':

                return func(cls, *args, **kwargs)

            if type == 'n':

                return func(*args, **kwargs)

        setattr(cls, func.__name__, wrapper)

        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func

        return func # returning func means func can still be used normally

    return decorator



# Convinience names

def add_inst_method(cls):

    return add_method(cls,'o')

def add_cls_method(cls):

    return add_method(cls,'c')

def add_static_method(cls):

    return add_method(cls,'n')

# Now, we can use a decorator add_method to add instance methods to a class

# "on the fly"



ss = '''Usage:

class AAA:

    test1 = "huhu"

    pass



@add_static_method(AAA) 

def test(test2="hihi"):

    print(AAA.test1+test2)

    

@add_inst_method(AAA) # or add_method(A,'o') (o for object)

def set_v(self,value):

    self.v = value

# setattr(A,'set_v',set_v)



@add_inst_method(AAA)

def get_v(self):

    return self.v

# setattr(A,'get_v',get_v)



a = AAA()

a.test()

a.set_v("toll")

a.get_v()

'''

print("Methods can now be added dynamically!")
## Entwickeln wir erst die Idee



### Hilfsfunktion

def subset(K,k):

    # K is a list of elements of a set

    li = len(K)

    bs = ('{0:0'+str(li)+'b}').format(k) # bitstring

    result = set()

    for i in range(li):

        if bs[i] == '1': result.add(K[i])

    return result







# K is a subset of X

# D is the (complete) set of domains of the variables in X 

# K will be left unchanged.

# D will be CHANGED as the main effect of propating (if it is changed)

# False will be returned, if nothing has been changed.

# True will be returned, if something has been changed.

def propagate_alldiff(K,D):

    # All elements of the Powerset of K will be checked

    # This INCLUDES the case for a single variable

    # We will exclude the empty set and K itself.

    # This still leaves us with an exponential number of

    # subsets (relative to |K|)!

    changed = True

    while changed:

        changed = False

        for k in range(1,2**len(K)):

            sset = subset(K,k)

            union = set()

            for s in sset:

                union = union.union(set(D[s]))

            print("Set: ",sset," Union:",union)

            if len(union) < len(sset):

                print("NOT SOLVABLE!")

                return

            if len(union) == len(sset) and len(sset) < len(K):

                # We can propagate!

                for s in set(K).difference(sset):

                    nd = set(D[s]).difference(union)

                    if nd != set(D[s]):

                        changed = True

                        print("Domain of",s,"changed to:",nd)

                        D[s] = list(nd)

        

dom = {'A':[1,2],'B':[2,3],'C':[1,2],'D':[3,4,2]}

print(dom)

propagate_alldiff(['A','B','C','D'],dom)

print(dom)





dom = {'A':[1,2],'B':[2,3],'C':[1,2],'D':[1,3,2]}

print("\nNEW EXPERIMENT: ",dom)

propagate_alldiff(['A','B','C','D'],dom)



# Hm, sieht so aus, als würde man die Probleme schon vorher finden und 

# müßte K nicht betrachten?

dom = {'A':[1,2,3],'B':[1,2,3],'C':[1,2,3],'D':[1,3,2]}

print("\nNEW EXPERIMENT: ",dom)

propagate_alldiff(['A','B','C','D'],dom)



# ok, ja, kann man weglassen
# Fresh Domains, just in case

D = { x: [1,2,3,4,5] for x in X}



@add_inst_method(C)     

def _propagate_alldiff(self,K,D):

    overall_change = False

    changed = True

    while changed:

        changed = False

        for k in range(1,2**len(K)):

            sset = subset(K,k)

            union = set()

            for s in sset:

                union = union.union(set(D[s]))

            if len(union) < len(sset):

                for s in sset:

                    D[s] = [] # unsolvable, make them all empty

                return True # we changed something

            if len(union) == len(sset) and len(sset) < len(K):

                # We can propagate!

                for s in set(K).difference(sset):

                    nd = set(D[s]).difference(union)

                    if nd != set(D[s]):

                        changed = True

                        overall_change = True

                        print("Domain of",s,"changed to:",nd)

                        D[s] = list(nd)

    return overall_change 





try_straight(c,interactive=False)    