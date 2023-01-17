def times2(x) :

    return x*2
times2
times2(30)
def add1To(f) :

    def _inner(x) :

        return 1 + f(x)

    return _inner
add1To(times2)
add1To(times2)(30)
def add1To(f) :

    return lambda x : 1 + f(x)
add1To(times2)(30)
def addValueTo(f, val=1) :

    return lambda x : val + f(x)
addValueTo(times2,5)(30)
def concatWithBlank(*args) :

    return ' '.join(str(arg) for arg in args)



concatWithBlank('a','b')
concatWithBlank('a','b','c','d','e')
x = ['c','d']

concatWithBlank(*x)
concatWithBlank('a','b',*range(5),'d')
def f(x=100, **kwargs) :

    print('x',x,'kwargs',kwargs)

    

f(y=99, z=22, x=33)
f(y=99, z=22)
k = {'a':22, 'b':33, 'x':44}

f(**k)
from functools import partial



def f(a,b) :

    return concatWithBlank(a,b)



partial(f,'xy')('z')
def myPartial(f, *fixedArgs, **fixedKwargs) :

    def _inner(*args, **kwargs) :

        allArgs = fixedArgs + args

        allKwargs = dict(fixedKwargs)

        allKwargs.update(kwargs)

        return f(*allArgs, **allKwargs)

    return _inner



myPartial(f,'xy')('z'), partial(f,'xy')('z')
def decoratorFunc(f) :

    return 99



@decoratorFunc

def times2(x) :

    return x*2



times2
def logArgs(f) :

    def _inner(*args,**kwargs) :

        print(f.__name__,'called with',args,kwargs)

        return f(*args,**kwargs)

    return _inner



@logArgs

def times2(x) :

    return x*2



times2(30)
def annealer(f):

    def _inner(start, end):

        return partial(f, start, end)

    return _inner



@annealer

def sched_lin(start, end, pos):

    return start + pos*(end-start)

@annealer

def sched_cos(start, end, pos):

    return start + (1 + math.cos(math.pi*(1-pos))) * (end-start) / 2



sched_lin(90.0,100.0)(0.3)
class a() :

    def __call__(self,m) :

        print('I was called with argument',m)



aObj = a()

aObj('xyz')
class a() :

    def __init__(self) :

        self.x = 99

    def __getattr__(self,k) :

        return str(k) + '???'



aObj = a()

aObj.x  # aObj has this attribute so it will be returned
aObj.y  # aObj doesn't have this attribute so its __getattr__ will be called
x = 'xyz'

print(x,f'{x!s}',f'{x!r}')
class hugeFakeArray() :

    def __init__(self,size) :

        self.members = {}

        self.size = size

    def __getitem__(self,n) :

        return self.members.get(n,n)

    def __setitem__(self,n,v) :

        self.members[n] = v

    def __len__(self) :

        return self.size

        

x = hugeFakeArray(200000000000000)

x[1000000000]
x[1000000000]=20; x[1000000000]
len(x)