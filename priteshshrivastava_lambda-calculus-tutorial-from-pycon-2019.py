## Switch

def LEFT(a):

    def f(b):

        return a

    return f



def RIGHT(a):

    def f(b):

        return b

    return f
LEFT("5v")("ground")  ## 2 arguments are curried
RIGHT("5v")("ground")
## Boolean variables

def TRUE(x):

    '''

    Return the 1st argument

    '''

    return lambda y: x



def FALSE(x):

    '''

    Return the second argument

    '''

    return lambda y: y
TRUE(1)(2) 
FALSE(0)(1)
## NOT gate

def NOT(x):

    return x(FALSE)(TRUE)
NOT(TRUE)
NOT(FALSE)
## AND & OR gates

def AND(x):

    '''

    If 1st argument is true, return 2nd argument

    If 1st argument is false, return false

    x is also a single argument function

    Curry 2 args using lambda of y

    '''

    return lambda y : x(y)(x)

    

    

def OR(x):

    '''

    If 1st argument is true, return true

    If 1st argument is false, return 2nd argument

    '''

    return lambda y : x(x)(y)
AND(TRUE)(FALSE)
AND(TRUE)(TRUE)
OR(FALSE)(TRUE)
OR(FALSE)(FALSE)
## Numbers with functions

## Church Numerals

ONE = lambda f : lambda x : f(x)

TWO = lambda f : lambda x : f(f(x))

THREE = lambda f : lambda x : f(f(f(x)))
ONE
TWO
## How to use Church numerals as functions ?

## We'll cheat for a bit by using `+` and the number 1. But this is only for illustration purposes. 

def incr(x): 

    return x + 1

incr(0)
ONE(incr)(0)
TWO(incr)(0)
THREE(incr)(0)
## Exponentiation

THREE(TWO)(incr)(0)

## Start from outermost THREE as THREE has closure

## incr & TWO don't have closure yet to be evaluated !!

## TWO(TWO(TWO(incr)))(0)

## Expanding outermost TWO as it has closure

## TWO(TWO(incr))(TWO(TWO(incr))(0))

## Now expanding inner TWO, 3rd from left as it has closure, ie 2 args curried

## TWO(TWO(incr))(TWO(incr)(TWO(incr)(0))

## Rightmost TWO has closure now so evaluating that

## TWO(TWO(incr))(TWO(incr)(2))

## Again, 3rd or rightmost TWO has closure

## TWO(TWO(incr))(4)

## Leftmost TWO has closure now

## TWO(incr)(TWO(incr)(4))

## Rightmost TWO has closure

## TWO(incr)(6)

## Final TWO has closure now

## 8
## How do you implement ZERO ?

ZERO = lambda f : lambda x : x



ZERO(incr)(0)