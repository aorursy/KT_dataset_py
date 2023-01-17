### Utilities

import humanize

import simplejson

import yaml

import gzip



try: import first

except Exception as exception: print('first: ',exception)







### Multiprocessing

from pathos import multiprocessing

from joblib import Parallel, delayed





# Compilers / Optimizers

from numba import njit





### Profilers

import memory_profiler

import timeit







### Data Science

import numpy as np

import pandas as pd

import sympy



try: import modin

except Exception as exception: print('modin: ',exception)

try: import pandarallel

except Exception as exception: print('pandarallel: ',exception)



    



### Visualization

import matplotlib.pyplot as plt

import seaborn

import pydot







### NLP

import nltk

from nltk import Prover9

import ply.lex as lex

import ply.yacc as yacc

import gensim 

from bs4 import BeautifulSoup

try: import ftfy

except Exception as exception: print('ftfy: ',exception)    







### Email

try: import flanker

except Exception as exception: print('flanker: ',exception)







### Machine Learning

import fastai

import sklearn

import tensorflow

import torch

import scipy

try: import flanker

except Exception as exception: print('flanker: ',exception)

try: import pomegranate

except Exception as exception: print('pomegranate: ',exception)





### Logic Programming

import ortools



try: import z3

except Exception as exception: print('z3-solver: ',exception)

try: import pylog

except Exception as exception: print('pylog: ',exception)

try: import pyke

except Exception as exception: print('pyke: ',exception)

try: import pytholog

except Exception as exception: print('pytholog: ',exception)

try: import clyngor

except Exception as exception: print('clyngor: ',exception)

try: import clorm

except Exception as exception: print('clorm: ',exception)

try: import pyDatalog

except Exception as exception: print('pyDatalog: ',exception)

try: import quepy

except Exception as exception: print('quepy: ',exception)

try: import kanren

except Exception as exception: print('kanren: ',exception)

try: import clorm

except Exception as exception: print('clorm: ',exception)



    



### Functional Programming

from pydash import py_ as _ # ★601 - "The kitchen sink of Python utility libraries for doing "stuff" in a functional way. Based on the Lo-Dash Javascript library".

import toolz          # ★2674 - "A functional standard library for Python".

import cytoolz

import funcy          # ★2289 - "A fancy and practical functional tools".

import more_itertools # ★1329 - "More routines for operating on iterables, beyond itertools".



try: import functional; 

except Exception as exception: print('pyfunctional: ',exception)

try: import fn             # ★2999 - fn.py "Functional programming in Python: implementation of missing features to enjoy FP" (unmaintained since 2014).

except Exception as exception: print('fn: ',exception)

try: import hask           # ★634 - "Haskell language features and standard libraries in pure Python".

except Exception as exception: print('hask: ',exception)

try: import returns        # ★592 - "Make your functions return something meaningful, typed, and safe!"

except Exception as exception: print('returns: ',exception)

try: import oslash         # ★488 - "Functors, Applicatives, And Monads in Python".

except Exception as exception: print('oslash: ',exception)

try: import effect         # ★309 - "Effect isolation in Python, to facilitate more purely functional code".

except Exception as exception: print('effect: ',exception)

try: import underscore     # ★270 - "A Python port of excellent javascript library underscore.js".

except Exception as exception: print('underscore.py: ',exception)

try: import phi            # ★110 - "A library that intends to remove as much of the pain as possible from your functional programming experience in Python."

except Exception as exception: print('phi: ',exception)    

try: import pyramda        # ★106 - "Python package supporting heavy functional programming through currying. Translation of the Ramda library from javascript to python".

except Exception as exception: print('pyramda: ',exception)    

try: import pymonad        # - "a small library implementing monads and related data abstractions -- functors, applicative functors, and monoids -- for use in implementing functional style programs".

except Exception as exception: print('PyMonad: ',exception)   

try: import pymonet        # ★26 - "High abstract python library for functional programming. Contains algebraic data structures known from Haskell or Scala".

except Exception as exception: print('pyMonet: ',exception)   

try: import pfun           # ★43 - "Pure functional programming in python".

except Exception as exception: print('pfun: ',exception)   

try: import fnc            # ★57 - "Functional programming in Python with generators and other utilities".

except Exception as exception: print('fnc: ',exception)       

try: import unpythonic     # ★16 - "Supercharge your Python with parts of Lisp and Haskell."

except Exception as exception: print('unpythonic: ',exception)       



try: import deal

except Exception as exception: print('deal: ',exception)    

try: import chainable # ★142 - "Method chaining built on generators".

except Exception as exception: print('chainable: ',exception)

try: import lenses    # ★133 - "A python lens library for manipulating deeply nested immutable structures".

except Exception as exception: print('python-lenses: ',exception)        

try: import sumtypes  # "Sum Types, aka Tagged Unions, for Python".

except Exception as exception: print('sumtypes: ',exception)        

        





### Pattern matching

try: import pampy

except Exception as exception: print('pampy: ',exception)

try: import patternmatching

except Exception as exception: print('patternmatching: ',exception)







### Tranducers    

try: import transducer  #  ★45 - "This is a port of the transducer concept from Clojure to Python, with an emphasis on providing as Pythonic as interpretation of transducers as possible, rather than reproducing Clojurisms more literally".

except Exception as exception: print('Transducers: ',exception)



    



### Reactive Style    

try: import rxpy        # ★3402 -  "Reactive Extensions for Python".

except Exception as exception: print('RxPy: ',exception)

try: import broqer      # ★58 - "Library to operate with continuous streams of data in a reactive style"

except Exception as exception: print('broqer: ',exception)

    

    



# Algebraic Data Types

try: import adt

except Exception as exception: print('algebraic-data-types: ',exception)  # https://pypi.org/project/algebraic-data-types/





    

### Datatypes

import sortedcontainers

import sortedcollections



try: import orderedset

except Exception as exception: print('orderedset: ',exception)

try: import frozendict

except Exception as exception: print('frozendict: ',exception)    

    

    

### Immutable / persistent data structures

try: import pyrsistent  # ★1339 - "Persistent/Immutable/Functional data structures for Python".

except Exception as exception: print('Pyrsistent: ',exception)

try: import immutables  # ★656 - "An immutable mapping type for Python."

except Exception as exception: print('Immutables: ',exception)

try: import funktown    # ★75 - "Immutable Data Structures for Python".

except Exception as exception: print('Funktown: ',exception)

try: import amino       # ★31 - "functional data structures and type classes".

except Exception as exception: print('Amino: ',exception)

try: import pysistence  # "Pysistence is a project that seeks to make functional programming in python easier".    

except Exception as exception: print('Pysistence: ',exception)

    





### Caching

from fastcache import benchmark, clru_cache, lru_cache

import decorator

import cachetools



try: import xxhash

except Exception as exception: print('xxHash: ',exception)

try: import cached_property

except Exception as exception: print('cached_property: ',exception)

try: import kids.cache  

except Exception as exception: print('kids.cache: ',exception)

try: import cacheout

except Exception as exception: print('cacheout: ',exception)

try: import beaker

except Exception as exception: print('Beaker: ',exception)  

try: import dogpile

except Exception as exception: print('dogpile: ',exception)  # ! pip install dogpile - fails

    

    

# Cellular Automata

try: import cellular_automaton

except Exception as exception: print('cellular-automaton: ',exception)  # https://github.com/lantunes/cellpylib

try: import cellpylib as cpl

except Exception as exception: print('cellpylib: ',exception)  # https://github.com/lantunes/cellpylib    
Prover9()
### Utilities

! pip install first



### Data Science

! pip install pandarallel

! pip install modin



### Email

! pip install flanker

! pip install ftfy



### Caching

! pip install xxHash

! pip install cached_property

! pip install kids.cache

! pip install cacheout

! pip install Beaker



### Machine Learning

! pip install pomegranate
### Caching

! pip install dogpile    # ERROR: Command errored out with exit status 1: python setup.py egg_info Check the logs for full command output.
### Logic Programming

! pip install z3-solver

! pip install pylog

! pip install pytholog

! pip install clyngor

! pip install kanren

! pip install pydatalog

! pip install quepy

! pip install pyketools
### Logic Programming

! yes | conda install -c potassco clingo

! yes | conda install -c potassco clorm
### Functional Programming

! pip install pyfunctional  

! pip install pampy         

! pip install deal   

! pip install hask3

! pip install returns

! pip install oslash

! pip install effect

! pip install underscore.py

! pip install phi

! pip install pyramda

! pip install PyMonad

! pip install pyMonet

! pip install pfun

! pip install fn

! pip install fnc

! pip install unpythonic

! pip install chainable

! pip install lenses

! pip install sumtypes

! pip install patternmatching



### Tranducers    

! pip install transducer



### Reactive Style

! pip install rxpy

! pip install broqer



### Algebraic Data Types

! pip install algebraic-data-types



### DataTypes

! pip install orderedset

! pip install frozendict



### Immutable / persistent data structures

! pip install immutables

! pip install funktown

! pip install amino

! pip install pysistence
# Cellular Automata

! pip install cellular-automaton  # https://gitlab.com/DamKoVosh/cellular_automaton

! pip install cellpylib           # https://github.com/lantunes/cellpylib    
### Utilities

import humanize

import simplejson

import yaml



try: import first

except Exception as exception: print('first: ',exception)







### Multiprocessing

from pathos import multiprocessing





# Compilers / Optimizers

from numba import jit





### Profilers

import memory_profiler

import timeit







### Data Science

import numpy as np

import pandas as pd

import sympy



try: import modin

except Exception as exception: print('modin: ',exception)

try: import pandarallel

except Exception as exception: print('pandarallel: ',exception)



    



### Visualization

import matplotlib.pyplot as plt

import seaborn

import pydot







### NLP

import nltk

from nltk import Prover9

import ply.lex as lex

import ply.yacc as yacc

import gensim 

from bs4 import BeautifulSoup

try: import ftfy

except Exception as exception: print('ftfy: ',exception)    







### Email

try: import flanker

except Exception as exception: print('flanker: ',exception)







### Machine Learning

import fastai

import sklearn

import tensorflow

import torch

import scipy

try: import flanker

except Exception as exception: print('flanker: ',exception)

try: import pomegranate

except Exception as exception: print('pomegranate: ',exception)





### Logic Programming

import ortools



try: import z3

except Exception as exception: print('z3-solver: ',exception)

try: import pylog

except Exception as exception: print('pylog: ',exception)

try: import pyke

except Exception as exception: print('pyke: ',exception)

try: import pytholog

except Exception as exception: print('pytholog: ',exception)

try: import clyngor

except Exception as exception: print('clyngor: ',exception)

try: import clorm

except Exception as exception: print('clorm: ',exception)

try: import pyDatalog

except Exception as exception: print('pyDatalog: ',exception)

try: import quepy

except Exception as exception: print('quepy: ',exception)

try: import kanren

except Exception as exception: print('kanren: ',exception)

try: import clorm

except Exception as exception: print('clorm: ',exception)



    



### Functional Programming

from pydash import py_ as _ # ★601 - "The kitchen sink of Python utility libraries for doing "stuff" in a functional way. Based on the Lo-Dash Javascript library".

import toolz          # ★2674 - "A functional standard library for Python".

import cytoolz

import funcy          # ★2289 - "A fancy and practical functional tools".

import more_itertools # ★1329 - "More routines for operating on iterables, beyond itertools".



try: import functional; 

except Exception as exception: print('pyfunctional: ',exception)

try: import fn             # ★2999 - fn.py "Functional programming in Python: implementation of missing features to enjoy FP" (unmaintained since 2014).

except Exception as exception: print('fn: ',exception)

try: import hask           # ★634 - "Haskell language features and standard libraries in pure Python".

except Exception as exception: print('hask: ',exception)

try: import returns        # ★592 - "Make your functions return something meaningful, typed, and safe!"

except Exception as exception: print('returns: ',exception)

try: import oslash         # ★488 - "Functors, Applicatives, And Monads in Python".

except Exception as exception: print('oslash: ',exception)

try: import effect         # ★309 - "Effect isolation in Python, to facilitate more purely functional code".

except Exception as exception: print('effect: ',exception)

try: import underscore     # ★270 - "A Python port of excellent javascript library underscore.js".

except Exception as exception: print('underscore.py: ',exception)

try: import phi            # ★110 - "A library that intends to remove as much of the pain as possible from your functional programming experience in Python."

except Exception as exception: print('phi: ',exception)    

try: import pyramda        # ★106 - "Python package supporting heavy functional programming through currying. Translation of the Ramda library from javascript to python".

except Exception as exception: print('pyramda: ',exception)    

try: import pymonad        # - "a small library implementing monads and related data abstractions -- functors, applicative functors, and monoids -- for use in implementing functional style programs".

except Exception as exception: print('PyMonad: ',exception)   

try: import pymonet        # ★26 - "High abstract python library for functional programming. Contains algebraic data structures known from Haskell or Scala".

except Exception as exception: print('pyMonet: ',exception)   

try: import pfun           # ★43 - "Pure functional programming in python".

except Exception as exception: print('pfun: ',exception)   

try: import fnc            # ★57 - "Functional programming in Python with generators and other utilities".

except Exception as exception: print('fnc: ',exception)       

try: import unpythonic     # ★16 - "Supercharge your Python with parts of Lisp and Haskell."

except Exception as exception: print('unpythonic: ',exception)       



try: import deal

except Exception as exception: print('deal: ',exception)    

try: import chainable # ★142 - "Method chaining built on generators".

except Exception as exception: print('chainable: ',exception)

try: import lenses    # ★133 - "A python lens library for manipulating deeply nested immutable structures".

except Exception as exception: print('python-lenses: ',exception)        

try: import sumtypes  # "Sum Types, aka Tagged Unions, for Python".

except Exception as exception: print('sumtypes: ',exception)        

        





### Pattern matching

try: import pampy

except Exception as exception: print('pampy: ',exception)

try: import patternmatching

except Exception as exception: print('patternmatching: ',exception)







### Tranducers    

try: import transducer  #  ★45 - "This is a port of the transducer concept from Clojure to Python, with an emphasis on providing as Pythonic as interpretation of transducers as possible, rather than reproducing Clojurisms more literally".

except Exception as exception: print('Transducers: ',exception)



    



### Reactive Style    

try: import rxpy        # ★3402 -  "Reactive Extensions for Python".

except Exception as exception: print('RxPy: ',exception)

try: import broqer      # ★58 - "Library to operate with continuous streams of data in a reactive style"

except Exception as exception: print('broqer: ',exception)

    

    



# Algebraic Data Types

try: import adt

except Exception as exception: print('algebraic-data-types: ',exception)  # https://pypi.org/project/algebraic-data-types/





    

### Datatypes

import sortedcontainers

import sortedcollections



try: import orderedset

except Exception as exception: print('orderedset: ',exception)

try: import frozendict

except Exception as exception: print('frozendict: ',exception)    

    

    

### Immutable / persistent data structures

try: import pyrsistent  # ★1339 - "Persistent/Immutable/Functional data structures for Python".

except Exception as exception: print('Pyrsistent: ',exception)

try: import immutables  # ★656 - "An immutable mapping type for Python."

except Exception as exception: print('Immutables: ',exception)

try: import funktown    # ★75 - "Immutable Data Structures for Python".

except Exception as exception: print('Funktown: ',exception)

try: import amino       # ★31 - "functional data structures and type classes".

except Exception as exception: print('Amino: ',exception)

try: import pysistence  # "Pysistence is a project that seeks to make functional programming in python easier".    

except Exception as exception: print('Pysistence: ',exception)

    





### Caching

from fastcache import benchmark, clru_cache, lru_cache

import decorator

import cachetools



try: import xxhash

except Exception as exception: print('xxHash: ',exception)

try: import cached_property

except Exception as exception: print('cached_property: ',exception)

try: import kids.cache  

except Exception as exception: print('kids.cache: ',exception)

try: import cacheout

except Exception as exception: print('cacheout: ',exception)

try: import beaker

except Exception as exception: print('Beaker: ',exception)  

try: import dogpile

except Exception as exception: print('dogpile: ',exception)  # ! pip install dogpile - fails

    

    

# Cellular Automata

try: import cellular_automaton

except Exception as exception: print('cellular-automaton: ',exception)  # https://github.com/lantunes/cellpylib

try: import cellpylib as cpl

except Exception as exception: print('cellpylib: ',exception)  # https://github.com/lantunes/cellpylib    
# Source: https://github.com/Z3Prover/z3/blob/master/examples/python/socrates.py

from z3 import *



Object = DeclareSort('Object')



Human = Function('Human', Object, BoolSort())

Mortal = Function('Mortal', Object, BoolSort())



# a well known philosopher 

socrates = Const('socrates', Object)



# free variables used in forall must be declared Const in python

x = Const('x', Object)



axioms = [ForAll([x], Implies(Human(x), Mortal(x))), 

          Human(socrates)]





s = Solver()

s.add(axioms)



print(axioms, s.check()) # prints sat so axioms are coherent



# classical refutation

s.add(Not(Mortal(socrates)))

print(s, s.check()) # prints unsat so socrates is Mortal