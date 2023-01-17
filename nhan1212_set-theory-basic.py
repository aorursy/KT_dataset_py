set_A = {1, 2, 3, "hello", "cut"}

type(set_A)
set_B = set([1, 2, 3, "hello", "cut"])

type(set_B)
import numpy as np

set_C = set(np.arange(0, 3, 1))

type(set_C)
S = set(["cat1", "dog1", "cat2", "dog2", "cat3", "cat4", 2j, False, 1, 1j+2, 2.5, 1+2j])

A = set(["cat1", "dog1", "cat2", "dog2", 1, 2.5, False, 1+2j])

B = set(["cat1", "dog1", "cat2", "dog2", "cat3", "cat4", 2j])
print(S.difference(A))

print(S - A)
print(S - A)

print(S - B)
print(B - A)

print(A - B)
A - A
print(A - (A - B))

print(B - (B - A))

print(B.intersection(A))

print(A.intersection(B))
A & B
print(A.union(B))

print(B.union(A))
print(A | B)
print(A.symmetric_difference(B))

print(B.symmetric_difference(A))

print('Verify: \n')

(A - B).union(B - A)
print('Illustration example')

##############################################

A = set([1 , 2, "1"])

B = set([2, "2", 1])

C = set(["3", 1, 2])

S = set([1,2,3, "1", "2", "3"])

E = set([])

print('Let S = %s be the sample_space and \n \t\t\t A = %s, B = %s, C = %s be its subsets'%(S , A, B, C))

##############################################

print('properties 1: \n\t  i) A sym_diff B = %s = %s = B sym_diff A '%(

    A.symmetric_difference(B), B.symmetric_difference(A)))

print('\t ii) (A sym_diff B) sym_diff C = %s = %s = A sym_diff (B sym_diff C)'%(

    (A.symmetric_difference(B)).symmetric_difference(C), 

    A.symmetric_difference((B).symmetric_difference(C))))

print('\tiii) A sym_diff B = %s = %s = A^c sym_diff B^c' %(

    A.symmetric_difference(B), (S - A).symmetric_difference(S - B) ) )

##############################################

print('properties 2: \n \t A sym_diff B = %s = %s = %s \ %s = (A U B) \ (A \cap B)'%(

    A.symmetric_difference(B), 

    ( A.union(B) ) - ( A.intersection(B) ),  

    A.union(B), A.intersection(B) ) )

##############################################

print('properties 3:')

print('\t  i) A sym_diff emptyset = %s = A'%(A.symmetric_difference(E)))

print('\t ii) A sym_diff A = %s = emptyset'%(A.symmetric_difference(A)))

##############################################

print('properties 4:')

print('\t  i) (A sym_diff B) sym_diff (B sym_diff C) = %s = %s = A sym_diff C'%(

    (A.symmetric_difference(B) ).symmetric_difference( (B.symmetric_difference(C) ) ),

    A.symmetric_difference(C)))

print('\t ii) A cap (B sym_diff C) = %s = %s = (A cap B) sym_diff (A cap C)'%(

    A.intersection( B.symmetric_difference(C) ), 

    ( A.intersection(B) ).symmetric_difference( A.intersection(C) ) ) )

##############################################
print(A ^ B)

print(A.symmetric_difference(B))
D = set([5, "6", "2"])

print('Is A and B disjoint? \t\t%s'%A.isdisjoint(B))

print('Is A and C disjoint? \t\t%s'%A.isdisjoint(C))

print('Is B and C disjoint? \t\t%s'%C.isdisjoint(B))

print('Is A and E(= emptyset) disjoint? %s'%A.isdisjoint(E))

print('Is D and A disjoint? \t\t%s'%D.isdisjoint(A))

print('Is D and B disjoint? \t\t%s'%D.isdisjoint(B))

print('Is D and C disjoint? \t\t%s'%D.isdisjoint(C))

print('Is D and E disjoint? \t\t%s'%E.isdisjoint(D))
print('"A is a subset of B??". This statement is: %s.'%A.issubset(B))

print('"A is contained in S??". This statement is: %s.'%S.issuperset(A))

print('"E is a subset of B??". This statement is: %s.'%E.issubset(B))

print('"D is a subset of D??". This statement is: %s.'%D.issubset(D))

print('"D is contained in D??". This statement is: %s.'%D.issuperset(D))
def is_sub_propsub(x , y):   

    """ 

        Input: x, y (set)

        return: x is subset or proper_subset or not a subset of y

    """

    if x < y:

        print('%s is a proper subset of %s'%(str(x), str(y)))

    elif x <= y:

        print('%s is a subset or identical to %s'%(str(x), str(y)))

    else:

        print('%s is not subset of %s'%(str(x), str(y)))

        

is_sub_propsub(D, D)

is_sub_propsub(D, S)

is_sub_propsub(A, S)

is_sub_propsub(A, B)
help(is_sub_propsub)
def is_sup_propsup(x , y):

    if x > y:

        print('%s is a proper superset of %s'%(str(x), str(y)))

    elif x >= y:

        print('%s is a superset or identical to %s'%(str(x), str(y)))

    else:

        print('%s is not superset of %s'%(str(x), str(y)))

        

is_sup_propsup(A, A)

is_sup_propsup(S, A)

is_sup_propsup(D, A)