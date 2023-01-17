

import pandas as pd
import numpy as np

testfile='../input/test.csv'
data = open(testfile).readlines()

sequences={}   #(key, value) = (id , sequence)
for i in range(1,len(data)): 
    line=data[i]
    line =line.replace('"','')
    line = line[:-1].split(',')
    id = int(line[0])
    sequence=[int(x) for x in line[1:]];
    sequences[id]=sequence
def checkRecurrence(seq, order= 2, minlength = 7):
    """
    :type seq: List[int]
    :type order: int
    :type minlength: int 
    :rtype: List[int]
    
    Check whether the input sequence is a recurrence sequence with given order.
    If it is, return the coefficients for the recurrenec relation.
    If not, return None.
    """     
    if len(seq)< max((2*order+1), minlength):
        return None
    
    ################ Set up the system of equations 
    A,b = [], []
    for i in range(order):
        A.append(seq[i:i+order])
        b.append(seq[i+order])
    A,b =np.array(A), np.array(b)
    try: 
        if np.linalg.det(A)==0:
            return None
    except TypeError:
        return None
   
    #############  Solve for the coefficients (c0, c1, c2, ...)
    coeffs = np.linalg.inv(A).dot(b)  
    
    ############  Check if the next terms satisfy recurrence relation
    for i in range(2*order, len(seq)):
        predict = np.sum(coeffs*np.array(seq[i-order:i]))
        if abs(predict-seq[i])>10**(-2):
            return None
    
    return list(coeffs)


def predictNextTerm(seq, coeffs):
    """
    :type seq: List[int]
    :type coeffs: List[int]
    :rtype: int
    
    Given a sequence and coefficienes, compute the next term for the sequence.
    """
    
    order = len(coeffs)
    predict = np.sum(coeffs*np.array(seq[-order:]))
    return int(round(predict))

seq = [1,5,11,21,39,73,139,269,527]
print (checkRecurrence(seq,3))
print (predictNextTerm(seq, [2,-5,4]))

#how many orders you want to check
#will search up to inspect - 1
inspect = 10

orders = [{}]*inspect
for r in range(2,inspect):
	orders[r] = {}
	for id in sequences:  
		if id in orders[r-1]: 
			continue
		seq = sequences[id]
		coeff = checkRecurrence(seq,r)
		if coeff!=None:
			predict = predictNextTerm(seq, coeff)
			orders[r][id]=(predict,coeff)

	print ("We found %d sequences\n" %len(orders[r]))

	print  ("Some examples\n")
	print ("ID,\t  Prediction,\t  Coefficients")
	for key in sorted(orders[r])[0:5]:
	    value = orders[r][key]
	    print ("%s, %s, \t %s" %(key, value[0], [int(round(x)) for x in value[1]]))



print("Conclusion:")
print("Number of sequences in the test set:", len(sequences))
for i in range(2,len(orders)):
	print("Number of %d order sequences:"%i, len(orders[i]))



