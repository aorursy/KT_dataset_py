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
order2Seq={}   #(key, value) = (sequence id, [prediction, coefficients])
for id in sequences:  
    seq = sequences[id]
    coeff = checkRecurrence(seq,2)
    if coeff!=None:
        predict = predictNextTerm(seq, coeff)
        order2Seq[id]=(predict,coeff)

print ("We found %d sequences\n" %len(order2Seq))

print  ("Some examples\n")
print ("ID,  Prediction,  Coefficients")
for key in sorted(order2Seq)[0:5]:
    value = order2Seq[key]
    print ("%s, %s, %s" %(key, value[0], [int(round(x)) for x in value[1]]))
order3Seq={}
for id in sequences:
    if id in order2Seq:
        continue
    seq = sequences[id]
    coeff = checkRecurrence(seq,3)
    if coeff!=None:
        predict = predictNextTerm(seq, coeff)
        order3Seq[id]=(predict,coeff)

print ("We found %d sequences\n" %len(order3Seq))

print  ("Some examples\n")
print ("ID,  Prediction,  Coefficients")
for key in sorted(order3Seq)[0:5]:
    value = order3Seq[key]
    print ("%s, %s, %s" %(key, value[0], [int(round(x)) for x in value[1]]))
order4Seq={}
for id in sequences:  
    if id in order2Seq or id in order3Seq:
        continue
    seq = sequences[id]
    coeff = checkRecurrence(seq,4)
    if coeff!=None:
        predict = predictNextTerm(seq, coeff)
        order4Seq[id]=(predict,coeff)

print ("We found %d sequences \n" %len(order4Seq))
print  ("Some examples\n")
print ("ID,  Prediction,  Coefficients")
for key in sorted(order4Seq)[4:5]:
    value = order4Seq[key]
    print ("%s, %s, %s" %(key, value[0], [int(round(x)) for x in value[1]]))

print (sequences[239][0:17])
print("Conclusion:")
print("Number of sequences in the test set:", len(sequences))
print("Number of 2nd order sequences:", len(order2Seq))
print("Number of 3rd order sequences:", len(order3Seq))
print("Number of 4th order sequences:", len(order4Seq))