import pandas as pd
import numpy as np
import time
t0=time.time()

trainfile='../input/train.csv'
testfile='../input/test.csv'

train_df= pd.read_csv(trainfile, index_col="Id")
test_df = pd.read_csv(testfile, index_col="Id")

train_seqs= train_df['Sequence'].to_dict()
test_seqs= test_df['Sequence'].to_dict()

for key in train_seqs:
    seq=train_seqs[key]
    seq=[int(x) for x in seq.split(',')]
    train_seqs[key]=seq

for key in test_seqs:
    seq=test_seqs[key]
    seq=[int(x) for x in seq.split(',')]
    test_seqs[key]=seq

MIN_LENGTH = 10  #Ignore sequences with length<10

print ("Time Elapsed:  %.2f seconds" %(time.time()-t0))
#import fractions
import math
def findGCD(seq):
    """ Compute the greatest common divisor of a list of numbers. """
    gcd = seq[0]
    for i in range(1,len(seq)):
        #gcd=fractions.gcd(gcd, seq[i])
        gcd=math.gcd(gcd, seq[i])
    return gcd

def findSignature(seq, n = MIN_LENGTH):
    """ Compute the signature of the sequence using the first n elements
        if the length of sequence is less than n, return the empty tuple. """  
    if len(seq)<n:
        return tuple([])
    
    difference = [seq[i]-seq[i-1] for i in range(1,n)]
    nonzero_difference = [d for d in difference if d!=0]
    if len(nonzero_difference)==0:
        return tuple([0]*(n-1))
    else:
        sign = 1 if nonzero_difference[0]>0 else -1
        
    gcd = findGCD(difference)
    return tuple([sign*x/gcd for x in difference])
def findLine(x,y, n, requireInteger=True, useNumpy=False):  
    """ Find [m,b] so that y=mx+b holds for the first n points: (x1,y1), (x2,y2),...(xn,yn)

    Args:
        x,y: list[int]
        n: int, number of points fitted
        requireInteger: boolean, whether m,b must be integers
        
    Returns:
        [m,b]: int m, int b
    
    Remark:
        This should be faster than numpy.polyfit(x,y,1) 
    """
     
    #  Find m,b use the first two points (x0,y0),(x1,y1) 
    #  Formula: m = (y1-y0)/(x1-x0).
    #  If the denominator becomes zero, use the next points.   
    x0 = x[0]
    i = 1
    while(i<n-1 and x[i]==x[0]):
        i+=1
    x1=x[i]
    if x1==x0:
        return None
    else:
        y0,y1 = y[0],y[i]
    m = 1.0*(y1-y0)/(x1-x0)
    b = y[0]-m*x[0]
    
    # Check if m,b are integers
    if requireInteger:
        m_int = int(round(m))
        b_int = int(round(b))
        if abs(m-m_int)>10**(-2) or abs(b-b_int)>10**(-2):
            return None
        else:
            m, b = m_int, b_int
    
    # Check if the next points satisfty y=mx+b
    if useNumpy:
        y_predict = m*np.array(x)+b
        difference = np.abs(np.array(y[0:n])-y_predict)
        error = np.max(difference)   
    else:
        y_predict = [m*x[i]+b for i in range(n)]
        difference = [abs(y[i]-y_predict[i]) for i in range(n)]
        error = max(difference)
        
    if error<10**(-2):
        return [m,b]
import time
t1= time.time()


# Compute signatures using the first 10 elements.
minlength = MIN_LENGTH
train_df['Signature'] = [findSignature(train_seqs[id][:minlength], minlength) for id in train_df.index]
test_df['Signature'] = [findSignature(test_seqs[id][:minlength], minlength) for id in test_df.index]

# Group data frames by signatures
train_gb = train_df.groupby(['Signature'], sort=True)
test_gb = test_df.groupby(['Signature'], sort=True)

# Find signatures that appear in both train/test sets
commonSignatures = list(set(test_gb.groups.keys()).intersection(train_gb.groups.keys()))
commonSignatures.remove(tuple([]))

print ("Time Elapsed: %.0f seconds" %(time.time()-t1))
result={}
import time
t0=time.time()

# For every (test, train) pair of sequences with the same signature,
# Let (x,y)= (test, train) or (x,y)=(train, test),
# verify whether y=mx+b.
# Requirement: Train sequence must be longer than test sequence to make prediction

for signature in commonSignatures:
    for test_id in test_gb.groups[signature]:
        test_seq = test_seqs[test_id]
        n = len(test_seq)
        train_candidates = train_gb.groups.get(signature)

        for train_id in train_candidates:
            train_seq=train_seqs[train_id]
            if len(train_seq)<=n: # too short to  make prediction
                continue
             
            # Check if train = m*test + b
            line = findLine(train_seq,test_seq, n)  
            if line:
                [m,b] = line
                predict = str(m*train_seq[n]+b)
                result[test_id] = (train_id, [m,b], '(train,test)', predict)
                break
            
            # Check if test = m*train + b
            line = findLine(test_seq,train_seq, n)
            if line:
                [m,b] = line
                if m!=0:
                    predict = str((train_seq[n]-b)/m)
                    result[test_id] = (train_id, [m,b], '(test,train)', predict)
                    break
print ("Time Elapsed: %.0f seconds" %(time.time()-t0))
match_df = pd.DataFrame.from_dict(result, orient='index', dtype=None)
match_df.columns=['TrainID', '[m,b]','(x,y)', 'Prediction']
match_df.index.name="TestID"
match_df=match_df.sort_index()

match_df.to_csv("matchPairs.csv")
print ("Sample output, rows 25-30: ")
match_df[25:30]
print ("Test Sequence #1692: ", (test_seqs[1692]), "\n")
print ("Train Sequence #37421: ", (train_seqs[37421]))
print ("Conclusion: \n")
print ("Number of test sequences: %s" %len(test_seqs))
print ("Number of matches found: %s   (%.2f%% of the test set)" %(len(match_df) , 100.0*len(match_df)/len(test_seqs)))