
def weights_exact(step,sum,n):
    if sum%step!=0:
        return "This step is not allowed."
    w=list()
    weights=list()
    w[0:n]=[0]*(n)
    w[0]=sum
    counter=0
    i=0
    weights.append(w.copy())
    while w[n-1]!=sum:
        w[i] =0
        i=i+1
        w[i]=w[i]+step        
        counter=counter+step
        if counter==sum:
            counter=counter-w[i]
        else:
            w[0]=sum-counter
            i=0
        weights.append(w.copy())
    return weights
    
        
w=weights_exact(10,50,3)
print(w)
def weights(step,sum,n):    
    sum_adj=round(int(sum/step)*step,5)
    w=list()
    weights=list()
    w[0:n]=[0]*n
    w[0]=sum    
    counter=0
    i=0
    weights.append(w.copy())
    while round(w[n-1],5)!=sum:
        w[i] =0        
        i=i+1     
        w[i]=round(w[i]+step,5)    
        counter=round(counter+step,5)
        if counter==sum_adj:
            counter=round(counter-w[i],5)
            w[i]=round(w[i]+sum-sum_adj,5)
        else:
            w[0]=round(sum-counter,5)
            i=0
        weights.append(w.copy())
    return weights
    
w=weights(33.33,100,3)
print(w)
