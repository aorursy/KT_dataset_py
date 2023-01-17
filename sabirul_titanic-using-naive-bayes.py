



#return possibility of a class

def getProb(vals,cl):

    counter=[colum[0] for colum in vals].count(cl)

    prob=float(counter)/len(vals)    

    return (prob,counter)



#look in the feature

def getFeatures(vals,pos):

    temp=[]

    for i in range(0,len(vals)):

          try:    

            temp.index(vals[i][pos])

    

          except:

            

            temp.append(vals[i][pos])

           

    return len(temp)

    



#get probability of a feature in a class

def getProbFeature(vals,cl,feature,pos,fcounter):

    #smoothing

    ccounter=1 

    

    for i in range(0,len(vals)):

        if vals[i][pos]==feature and vals[i][0]==cl:

            ccounter+=1

    

    

    prob=float(fcounter)/(ccounter+getFeatures(vals,pos)) 

    

    return prob

                               

                               



#probabilty of a given feautre

def probability(vals,cl,feats):

    

    pc,ccounter=getProb(vals,cl)

    var=pc



    count=0.0

    for i in range(0,len(vals)):

        if(vals[i][3]==''):

             continue

        count=count+float(vals[i][3])



    averageAge=count/len(vals)

    

    if(feats[2]==''):

        feats[2]=averageAge

        

    

    for i in range(0,len(feats)):

        

        if(i==6): 

            continue

        if(feats[i]==''):

            continue

        var=var*getProbFeature(vals,cl,feats[i],i+1,ccounter)



    return var

    

    

def getList(file,col):

    f=open(file,"r")

    l=f.readlines()

    vals=[x.strip('\n').split(",") for x in l][1:]

    

    for i in range(0,len(vals)):

             a=vals[i][col].strip(' ').split(".")

             vals[i].append(a[0])

    return vals









def naiveBayes(features):



    vals=getList('train.csv',4)





    for x in vals:

        del(x[0], x[2], x[3], x[6], x[7])

             

    pDeceased=probability(vals,"0",features)

    pSurvived=probability(vals,"1",features)

    var={'1':pSurvived,'0':pDeceased}

        

    return max(var,key=var.get)

                               

                               



## printing to csv



import csv



with open('predict.csv', 'w') as csvfile:

    fieldnames = ['PassengerId', 'Survived']

    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    tmp=getList('test.csv',3)



    for x in tmp:

        del(x[2], x[3], x[6], x[7])

    



    writer.writeheader()

    

    for i in range(0,418):

         writer.writerow({'PassengerId': tmp[i][0], 'Survived': naiveBayes([tmp[i][1],tmp[i][2],tmp[i][3],tmp[i][4],tmp[i][5],tmp[i][6],tmp[i][7],tmp[i][8]])})






