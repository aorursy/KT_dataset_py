import numpy as np 

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import mixture

from sklearn.metrics import confusion_matrix

%matplotlib inline
df = pd.read_csv("../input/HR_comma_sep.csv")

mymap = {'accounting':1,'hr':2,'IT':3, 'management': 4 ,'marketing':3, 'product_mng' :5 , 

         'RandD':6,'sales':7,'support':8,'technical':9,}

dfh =df.applymap(lambda s: mymap.get(s) if s in mymap else s)

mymap1 = {'low':1,'medium':2,'high':3}

#

dfh1 =dfh.applymap(lambda s: mymap1.get(s) if s in mymap1 else s)

dfh1.head(5)
#InterimDF Dropped-"left"[Target_Feature]

dfh2 = dfh1.drop(df.columns[[6]],axis=1,inplace=False) 

names = dfh2.columns.values



print(names)

print(dfh2.shape)

print("_"*90)





dfh3 = pd.DataFrame(dfh1["left"]) #  Interim DF only - "left"



names1 = dfh3.columns.values

print(names1)

print(dfh3.shape)

print("_"*90)

#

print(dfh3["left"].value_counts()) # Here - 0 == Live Employee , 1 == Exited Employee / Attrited Employee 

#

print("_"*90)

frames = [dfh2, dfh3]

df3 = pd.concat(frames,axis=1)

print(df3.shape)

print(df3.head(5))

#df3.to_csv('dfh3_tree_kaggle.csv') # Ok for down csv 
# Convert DF to Numpy Array 

# 1st Numpy Array == X , all Features





import numpy as np



X = df3.iloc[:,0:10].values # All Features also Target Feature "left"

print(X.shape)

print ("_"*90)

print('Percentage of Class Label ==1 = {:.4f}'.format(df3["left"].mean()))

print('Percentage of Class Label ==0 = {:.4f}'.format(1-df3["left"].mean()))

print ("_"*90)

print ("Model that Predicts 76.19% Accuracy is Non Predictor OR NO_Model- as it will always predict Dominant Class")

print ("Dominant Class = ZERO or LIVE EMPLOYEE - we need more than 76.19% Accuracy Score.")
my_data = X

print(my_data) # Step isnt required couldve kept it as X - just to sync with original code from book 
class decisionnode:

  def __init__(self,col=-1,value=None,results=None,tb=None,fb=None): # __init__ methods can take any number of arguments, and just like functions, the arguments can be defined with default values, making them optional to the caller.

    self.col=col

    self.value=value

    self.results=results

    self.tb=tb    # tb= True Branch / Right Branch 

    self.fb=fb    # fb = False Branch / Left B 



# Divides a set on a specific column. Can handle numeric or nominal values

def divideset(rows,column,value):

    # Make a function that tells us if a row is in 

    # the first group (true) or the second group (false)

   split_function=None

   if isinstance(value,int) or isinstance(value,float):    ## IF - OR - ELSE - Loop

      split_function=lambda row:row[column]>=value         # Code fro Numeric Values 

   else:

      split_function=lambda row:row[column]==value         # Code for Nominal Values 

   

   # Divide the rows into two sets and return them

   set1=[row for row in rows if split_function(row)]

   set2=[row for row in rows if not split_function(row)]

   #print set1               # Own Code Snippet

   #print type(set1)         # Own Code Snippet

   #print "_"*120 

   #print set2               # Own Code Snippet

   #print type(set2)         # Own Code Snippet



   return (set1,set2)



# All the - # Own Code Snippet's - above need not be run ...
# Counts the UNIQUE VALUES for each PRED CLASS or TARGET VAR Values in LAST COLUMN of DATA SET 

# if a Certain Data Set doesnt have TARGET VAR as LAST FEATURE or COLUMN we need to make it so .....



def uniquecounts(rows):

   results={}

   for row in rows:

      # The result is the last column

      rr=row[len(row)-1]

      

      if rr not in results: results[rr]=0

      results[rr]+=1

   return results
# Unique Count of Classes of Target Feature 



uniquecounts(my_data)


def giniimpurity(rows):

  total=len(rows)

  #print(total)          #______________________________________Test

  counts=uniquecounts(rows)

  #print(counts)         #______________________________________Test

  imp=0

  for k1 in counts:

    p1=float(counts[k1])/total

    #print ("_"*100+"_||_") #______________________________________Test

    #print ((counts[k1]))   #______________________________________Test

      

    for k2 in counts:

      if k1==k2: continue

      p2=float(counts[k2])/total

      #print ("_"*100)      #______________________________________Test

      #print ((counts[k2])) #______________________________________Test

      imp+=p1*p2

  #print ("_"*60)           #______________________________________Test

  #print ((counts[k1]))     #______________________________________Test 

  #print ("_"*60)           #______________________________________Test

  #print ((counts[k2]))     #______________________________________Test

  return imp



# Probability that a randomly placed item will be in the wrong category

# Wiki -- Used by the CART (classification and regression tree) algorithm, Gini impurity 

# is a - "measure of how often"- a randomly chosen element from the set would be incorrectly labeled 

# if it was randomly labeled according to the distribution of labels in the subset.
giniimpurity(my_data)
# Original entropy Func.

#



def entropy(rows):

   from math import log             

   log2=lambda x:log(x)/log(2)      

   results=uniquecounts(rows)       ## Same as -"counts=uniquecounts(rows)" -in GiniImpurity Function

   # Now calculate the entropy

   ent=0.0

   for r in results.keys():         ## This DATA_SET has TWO results.keys() as only 2 CLASSES of TARGET VAR. 

      p=float(results[r])/len(rows) ## Probability of Occurence of TARGET CLASS 0 OR 1 

      ent=ent-p*log2(p)

   return ent

   print(ent)
entropy(my_data)
#SandBox for --- def entropy(rows):



results=uniquecounts(my_data)   # Same as -"counts=uniquecounts(rows)" -in GiniImpurity Function

for r in results.keys():

        print("TEST - as many times this prints - that many results.keys() and Classes ")

   
#SandBox for --- def entropy(rows):



from math import log             

log2=lambda x:log(x)/log(2)      ## x = Parameter is passed Value = p below.



results=uniquecounts(my_data)    # Same as -"counts=uniquecounts(rows)" -in GiniImpurity Function

ent=0.0

for r in results.keys():    

    p=float(results[r])/len(my_data) # p=Probability of Occurence of TARGET CLASS 0 OR 1

    ent=ent-p*log2(p)

    print("Value for p==",p)            # First time through FOR LOOP - Prints the p for results.keys - 1st VALUE then the next . 

    print("Value for ent==",ent)           # Seen above also- get FINAL_VALUE of ENTROPY  == ent = 0.79183

    

    

# Entropy is the sum of p(x)log(p(x)) across all the different possible results - all possible Unique Values of TARGET VAR like 0 and 1 - Live or Exited.   

# Simply stated - Entropy is Inversely Proportional to Purity or HomoGenity of the SET.  



# If All Observations in SET are of 1 CLASS only - say ATTRITED - then SET is Totally PURE or HOMOGENOUS.

# In such a case ENTROPY is Lowest 

    
## SAND BOX Code Need not be run - dividing the Data Manually into sets , the column values and split point values 

## have been randomly and manually chosen - this code is sandbox to explain whats happening with this function 



# Split -1 

set1,set2=divideset(my_data,0,0.36) # Column = 0 = satisfaction_level , Value >= 0.36 Assigned to SET -1 



print ("_"*100+"__|||__")

print ("# Split -1")

print ("Metrics_SET(1):")

print ('Gini_Impurity :-{:.4f}'.format(giniimpurity(set1)))

print ("Entropy       :-{:.4f}".format(entropy(set1)))

print ("UniqueCount_TargetVar:-",uniquecounts(set1))

print ("_"*50+"__|||__")

print ("Metrics_SET(2):")

print ('Gini_Impurity :-{:.4f}'.format(giniimpurity(set2)))

print ("Entropy       :-{:.4f}".format(entropy(set2)))

print ("UniqueCount_TargetVar:-",uniquecounts(set2))





# Split -2

set1,set2=divideset(my_data,1,0.55) # Column = 1 = last_evaluation , Value >= 0.55 Assigned to SET -1 



print ("_"*100+"__|||__")

print ("# Split -2")

print ("Metrics_SET(1):")

print ('Gini_Impurity :-{:.4f}'.format(giniimpurity(set1)))

print ("Entropy       :-{:.4f}".format(entropy(set1)))

print ("UniqueCount_TargetVar:-",uniquecounts(set1))

print ("_"*50+"__|||__")

print ("Metrics_SET(2):")

print ('Gini_Impurity :-{:.4f}'.format(giniimpurity(set2)))

print ("Entropy       :-{:.4f}".format(entropy(set2)))

print ("UniqueCount_TargetVar:-",uniquecounts(set2))





# Split -3

set1,set2=divideset(my_data,3,148) # Column = 3 = average_montly_hours, Value >= 148 Assigned to SET -1 



print ("_"*100+"__|||__")

print ("# Split -3")

print ("Metrics_SET(1):")

print ('Gini_Impurity :-{:.4f}'.format(giniimpurity(set1)))

print ("Entropy       :-{:.4f}".format(entropy(set1)))

print ("UniqueCount_TargetVar:-",uniquecounts(set1))

print ("_"*50+"__|||__")

print ("Metrics_SET(2):")

print ('Gini_Impurity :-{:.4f}'.format(giniimpurity(set2)))

print ("Entropy       :-{:.4f}".format(entropy(set2)))

print ("UniqueCount_TargetVar:-",uniquecounts(set2))
def buildtree(rows,scoref=entropy):

  if len(rows)==0: return decisionnode()

  current_score=scoref(rows)



  # Set up some variables to track the best criteria

  best_gain=0.0

  best_criteria=None

  best_sets=None

  

  column_count=len(rows[0])-1       ### Commented below in Markdown Cell 

  for col in range(0,column_count): ###

    # Generate the list of different values in this column

    column_values={}                ###   

    for row in rows:                ###   

       column_values[row[col]]=1

    # Now try dividing the rows up for each value in this column

    for value in column_values.keys():

      (set1,set2)=divideset(rows,col,value)

      #print  "Col_Value_Keys :---",column_values.keys()  ##### OWN CODE for Print ............

      

      # Information gain

      p=float(len(set1))/len(rows)

      gain=current_score-p*scoref(set1)-(1-p)*scoref(set2)

      if gain>best_gain and len(set1)>0 and len(set2)>0:

        best_gain=gain

        best_criteria=(col,value)

        best_sets=(set1,set2)

  # Create the sub branches   

  if best_gain>0:

    trueBranch=buildtree(best_sets[0]) # Right Branch 

    falseBranch=buildtree(best_sets[1]) # Left Branch 

    return decisionnode(col=best_criteria[0],value=best_criteria[1],

                        tb=trueBranch,fb=falseBranch)

  else:

    return decisionnode(results=uniquecounts(rows))
# SandBox - Info_Gain

#

scoref=entropy

rows=my_data

p=float(len(set1))/len(rows)                        ### As-Is from above 

current_score=scoref(rows)                          ### As-Is from above 



# gain=current_score-p*scoref(set1)-(1-p)*scoref(set2)

# gain = is difference between current entropy and weighted-average entropy of the two sets. 



print("Length of set1:--------------" ,float(len(set1)))

print("Length of rows or my_data:--- ",len(my_data))

print("Current Score :--------------", scoref(my_data)) # 0.7918370416559191 - for HR_Analytics_Data Set 

print("Value of Gain :--------------" , current_score-p*scoref(set1)-(1-p)*scoref(set2)) # 0.011324073460184436



best_sets=(set1,set2)

print("_"*90)

print("Best Sets is a :--",type(best_sets))



print("_"*90)

#print (best_sets[0]) # OK Dont Print - Large Output 

#print("_#_"*50)

#print (best_sets[1]) # OK Dont Print - Large Output 



#buildtree(best_sets[0]) #Run only offline Not on Kaggle - Large OutPut Memory intensive not required 
#

def printtree(tree,indent=''):

   # Is this a leaf node?

   if tree.results!=None:

      print(str(tree.results))

   else:

      # Print the criteria

      print(str(tree.col)+':'+str(tree.value)+'? ')



      # Print the branches

      print(indent+'T->',)

      printtree(tree.tb,indent+'  ')

      print(indent+'F->',)

      printtree(tree.fb,indent+'  ')
# Function - printtree()



tree_1=buildtree(my_data) # Dont print - large OutPut 

#printtree(tree_2) # Dont print actual tree created in line 1 here 
def getwidth(tree):

  if tree.tb==None and tree.fb==None: return 1 # 1 is Default WIDTH of TREE if NO BRANCHES 

  return getwidth(tree.tb)+getwidth(tree.fb)



def getdepth(tree):

  if tree.tb==None and tree.fb==None: return 0

  return max(getdepth(tree.tb),getdepth(tree.fb))+1





from PIL import Image,ImageDraw



def drawtree(tree,jpeg='tree.jpg'):

  w=getwidth(tree)*100

  h=getdepth(tree)*100+120



  img=Image.new('RGB',(w,h),(255,255,255))

  draw=ImageDraw.Draw(img)



  drawnode(draw,tree,w/2,20)

  img.save(jpeg,'JPEG')

  

def drawnode(draw,tree,x,y):

  if tree.results==None:

    # Get the width of each branch

    w1=getwidth(tree.fb)*100

    w2=getwidth(tree.tb)*100



    # Determine the total space required by this node

    left=x-(w1+w2)/2

    right=x+(w1+w2)/2



    # Draw the condition string

    draw.text((x-20,y-10),str(tree.col)+':'+str(tree.value),(0,0,0))



    # Draw links to the branches

    draw.line((x,y,left+w1/2,y+100),fill=(255,0,0))

    draw.line((x,y,right-w2/2,y+100),fill=(255,0,0))

    

    # Draw the branch nodes

    drawnode(draw,tree.fb,left+w1/2,y+100)

    drawnode(draw,tree.tb,right-w2/2,y+100)

  else:

    txt=' \n'.join(['%s:%d'%v for v in tree.results.items()])

    draw.text((x-20,y),txt,(0,0,0))

def classify(observation,tree): # OBS = 1 Row of Data all Features besides Target. # Tree = tree created above. 

  if tree.results!=None:

    return tree.results

  else:

    v=observation[tree.col]

    branch=None

    if isinstance(v,int) or isinstance(v,float):

      if v>=tree.value: branch=tree.tb

      else: branch=tree.fb

    else:

      if v==tree.value: branch=tree.tb

      else: branch=tree.fb

    return classify(observation,branch) # 
obs0 = my_data[0,:]

obs1 = my_data[1,:]

obs2 = my_data[2,:]

obs3 = my_data[3,:]

obs4 = my_data[4,:]

obs5 = my_data[5,:]

obs6 = my_data[6,:]

obs2003 = my_data[2003,:]

obs2004 = my_data[2004,:]

obs2005 = my_data[2005,:]

obs2006 = my_data[2006,:]

obs2007 = my_data[2007,:]

obs2008 = my_data[2008,:]

obs2009 = my_data[2009,:]





print(obs0)

print(obs1)

print(obs2)

print(obs3)

print(obs4)

print(obs5)

print(obs6)

print(obs2003)

print(obs2004)

print(obs2005)

print(obs2006)

print(obs2007)

print(obs2008)

print(obs2009)



print(classify(obs0,tree_1))

print(classify(obs1,tree_1)) # 

print(classify(obs2,tree_1)) # 

print(classify(obs3,tree_1)) # 

print(classify(obs4,tree_1)) # 

print(classify(obs5,tree_1)) # 

print(classify(obs6,tree_1)) # 

print(classify(obs2003,tree_1)) # 

print(classify(obs2004,tree_1)) # 

print(classify(obs2005,tree_1)) # 

print(classify(obs2006,tree_1)) # 

print(classify(obs2007,tree_1)) # 

print(classify(obs2008,tree_1)) # 

print(classify(obs2009,tree_1)) # 



# As seen below most PREDICTED classes are correct 

# Further ToBeDone ...................................Prune etc ... 

# .... back soon . 