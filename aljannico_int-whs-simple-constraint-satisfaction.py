## Copyright bei Al, use freely but at your own risk! 

__version__ = 0.5 # For Python3



# Simple debugging

# debug = True  

debug = False



# Define some functions for constraints



def lessthan(i,j):

  return i <= j

  

def greaterthan(i,j):

  return i >= j

  

def equalto(i,j):

  return i == j

  

def notequalto(i,j):

  return not i == j



def strictlylessthan(i,j):

  return i < j



def strictlygreaterthan(i,j):

  return i > j



def unconstrained(i,j):

  return True

  

def divides(i,j):

  return not (j % i)



def is_divided_by(i,j):

  return divides(j,i)



def symmetric(func):

  if (func == lessthan): return greaterthan

  if (func == greaterthan): return lessthan

  if (func == notequalto): return notequalto

  if (func == strictlylessthan): return strictlygreaterthan

  if (func == strictlygreaterthan): return strictlylessthan

  if (func == equalto): return equalto

  if (func == divides): return is_divided_by

  if (func == is_divided_by): return divides

  ## Error handling - not truely the invers (though it would be easy to give it)

  if (func == unconstrained): return unconstrained

# Define basic algorithms

## Ändert D und gibt ein 2-Tupel von booleschen Werten zurück: 

## (CHANGE,EMPTY) - when CHANGE auf True steht, gab es eine

## Veränderung, wenn EMPTY auf True steht, ist das Domain von i

## leer

def revise(i,j):

  global D

  global CN

  global C



  if i < j:

    if debug: print("Enforce consistency of C_%d_%d: %d %s %d" % (i,j,i,CN[i][j].__name__,j))

    constraint = CN[i][j]

  else:

    if debug: print("Enforce consistency of C_%d_%d: %d %s %d" % (i,j,i,symmetric(CN[j][i]).__name__,j))

    constraint = symmetric(CN[j][i])



  if debug: print(" Domain %d: %s" % (i,repr(D[i])))

  if debug: print(" Domain %d: %s" % (j,repr(D[j])))

  

  if constraint == unconstrained: return (False,False)

  ## Watch it - this relies on the precondition that D[i] is not empty!

      

  ## Naive Checking

  change = False

  empty = True

  DNew = []

  for x in D[i]:

    xin = False

    for y in D[j]:

      if debug: print("Checking (%d,%d): %s" % (x,y,repr(constraint(x,y))))

      if constraint(x,y):

        DNew.append(x)

        empty = False

        xin = True

        break

    ## x will not be in der reduced set

    if not xin:

      print (" Checking C_%d_%d: %d %s %d -> REMOVED: %d" % \

            (i,j,x,repr(constraint(x,y)),y,x))

      change = True



  if debug: print("DNew: %s (Changed D[%d]: %s)" % (repr(DNew),i,repr(change)))

  if change:

    D[i] = DNew[:]



  ## Returning...

  return (change,empty)



## Naive Implementierung, entspricht AC-1

def enforce_arc_consistency():

  round = 1

  something_changed = True

  while something_changed:

    print("Enforcing Arc Consistency - Round %d" % round)

    round += 1

    something_changed = False

    for i in range(1,n):

      for j in range(i+1,n+1):

        

        (change,empty) = revise(i,j)

        ## Falls ein Domain leer wurde, ist das nicht lösbar

        if empty: return False

        ## Hat sich etwas verändert

        something_changed = something_changed or change



        # Rückrichtung        

        (change,empty) = revise(j,i)

        if empty: return False

        something_changed = something_changed or change

  print(" No Change!")

  return True



### Implementation of AC-4

def init_AC4():

  global D

  global C



  S = [None]

  Q = []

  counter = [None]

  # Initialize and populate the S structure

  for i in range(1,n+1):

    S.append({})

    counter.append([None])

    # Init

    for a in D[i]:

      S[i][a] = []

    for j in range(0,n+1):

      counter[i].append({})

  # Populate

  for i in range(1,n+1):

    for j in range(1,n+1):

      if not i == j:

        DNew = []

        for a in D[i]:

          count = 0

          for b in D[j]:

            if debug: print("Check (",i,":",a,",",j,":",b,"):")

            try:

              if i < j:

                C[i][j].index((a,b))

              else:

                C[j][i].index((b,a))

              count += 1

              # Value b of j supports value a of i

              S[j][b].append((i,a))

              if debug: print(" --> Found")

            except:

              if debug: print(" --> NOT Found")

              pass

          # print("C: ",i,j,a,count)

          counter[i][j][a] = count

          if count == 0:

            # no value has been found in D[j] to support a

            Q.append((i,a))

            print("Remove ",a," from D[",i,"] - no support from ",j)

          else:

            DNew.append(a)

        D[i] = DNew[:]

  # In Q are all pairs of Domain index, value which had no support in some D[j]]

  return Q, counter, S



def enforce_AC4():

  Q, counter, S = init_AC4()

  print ("Q: ", Q)

  print ("Counter: ", counter)

  print ("S: ", S)

  input()

  while Q:

    j,b = Q.pop() 

    for i,a in S[j][b]:

      counter[i][j][a] -= 1  

      if counter[i][j][a] == 0 and a in D[i]:

        D[i].remove(a)

        Q.append((i,a))

        print ("Remove ",a," from D[",i,"] - support ",b," from ",j," removed!")



  

def enforce_p_consistency(i,k,j):

  global C

  global D



  if debug: print ("Check (%d,%d) against %d" % (i,j,k))

  CNew = []

  empty = True

  change = False

  for e in C[i][j]:

    # e ist ein Tupel aus Rij, zu dem Partner-Kanten für ein Dreieck mit

    # einem Element aus D_k gefunden werden soll

    if debug: print(" Checking %s" % repr(e))

    xin = False

    for z in D[k]:

      try:

        if i < k:

          if debug: print("Looking for (%d,%d) in R_%d_%d" % (e[0],z,i,k))

          C[i][k].index((e[0],z))

        else:

          if debug: print("Looking for (%d,%d) in R_%d_%d" % (z,e[0],k,i))

          C[k][i].index((z,e[0]))

        if k < j:

          if debug: print("Looking for (%d,%d) in R_%d_%d" % (z,e[1],k,j))

          C[k][j].index((z,e[1]))

        else:

          if debug: print("Looking for (%d,%d) in R_%d_%d" % (e[1],z,j,k))

          C[j][k].index((e[1],z))

        # The triangle is complete!

        if debug: print("   Found!")

        CNew.append(e)

        xin = True

        empty = False

        break

      except:

        if debug: print("   NOT Found!")

        # One or both not found

        pass

      

    if not xin:

      print("  Checking C_%d_%d with x_%d: REMOVED %s" % (i,j,k,repr(e)))

      change = True

      

  if change:

    C[i][j] = CNew[:]

  

  ## Returning...

  return (change,empty)

  



def enforce_path_consistency():

  something_changed = True

  round = 1

  while something_changed:

    something_changed = False

    print("Runde %d" % round)

    round += 1

    for i in range(1,n):

      for j in range(i+1,n+1):

        # Jetzt mit jeder übrigen Variable k nach Dreiecken der Form:

        # (k,i) (i,j) (j,k) suchen

        # Auf die richtige Ordnung muss jeweils geachtet werden

        for k in range (1,n+1):

          ## Falsch bei Rina Dechter

          if (not k == i) and (not k == j):

            (change,empty) = enforce_p_consistency(i=i,k=k,j=j)

            if empty: return False

            something_changed = something_changed or change

  print("  Nothing changed")

  return True





  
# Some helpful functions



def clean_domains():

  global C

  global D



  def check_value(i,x):

    for j in range(1,i):

      if debug: print("  Check in R_%d_%d" % (j,i))

      # j < i

      for e in C[j][i]:

        if e[1] == x:

          break

      else: return False

      

    for j in range(i+1,n+1):

      if debug: print("  Check in R_%d_%d" % (i,j))

      # j > i

      for e in C[i][j]:

        if e[0] == x:

          break

      else: return False

    return True

  # End of local function check_value



  print("Cleaning Domains")

  for i in range(1,n+1):

    DNew = []

    for x in D[i]:

      if debug: print("Check Value %d of %d" % (x,i))

      if check_value(i,x):

        DNew.append(x)

      else:

        print("  Value %d removed from Domain D_%d" % (x,i))

    D[i] = DNew[:]

  print("Cleaning done")

        

def initial_relation(i,j):        

  global C

  global D

  global CN

  

  constraint = CN[i][j]

  

  for x in D[i]:

    for y in D[j]:

      if constraint(x,y):

        C[i][j].append((x,y))



        

def clean_constraints():

  global C

  global D



  print("Cleaning Constraints...")

  for i in range(1,n):

    for j in range(i+1,n+1):

      CNew = []

      for a,b in C[i][j]:

        if a in D[i] and b in D[j]:

          CNew.append((a,b))

      C[i][j] = CNew[:]



# Number of Variables

arc_checking = "AC1"

#arc_checking = "AC4" 
# Problem definition



def get_problem(problem):



    variables = [None,8,5,3,4]

    n = variables[problem]



    # Initial Domainvalues 

    if problem == 1:

        D = [None]

        for i in range(1,n+1):

            D.append([1,2,3,4])

    if problem == 2:

        D = [None,[1,2,3,4],[1,2,3],[0,1,2,3,4],[2,3,4],[0,1,2,3]]

    if problem == 3:

        D = [None,[2,5],[2,5],[2,4]]

    if problem == 4:

        D = [None,[0,1,3,4],[0,2,3,5],[1,2,4,5],[-1,3,4,7]]

    if debug: print(D)



    ## Naming the Constrains (setting unconstrained constraints to unconstrained)



    ## Das 0te-Element wird nicht benutzt, um der Aufgabenstellung zu entsprechen

    C = [None]

    CN = [None]

    for i in range(1,n):

        CN.insert(i,[])

        C.insert(i,[])

        for j in range(0,i+1):

            CN[i].insert(j,None)

            C[i].insert(j,None)

        for j in range(i+1,n+1):

            CN[i].insert(j,unconstrained)

            C[i].insert(j,[])



    if problem == 1:

        CN[1][2] = strictlylessthan

        CN[1][7] = lessthan

        CN[1][8] = greaterthan

        CN[2][3] = equalto

        CN[2][4] = notequalto

        CN[3][5] = strictlylessthan

        CN[3][6] = strictlygreaterthan

        CN[3][7] = strictlygreaterthan

        CN[4][5] = equalto

        CN[7][8] = greaterthan     



    if problem == 2:

        CN[1][2] = strictlylessthan

        CN[2][3] = strictlygreaterthan

        CN[3][4] = greaterthan

        CN[4][5] = strictlygreaterthan

        CN[1][5] = strictlygreaterthan

        

    if problem == 3:

        CN[1][2] = divides

        CN[1][3] = divides



    if problem == 4:

        D = [None,[0,1,3,4],[0,2,3,5],[1,2,4,5],[-1,3,4,7]]

        CN[1][2] = strictlygreaterthan

        CN[2][3] = greaterthan

        CN[3][4] = greaterthan

        CN[1][4] = strictlylessthan

    

    return n,D,C,CN
# debug = False

debug = True



# Solving the stuff

def solve():

  global n

  global D

  global C



  print("Solving the network...")

  sol = []



  def consistent(i,x,val):

    # Nothing to check yet

    if i == 1: return True

    # Den Wert für x gegen alle schon

    # vorgenommenen Zuweisungen checken.

    for j in range(1,i):

      if not (val[j],x) in C[j][i]: return False

    return True

  

  def solve_it(i,n,val):

    if debug: print(" Variable %d of %d" % (i,n))

    if i > n:

      sol.append(val[:])

      return



    found = False

    for x in D[i]:

      if debug: print("  Check Value %d" % x)

      if consistent(i,x,val):

        nval = val + [x]

        found = found or solve_it(i+1,n,nval)

      

  solve_it(1,n,[None])

  for e in sol:

    print(repr(e[1:]))

  return sol

      

######## Helper functions ##########

def show_network():

  global n

  global D

  global C

  global CN

  

  print("\nConstraint Network:")

  print(" %d Variables with the following domain" % n)

  for i in range(1,n+1):

    print("  Domain[%d] of Variable x_%d: %s" % (i,i,repr(D[i])))



  print(" Initial Constraints")

  for i in range(1,n):

    for j in range(i+1,n+1):

      if not CN[i][j] == unconstrained:

        print("  C_%d_%d: x_%d %s x_%d" % (i,j,i,CN[i][j].__name__,j))



  print(" Constraint Relations")

  for i in range(1,n):

    for j in range(i+1,n+1):

      print("  R_%d_%d: %s" % (i,j,repr(C[i][j])))



#### Start of main processing  

def compute_solution(problem=4):

    global n

    global D

    global C

    global CN

    

    n,D,C,CN = get_problem(problem)

    if arc_checking == "AC4":

        for i in range(1,n):

            for j in range(i+1,n+1):

                initial_relation(i,j)

        show_network()

        # input("Bitte Taste drücken")



        enforce_AC4()

        clean_constraints()

        show_network()

        # input("Bitte Taste drücken")



    if arc_checking == "AC1":

        # Enforcing Arc Consistency

        if not enforce_arc_consistency():

            print("Network not solvable! Arc consistency enforcing failed!")

            return

        else:

            ## Initial Population of the constraint relations

            for i in range(1,n):

                for j in range(i+1,n+1):

                    initial_relation(i,j)

            show_network()

            # input("Bitte Taste drücken")



    # Enforcing Path Consistency

    if not enforce_path_consistency():

        print("Network not solvable! Path consistency enforcing failed!")

        return



    ## Path Consistency COULD remove tuples from the relations such that

    ## not every value from the participating variable's domains 

    ## will be present anymore in some relation.

    ## Below, we clean the domains by intersecting the sets that result 

    ## from projecting out the respective

    ## variable from all the constraints it participates in.

    clean_domains()

    show_network()

    sol = solve()

    if len(sol) == 0:

        print("Netzwerk hat keine Lösung")



print(20*'#'+"\nProblem 4\n"+20*'#'+"\n")

compute_solution(problem=4)

print("\n\n"+20*'#'+"\nProblem 2\n"+20*'#'+"\n")

compute_solution(problem=2)
### Solving a specific path consistency problem "directly" 

n = 4

D = [None,[1,2,4],[2,3],[3,5,6],[4,5]]

C = [None]

for i in range(1,n):

    C.insert(i,[])

    for j in range(0,i+1):

        C[i].insert(j,None)

    for j in range(i+1,n+1):

        C[i].insert(j,[])

        

C[1][2] = [(1,2),(2,2),(4,3)]

C[1][3] = [(1,3),(1,6),(2,5),(4,6)]

C[1][4] = [(1,4),(2,4),(4,4),(4,5)]

C[2][3] = [(2,3),(2,6),(3,5),(3,6)]

C[2][4] = [(2,4),(2,5),(3,4),(3,5)]

C[3][4] = [(3,5),(5,5),(6,4)]



# Now, path consistency will be enforced!

enforce_path_consistency()

clean_domains() # not necessary if done by hand - but a good idea anyway ;)

show_network()



sol = solve()

if len(sol) == 0:

    print("Netzwerk hat keine Lösung")
