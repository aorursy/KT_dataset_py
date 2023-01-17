## Problemrepräsentation

problem1 = (("1","101"),("10","00"),("011","11"))

# Not elegant, made for demo purposes
def isValid(x,y):
    lx = len(x)
    ly = len(y)
    length = lx if lx < ly else ly
    for i in range(length):
        if x[i] != y[i]:
            return False
    return True

# some tests
print(isValid("100","1001"))   # -> True
print(isValid("1001","100"))   # -> True
print(isValid("","100"))       # -> True 
print(isValid("100",""))       # -> True
print(isValid("",""))          # -> True
print(isValid("1001","1000"))  # -> False      
print(isValid("0","1"))        # -> False
class Queue:
    def __init__(self):
        # Creates an internal list to manage the queue
        self._queue = []
        
    def push(self,element):
        # Appends the new element to the end of the internal list
        self._queue.append(element)
        
    def pop(self): 
        # Returns the first element in the internal list
        return self._queue.pop(0)
    
    def isEmpty(self):
        # Checks, if the list is empty
        return len(self._queue) == 0

    
def search(problem,debug=True):
    first = ("","") # no domino selected
    q = Queue()
    q.push(first) 
    
    stop = False
    solution = None
    node_count = 0
    while not stop:
        if q.isEmpty(): # Wir können nichts mehr ausprobieren!
           stop = True
        else:
            x,y = q.pop()
            node_count += 1
            if not node_count % 50: print(".",end="")
            if node_count and not node_count % 4000: print("  (",node_count,")")
            if debug: print("x,y:",x,y)
            # Check, if this is a solution
            if x == y and x != '' and y != '':
                solution = (x,y)
                stop = True
            elif isValid(x,y):
                for domino in problem:
                    xi,yi = domino
                    q.push((x+xi,y+yi))
            else:
                # Ok, ignore this search node
                pass # do nothing
    print("\nThis search inspected ",node_count,"nodes in the search tree!")
    return solution
result1 = search(problem1)
print("Result: ",result1)
problem2 = (("001","0"),("01","011"),("01","101"),("10","001"))
result2 = search(problem2,debug=False)
print("\nResult for Problem 2: ",result2)
problem3 = (("0","1"),) # see https://wiki.python.org/moin/TupleSyntax, "," creates a tuple, not "()"
print(search(problem3))
def search(problem,debug=True):
    first = ("","",[]) # no domino selected (CHANGED)
    q = Queue()
    q.push(first) 
    
    stop = False
    solution = None
    node_count = 0
    while not stop:
        if q.isEmpty(): # Wir können nichts mehr ausprobieren!
           stop = True
        else:
            x,y,idx = q.pop() # (CHANGED)
            node_count += 1
            if not node_count % 50: print(".",end="")
            if node_count and not node_count % 4000: print("  (",node_count,")")
            if debug: print("x,y:",x,y,idx) # (CHANGED)
            # Check, if this is a solution
            if x == y and x != '' and y != '':
                solution = idx,x,y # (CHANGED)
                stop = True
            elif isValid(x,y):
                for index,domino in enumerate(problem): # (CHANGED)
                    xi,yi = domino
                    q.push((x+xi,y+yi,idx+[index])) # (CHANGED)
            else:
                # Ok, ignore this search node
                pass # do nothing
    print("\nThis search inspected ",node_count,"nodes in the search tree!")
    return solution
result1 = search(problem1)
print("Result: ",result1)
problem2 = (("001","0"),("01","011"),("01","101"),("10","001"))
result2 = search(problem2,debug=False)
print("\nResult for Problem 2: ",result2)
def isValid(x,y):
    global count_vgl
    lx = len(x)
    ly = len(y)
    length = lx if lx < ly else ly
    for i in range(length):
        count_vgl += 1 # Frage: Warum steht das nicht im if oder hinter dem if?
        if x[i] != y[i]:
            return False
    return True
count_vgl = 0 # stilistisch nicht schön, globale Variable, use at your own risk!
result1 = search(problem1,debug=True)
print("Anzahl an Einzelzeichen-Vergleichen:",count_vgl)
count_vgl = 0 # stilistisch nicht schön, globale Variable, use at your own risk!
result2 = search(problem2,debug=False)
print("Anzahl an Einzelzeichen-Vergleichen:",count_vgl)
def isValid(x,y,start=0):
    global count_vgl
    lx = len(x)-start
    ly = len(y)-start
    length = lx if lx < ly else ly
    for i in range(length):
        count_vgl += 1 # Frage: Warum steht das nicht im if oder hinter dem if?
        if x[i+start] != y[i+start]:
            return False,0
    return True,length+start

def search(problem,debug=True):
    first = ("","",[],0) # no domino selected (CHANGED)
    q = Queue()
    q.push(first) 
    
    stop = False
    solution = None
    node_count = 0
    while not stop:
        if q.isEmpty(): # Wir können nichts mehr ausprobieren!
           stop = True
        else:
            x,y,idx,tested = q.pop() 
            node_count += 1
            if not node_count % 50: print(".",end="")
            if node_count and not node_count % 4000: print("  (",node_count,")")
            if debug: print("x,y:",x,y,idx,tested) # (CHANGED)
            # Check, if this is a solution
            if x == y and x != '' and y != '':
                solution = idx,x,y
                stop = True
            else: # (CHANGED)
                valid,tested = isValid(x,y,start=tested) # (CHANGED)
                if valid: # (CHANGED)
                    for index,domino in enumerate(problem): 
                        xi,yi = domino
                        q.push((x+xi,y+yi,idx+[index],tested))                 
    print("\nThis search inspected ",node_count,"nodes in the search tree!")
    return solution
count_vgl = 0 # stilistisch nicht schön, globale Variable, use at your own risk!
result1 = search(problem1,debug=True)
print("Result: ",result1)
print("\nAnzahl an Einzelzeichen-Vergleichen:",count_vgl)
count_vgl = 0 # stilistisch nicht schön, globale Variable, use at your own risk!
result2 = search(problem2,debug=False)
print("Result: ",result2)
print("\nAnzahl an Einzelzeichen-Vergleichen:",count_vgl)
def isValid(x,y):
    global count_vgl
    lx = len(x)
    ly = len(y)
    length = lx if lx < ly else ly
    for i in range(length):
        count_vgl += 1 # Frage: Warum steht das nicht im if oder hinter dem if?
        if x[i] != y[i]:
            return False,None,None
    return True,x[length:],y[length:]

def search(problem,debug=True):
    first = ("","",[]) # no domino selected (CHANGED)
    q = Queue()
    q.push(first) 
    
    stop = False
    solution = None
    node_count = 0
    while not stop:
        if q.isEmpty(): # Wir können nichts mehr ausprobieren!
           stop = True
        else:
            x,y,idx = q.pop() # (CHANGED)
            node_count += 1
            if not node_count % 50: print(".",end="")
            if node_count and not node_count % 4000: print("  (",node_count,")")
            if debug: print("x,y:",x,y,idx) # (CHANGED)
            # Check, if this is a solution
            if x == y and x != '' and y != '':
                solution = idx,x,y 
                stop = True
            else:
                valid,new_x,new_y = isValid(x,y)
                if valid:
                    for index,domino in enumerate(problem): 
                        xi,yi = domino
                        q.push((new_x+xi,new_y+yi,idx+[index]))                 
    print("\nThis search inspected ",node_count,"nodes in the search tree!")
    return solution
count_vgl = 0 # stilistisch nicht schön, globale Variable, use at your own risk!
result1 = search(problem1,debug=True)
print("Result: ",result1)
print("\nAnzahl an Einzelzeichen-Vergleichen:",count_vgl)
# Hier noch der Code, um die String-Lösung für eine gegebene Sequenz von Indices zu erzeugen:

def solve(problem,sequence):
    x = []
    y = []
    for idx in sequence:
        x += [problem[idx][0]] # x-Position des Dominos mit Index idx
        y += [problem[idx][1]] # y-Position des Dominos mit Index idx
    return ''.join(x),''.join(y)
        
sequence,_,_ = search(problem1,debug=True)
print("Lösung als Sequenz von Indices: ",sequence)
x,y = solve(problem1,sequence)
print("Lösung als x- und y-String: ","x=",x,"und y=",y)
# Hier noch die Version von search aufschreiben, die ohne Rückgabe von x und y auskommt. 
# Und sie testen!