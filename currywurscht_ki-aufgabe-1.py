startzustand = [3,3,1]
zielzustand  = [0,0,0]
max_value = 3
def gib_folgezustaende(zustand):
    global max_value
    m,k,b = zustand
    incr  = -1 if b else +1
    b_new = 0 if b else 1
    fzustaende = [
        #wenn b = 1 wird 1 abgezogen, ist b= 0 wird 1 draufaddiert
        #zustaende fuer 2er boot
        [m+incr,k+incr,b_new],
        [m+incr,k,b_new],
        [m,k+incr,b_new],
        [m+incr*2,k,b_new],
        [m,k+incr*2,b_new]
        #Erweiterung um Berechnung der Folgezustände fuer ein 3er Boot bei einer max. Anzahl von 4 fuer m und k
        #zustaende fuer 3er boot
        ,[m+incr*3,k,b_new] #3m 0k
        ,[m,k+incr*3,b_new] #0m 3k
        ,[m+incr*2,k-1,b_new] #2m 1k
        ,[m-1,k+incr*2,b_new] # 1m 2k
    ]
    # Entferne alle, die für m bzw. k kleiner 0 oder größer 3 sind
    return [[k,m,b] for k,m,b in fzustaende 
                if k >= 0 and k <= max_value and m >= 0 and m <= max_value]

# Wir brauchen eine allgemeinere Version von is_valid:
def is_valid(zustand):
    global max_value
    m,k,b = zustand
    # es gibt im Westen mehr Kannibalen, als Missionare
    if m < k and m > 0: return False 
    # es gibt im Osten mehr Kannibalen, als Missionare
    if m > k and m < max_value: return False
    return True

def gib_valide_folgezustaende(zustand):
    return [z for z in gib_folgezustaende(zustand) if is_valid(z)]

# Rekursive Suche in die Tiefe (depth-first search with chronolocigal backtracking)
def suche(zustand,history,all_solutions=False,level=0,debug=1):
    if debug: print(level*' ',zustand," ->",end="")
    if zustand == zielzustand: return (True,history+[zustand])
    fzustaende = gib_valide_folgezustaende(zustand)
    if debug: print("  ",fzustaende)   
    if not len(fzustaende): return (False,[])
    for z in fzustaende:
        if z not in history+zustand:
            res1,res2 = suche(z,history+[zustand],all_solutions,level+1,debug)
            if res1: 
                if all_solutions:
                    print("Solution found: ",res1,res2)
                else:
                    return (res1,res2) # Just stop
        else:
            if debug == 2: print((level+1)*' '+"repeated",z)
    return (False,[])
# Jetzt können wir schon suchen!
max_value = 4
suche([4,4,1],[],all_solutions=False,debug=0)
