## Hier ist Platz für ihre Lösung in ihrem eigenen Notebook, senden Sie mir ein korrekte ausgeführtes Jupyter-Notebook, 
# per Link zu github oder zu einem öffentlichen Kaggle-Kernel!
## Hier ist Platz für ihre Lösung in ihrem eigenen Notebook, senden Sie mir ein korrekte ausgeführtes Jupyter-Notebook, 
# per Link zu github oder zu einem öffentlichen Kaggle-Kernel!
## Lösung von Felix Sattler

zielzustand = [0,0,0]


def gib_folgezustaende(zustand):
    global max_value
    m,k,b = zustand
    incr  = -1 if b else +1
    b_new = 0 if b else 1
    
    
    fzustaende = []
    #Manuelle Eingabe der Zustände für 3 Plätze im Boot
    #Folgezustände in Abhängigkeit der Anzahl m bzw. k
    #fzustaende = [
        
        #Anteilig im Boot 
    #    [m+incr,k+incr,b_new],
    #    [m+incr*2,k+incr,b_new],
    #    [m+incr,k+incr*2,b_new],
        
        #Je eine Gruppe im Boot mit max_incr = 3 (Plätze im Boot)
    #    [m+incr,k,b_new],
    #    [m+incr*2,k,b_new],
    #    [m+incr*3,k,b_new],
        
    #    [m,k+incr,b_new],
    #    [m,k+incr*2,b_new],
    #    [m,k+incr*3,b_new]
    #]
        
    #Aber das ist Mist; Hier sollte eine abstrakte Funktion hin;
    #Wie wäre es z.B. mit:
    for i in range(anzahl_boot+1):
        for j in range(anzahl_boot-i,-1,-1):
            # Verteilung im Boot ergibt sich nach 
            if not(i==0 and j==0):
                fzustaende.append([m+incr*i,k+incr*j,b_new])
            
    # Entferne alle, die für m bzw. k kleiner 0 oder größer max_value sind
    return [[k,m,b] for k,m,b in fzustaende 
                if k >= 0 and k <= max_value and m >= 0 and m <= max_value]

## Some tests
#print(gib_folgezustaende([3,3,1]))            
#print(gib_folgezustaende([2,2,0]))
#print(gib_folgezustaende([2,2,1]))
#print(gib_folgezustaende([1,1,0]))
#print(gib_folgezustaende([1,1,1]))

# Allgemeine Version von is_valid:
def is_valid(zustand):
    global max_value
    m,k,b = zustand
    # es gibt im Westen mehr Kannibalen, als Missionare
    if m < k and m > 0: return False 
    # es gibt im Osten mehr Kannibalen, als Missionare
    if m > k and m < max_value: return False
    return True


# Check for validity 
def gib_valide_folgezustaende(zustand):
    return [z for z in gib_folgezustaende(zustand) if is_valid(z)]

# Rekursive Suche in die Tiefe (depth-first search with chronolocigal backtracking)
def suche(zustand,history,all_solutions=False,level=0,debug=1):
    if debug: print(level*' ',zustand," ->",end="")
        
    # if compare(zustand,zielzustand): return (True,history+[zustand])
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

max_value = 4
anzahl_boot = 3
print(suche([4,4,1],[], all_solutions=False, debug=0))
print("  ++      UU\n (oo)   >(oo)<\n--II--  --II--\n /__\    {/\} \n_I  I_  _I  I_ \nWHOOLA CHAKA")

