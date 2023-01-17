## Noch nicht besonders schön ist die Funktion für die Folgezustände
## Gestalten wir die ein wenig knapper und sehen gleich einen
### Maximalwert für die Anzahl vor
### Um es später leichter verwenden zu können, geben wir den Wert
### global vor
max_value = 4

startzustand = [max_value,max_value,1]
zielzustand  = [0,0,0]

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
# Neue, erweiterte Funktion für ein Boot mit 3
def gib_folgezustaende(zustand):
    global max_value
    m,k,b = zustand
    incr  = -1 if b else +1
    b_new = 0 if b else 1
    
    # Logisch: wenn mein Boot p Plätze hat, dann kann (wenn
    # genügend Personal zur Verfügung steht) jede Kombination
    # von 2 positiven Werten, deren Addition etwas <= p ergibt, auf
    # die Reise geschickt werden, d.h. wir brauchen alle
    # Kombination von zwei nicht-negativen ganzen Zahlen,
    # die diese Eigenschaft erfüllen
    #  kombis = { (m,k) | 0 <= m <= p und 0 <= k <= p und m+k <= p }
    # (Hinweis: man könnte gib_folgezustaende generalieren, in dem man
    # all diese Kombinationen erzeugt und dann die tatsächlichen Einträge in 
    # der Liste fzustaende aus diesen Kombinationen generiert)
    
    fzustaende = [
        [m+incr,k+incr,b_new],
        [m+incr,k,b_new],
        [m,k+incr,b_new],
        [m+incr*2,k,b_new],
        [m,k+incr*2,b_new],
        [m+incr*3,k,b_new],
        [m,k+incr*3,b_new],
        [m+incr*2,k+incr*1,b_new],
        [m+incr*1,k+incr*2,b_new]
    ]
    # Entferne alle, die für m bzw. k kleiner 0 oder größer 3 sind
    return [[k,m,b] for k,m,b in fzustaende 
                if k >= 0 and k <= max_value and m >= 0 and m <= max_value]

print(gib_folgezustaende([4,4,1]))            
print(gib_folgezustaende([3,3,1]))            
print(gib_folgezustaende([2,2,0]))
print(gib_folgezustaende([2,2,1]))
print(gib_folgezustaende([1,1,0]))

# Some tests
print(gib_valide_folgezustaende([4,4,1]))            
print(gib_valide_folgezustaende([2,2,0]))
print(gib_valide_folgezustaende([2,2,1]))
print(gib_valide_folgezustaende([1,1,0]))
print(gib_valide_folgezustaende([1,1,1]))
# All solutions, debugging disabled
suche(startzustand,[],all_solutions=False,debug=0)
