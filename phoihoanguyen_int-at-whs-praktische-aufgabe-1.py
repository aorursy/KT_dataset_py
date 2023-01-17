## Lösung der Aufgabe 4 Missionare, 4 Kannibale, 3-Platz-Boot- Problem
startzustand = [4,4,1]
zielzustand  = [0,0,0]

print("Startzustand: ", startzustand, "\nZielzustand: ", zielzustand)

max_value = 4

def gib_folgezustaende(zustand):
    global max_value
    m,k,b = zustand
    incr  = -1 if b else +1
    b_new = 0 if b else 1
    
    fzustaende = [
        [m+incr,k+incr,b_new],
        [m+incr,k,b_new],
        [m,k+incr,b_new],
        [m+incr*2,k,b_new],
        [m,k+incr*2,b_new],
        #erweiterter Zustand
        [m+incr,k+incr*2,b_new],
        [m+incr*2,k+incr,b_new],
        [m+incr*3,k,b_new],
        [m,k+incr*3,b_new]
    ]
    # Entferne alle, die fuer m bzw. k kleiner 0 oder groeßer 4 sind
    return [[k,m,b] for k,m,b in fzustaende 
                if k >= 0 and k <= max_value and m >= 0 and m <= max_value]

## Tests für gib_folgezustaende()
print("\nTests für gib_folgezustaende: \n")
print(gib_folgezustaende([4,4,1]))            
print(gib_folgezustaende([2,2,0]))
print(gib_folgezustaende([2,2,1]))
print(gib_folgezustaende([1,1,0]))
print(gib_folgezustaende([1,1,1]))


## Methode für Ueberpruefung der Validen Zustaende
def is_valid(zustand):
    m,k,b = zustand
    # es gibt im Westen mehr Kannibalen, als Missionare
    if m < k and m > 0: return False 
    # es gibt im Osten mehr Kannibalen, als Missionare
    if m > k and m < 4: return False
    return True

def gib_valide_folgezustaende(zustand):
    return [z for z in gib_folgezustaende(zustand) if is_valid(z)]

# Some tests
# print(gib_valide_folgezustaende([4,4,1]))            
# print(gib_valide_folgezustaende([2,2,0]))
# print(gib_valide_folgezustaende([2,2,1]))
# print(gib_valide_folgezustaende([1,1,0]))
# print(gib_valide_folgezustaende([1,1,1]))


# Allgemeinere Version von is_valid:
def is_valid(zustand):
    global max_value
    m,k,b = zustand
    # es gibt im Westen mehr Kannibalen, als Missionare
    if m < k and m > 0: return False 
    # es gibt im Osten mehr Kannibalen, als Missionare
    if m > k and m < max_value: return False
    return True

# Some tests
# print(is_valid([4,4,1]))            
# print(is_valid([2,2,0]))
# print(is_valid([2,2,1]))
# print(is_valid([1,1,0]))
# print(is_valid([1,1,1]))


# Rekursive Suche in die Tiefe (depth-first search with chronolocigal backtracking)
print("\n\nTiefensuche für [4,4,1]\n")

count = 0

def suche(zustand,history,all_solutions=False,level=0,debug=1):
    global count
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
                    count +=1
                else:
                    return (res1,res2) # Just stop
        else:
            if debug == 2: print((level+1)*' '+"repeated",z)
    return (False,[])

# suche(startzustand,[],debug=2) # One 

# Alle Solutions ausgeben, debugging disabled
suche(startzustand,[],all_solutions=True,debug=0)

print("\nEs wurden insgesamt ", count, " Loesungen gefunden")
