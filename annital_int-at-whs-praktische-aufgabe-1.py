startzustand = [3,3,1]
zielzustand  = [0,0,0]

# Erzeuge alle denkbaren Folgezustände, wir schauen nicht auf feasibility
# Wir geben eine Liste von Listen (den Zuständen) zurück
def gib_folgezustaende(zustand):
    result = []
    m,k,b = zustand
    b_new = 0 if b else 1    
    # now try all potential operations
    # Let's keep it simple for now:
    if b:
        if k > 0:
            if m > 0:
                result.append([m-1,k-1,b_new])
            if k > 1:
                result.append([m,k-2,b_new])
            result.append([m,k-1,b_new])       
        if m > 1:
            result.append([m-2,k,b_new])
        if m > 0:
            result.append([m-1,k,b_new]) 
    if not b:
        if k < 3:
            if m < 3:
                result.append([m+1,k+1,b_new])
            if k < 2:
                result.append([m,k+2,b_new])
            result.append([m,k+1,b_new])       
        if m < 2:
            result.append([m+2,k,b_new])
        if m < 3:
            result.append([m+1,k,b_new])
            
    return result

## Some tests
print(gib_folgezustaende([3,3,1]))            
print(gib_folgezustaende([2,2,0]))
print(gib_folgezustaende([2,2,1]))
print(gib_folgezustaende([1,1,0]))
print(gib_folgezustaende([1,1,1]))
## Jetzt checken wir noch, ob die Zustände valide sind!
def is_valid(zustand):
    m,k,b = zustand
    # es gibt im Westen mehr Kannibalen, als Missionare
    if m < k and m > 0: return False 
    # es gibt im Osten mehr Kannibalen, als Missionare
    if m > k and m < 3: return False
    return True

def gib_valide_folgezustaende(zustand):
    return [z for z in gib_folgezustaende(zustand) if is_valid(z)]

# Some tests
print(gib_valide_folgezustaende([3,3,1]))            
print(gib_valide_folgezustaende([2,2,0]))
print(gib_valide_folgezustaende([2,2,1]))
print(gib_valide_folgezustaende([1,1,0]))
print(gib_valide_folgezustaende([1,1,1]))
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

suche(startzustand,[],debug=2) # One solution
# All solutions, debugging disabled
suche(startzustand,[],all_solutions=True,debug=0)
## Noch nicht besonders schön ist die Funktion für die Folgezustände
## Gestalten wir die ein wenig knapper und sehen gleich einen
### Maximalwert für die Anzahl vor
### Um es später leichter verwenden zu können, geben wir den Wert
### global vor
max_value = 3

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
        [m,k+incr*2,b_new]
    ]
    # Entferne alle, die für m bzw. k kleiner 0 oder größer 3 sind
    return [[k,m,b] for k,m,b in fzustaende 
                if k >= 0 and k <= max_value and m >= 0 and m <= max_value]

## Some tests
print(gib_folgezustaende([3,3,1]))            
print(gib_folgezustaende([2,2,0]))
print(gib_folgezustaende([2,2,1]))
print(gib_folgezustaende([1,1,0]))
print(gib_folgezustaende([1,1,1]))
# All solutions, debugging disabled
suche(startzustand,[],all_solutions=True,debug=0)
# Wir brauchen eine allgemeinere Version von is_valid:
def is_valid(zustand):
    global max_value
    m,k,b = zustand
    # es gibt im Westen mehr Kannibalen, als Missionare
    if m < k and m > 0: return False 
    # es gibt im Osten mehr Kannibalen, als Missionare
    if m > k and m < max_value: return False
    return True
# Jetzt können wir schon suchen!
max_value = 4
suche([4,4,1],[],all_solutions=True,debug=2)
## Hier ist Platz für ihre Lösung in ihrem eigenen Notebook, senden Sie mir ein korrekt ausgeführtes Jupyter-Notebook, 
# per Link zu github oder zu einem öffentlichen Kaggle-Kernel!
# Ab hier hat das Boot also 3 Plätze - dementsprechend gibt es mehr Folgezustände

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
        # die neuen:
        [m+incr*3,k,b_new],
        [m,k+incr*3,b_new],
        [m+incr*2,k+incr,b_new],
        [m+incr,k+incr*2,b_new]
    ]
    # Entferne alle, die für m bzw. k kleiner 0 oder größer max_value sind
    return [[k,m,b] for k,m,b in fzustaende 
                if k >= 0 and k <= max_value and m >= 0 and m <= max_value]

## Some tests
print(gib_folgezustaende([4,4,1]))
print(gib_folgezustaende([1,1,0]))
# Suche diesmal mit den neuen Folgezuständen, alles andere bleibt wie vorher
max_value = 4
suche([4,4,1],[],debug=0) # One solution, debug off
# Anstatt die Folgezustände mit jedem neuen Platz im Boot händisch zu erweitern, können die Folgezustände von der Anzahl der Plätze abhängig gemacht werden.

def gib_folgezustaende(zustand):
    global max_value
    global boot_plaetze
    m,k,b = zustand
    incr  = -1 if b else +1
    b_new = 0 if b else 1
    fzustaende = []
    
    # Anzahl der Wiederholungen für die innere for-Schleife in eine extra Variable
    jj = boot_plaetze+1
    for i in range(boot_plaetze+1):
        for j in range(jj):
            fzustaende.append([m+incr*i,k+incr*j,b_new])
        # Innere for-Schleife immer einen Durchlauf weniger
        jj = jj - 1
    
    # Der Zustand, bei dem das Boot leer übergesetzt hätte, muss entfernt werden (m und k sind unverändert, b ist b_new)
    fzustaende.remove([m,k,b_new])
    
    # Entferne alle, die für m bzw. k kleiner 0 oder größer max_value sind
    return [[k,m,b] for k,m,b in fzustaende 
                if k >= 0 and k <= max_value and m >= 0 and m <= max_value]

## Some tests
max_value = 4
boot_plaetze = 3
print(gib_folgezustaende([4,4,1]))
print(gib_folgezustaende([1,1,0]))
# Suche mit den automatisierten Folgezuständen
suche([4,4,1],[],debug=0) # One solution, debug off