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
solution_count = 0
def suche(zustand,history,all_solutions=False,level=0,debug=1):
    global solution_count
    if level == 0: solution_count = 0
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
                    solution_count += 1
                else:
                    return (res1,res2) # Just stop
        else:
            if debug == 2: print((level+1)*' '+"repeated",z)
    if level == 0: print(solution_count, "Solutions found")
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
# Durch einfache Anpassungen kann das Programm verallgemeinert werden

# Ueber die neu eingefuehrte Variable boat_size kann die Anzahl an Plaetzen im Boot festgelegt werden
boat_size = 2
max_value = 3

def gib_folgezustaende(zustand):
    global max_value
    m,k,b = zustand
    incr  = -1 if b else +1
    b_new = 0 if b else 1
    
    # Entsprechend der groesse  des Bootes werden folgezustaende erzeugt, leere oder ueberfuellte Boote werden aussortiert
    fzustaende = [[m + m_on_boat * incr, k + k_on_boat * incr, b_new] for m_on_boat in range(boat_size + 1) for k_on_boat in range(boat_size + 1) 
                  if boat_size >= m_on_boat + k_on_boat > 0]

    # Entferne alle, die für m bzw. k kleiner 0 oder größer 3 sind
    return [[k,m,b] for k,m,b in fzustaende 
                if k >= 0 and k <= max_value and m >= 0 and m <= max_value]

## Some tests (sollten die selben Ergebnisse wie zuvor liefern)
max_value = 3
boat_size = 2
startzustand = [3,3,1]
print(gib_folgezustaende([3,3,1]))            
print(gib_folgezustaende([2,2,0]))
print(gib_folgezustaende([2,2,1]))
print(gib_folgezustaende([1,1,0]))
print(gib_folgezustaende([1,1,1]))
# Test mit 3 Missionaren, 3 Kannibalen und einem 2er Boot(sollte die schon bekannten 4 Loesungen finden)
max_value = 3
boat_size = 2
startzustand = [3,3,1]
suche(startzustand,[],all_solutions=True,debug=0)
# Test mit 2 Missionaren, 2 Kannibalen und einem 2er Boot(findet 4 Loesungen)
max_value = 2
boat_size = 2
startzustand = [2,2,1]
suche(startzustand,[],all_solutions=True,debug=0)
# Test mit 4 Missionaren, 4 Kannibalen und einem 2er Boot(sollte keine Loesung finden)
max_value = 4
boat_size = 2
startzustand = [4,4,1]
suche(startzustand,[],all_solutions=True,debug=2)
# Test mit 4 Missionaren, 4 Kannibalen und einem 3er Boot(findet 200 Loesungen)
max_value = 4
boat_size = 3
startzustand = [4,4,1]
suche(startzustand,[],all_solutions=True,debug=0)