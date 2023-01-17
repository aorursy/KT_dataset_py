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
## Lösung der gestellten Aufgabe

m_max = 4    # Gesamtanzahl aller Missionare
k_max = 4    # Gesamtanzahl aller Kanibalen
b_cap = 3    # Anzahl der Plätze im Boot (Bootskapazität)

startzustand = (m_max, k_max, 1)
zielzustand  = (0, 0, 0)


## Diese Funktion führt eine rekursive Tiefensuche durch (depth-first search with chronolocigal backtracking)
def suche(zustand, history, all_solutions = False, level = 0, debug = 1):
    if debug >= 1: print(level * ' ', zustand, " ->", end = "")
        
    # Ist der aktuelle Zustand gleich dem Zielzustand, wurde eine Lösung gefunden.
    if zustand == zielzustand: 
        return (True, history + [zustand])
    
    # Ermittlung der Liste aller Folezustände
    fzustaende = gib_folgezustaende(zustand)
    # Filterung der Liste mit den Folgezuständen, sodass diese nur Zustände enthält, 
    # bei denen kein Missionar gefressen wurde
    fzustaende = [z for z in fzustaende if not wird_missionar_gegessen(z)]
    
    if debug >= 1: print("  ", fzustaende)
        
    # Es gibt keine Folgezustände, dieser Pfad ist eine Sackgasse und führte zu keiner Lösung
    if len(fzustaende) == 0: 
        return (False, [])
    
    for z in fzustaende:
        # Prüft ob man an diesem Zustand schon zuvor gewesen ist
        if z not in history + [zustand]: 
            # wenn nicht kann weiter gelaufen bzw. gesucht werden
            loesung_gefunden, loesung = suche(z, history + [zustand], all_solutions, level + 1, debug)
            if loesung_gefunden: 
                if all_solutions:
                    # Lösung nur ausgeben und weiter machen (vielleicht gibt es ja noch weitere Lösungen)
                    print("  Solution found: \n", loesung, "\n\n")
                else:
                    # Lösung ausgeben und diese Zurückgeben (Ende der Suche)
                    print("  Solution found: \n", loesung, "\n\n")
                    return (loesung_gefunden, loesung)
        else: 
            # wenn doch wird dieser Zustand ignoriert
            if debug >= 2: print((level + 1) * ' ' + "repeated", z)
    
    # Dieser Pfad führte zu keiner Lösung
    return (False, [])


## Diese Funktion liefert für einen Zustand eine Liste aller Folgezustände
def gib_folgezustaende(zustand):
    m, k, b = zustand
    
    # Algemein werden alle möglichen Übergänge ermittelt.
    uebergaenge = gib_uebergaenge()
    
    # Auf den übergebenen Zustand werden alle Übergänge angewendet und 
    # somit alle Folgezustände ermittelt. 
    fzustaende = []
    for uebergang in uebergaenge:
        fzustaende += [gib_folgezustand(zustand, uebergang)]
    
    # Die Liste enthält noch Zustände, die es garnicht geben kann.
    # Entferne aus der Menge der Folgezustände alle Zustände, 
    # bei denen m oder k kleiner 0 oder größer m_max bzw. k_max sind.
    return [folgezustand for folgezustand in fzustaende if ist_gueltig(folgezustand)]


## Diese Funktion enthält einen rekursiven Permutationsalgorithmus, 
## der alle möglichen Übergänge (Bootsbesetzunen) als Menge zurückliefert.
def gib_uebergaenge(uebergang = (0, 0)):
    m, k = uebergang
    if m + k > b_cap - 1:
        return {(m, k)}
    else:
        if m == 0 and k == 0:
            return gib_uebergaenge((m + 1, k)) | gib_uebergaenge((m, k + 1))
        else:
            return {(m, k)} | gib_uebergaenge((m + 1, k)) | gib_uebergaenge((m, k + 1))
    
    
## Diese Funktion wendet den Übergebenen Übergang auf den Übergeben Zustand an und
## erzeugt daraus einen Folgezustand.
def gib_folgezustand(zustand, uebergang):
    m, k, b = zustand
    dm, dk = uebergang
    if b:
        return (m - dm, k - dk, 0)
    else: 
        return (m + dm, k + dk, 1)


## Diese Funktion checkt, ob der übergebene Zustand gültig ist.
def ist_gueltig(zustand):
    m, k, b = zustand
    return k >= 0 and k <= k_max and m >= 0 and m <= m_max


## Diese Funktion checkt, ob in dem übergebenen Zustand Missionare gegessen werden.
def wird_missionar_gegessen(zustand):
    m, k, b = zustand
    
    # es gibt im Westen mehr Kannibalen als Missionare
    if m < k and m > 0: 
        return True    # dann werden die Missionare im Westen von den Kanibalen gegessen
    
    # es gibt im Osten mehr Kannibalen als Missionare
    if m > k and m < m_max: 
        return True    # dann werden die Missionare im Osten von der Kanibalen gegessen
    
    return False


# durchführen der Suche nach Lösungen
suche(startzustand,[],all_solutions=True,debug=2) 