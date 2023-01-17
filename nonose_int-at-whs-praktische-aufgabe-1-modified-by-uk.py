# Berechnung eines Lösungsweges für das Problem mit den 4 Kannibalen, den 4 Missionaren und einem Drei-Mann-Boot

#Definieren des Start- und des Zielzustandes (m,k,b) m=Missionare, k=Kannibale, b=Uferseite
startzustand = [4,4,1]
print("Startzustand:  ",startzustand)
zielzustand  = [0,0,0]
print("Zielzustand:  ",zielzustand)

#Zunächst definieren wir eine Methode, die uns alle möglichen Folgezustände ausgibt.

max_value = 4 #Wir definieren hier den Maximalwert für die Anzahl (in diesem Falle 4)
count = 0 #Anzahl gefundener Lösungen

def gib_folgezustaende(zustand):
    global max_value
    m,k,b = zustand
    incr  = -1 if b else +1
    b_new = 0 if b else 1
    
    fzustaende = [
        [m+incr,k+incr,b_new], #m+1 k+1 (2P) mk
        [m+incr,k,b_new], #m+1 , k (1P) m
        [m,k+incr,b_new], #m , k+1 (1P) k
        [m+incr*2,k,b_new], #m+2 , k (2P) mm
        [m,k+incr*2,b_new], #m, k+2 (2P) kk
        [m+incr*3,k,b_new], #m + 3 , k (3P) mmm
        [m,k+incr*3,b_new], #m, k+3 (3P) kkk
        [m+incr*2,k+incr,b_new], #m+2 ,k+1 (3P) mmk
        [m+incr,k+incr*2,b_new], #m+1, k+2 (3P) mkk
    ]
    # Entferne alle, die für m bzw. k kleiner 0 oder größer 4 sind
    return [[k,m,b] for k,m,b in fzustaende 
                if k >= 0 and k <= max_value and m >= 0 and m <= max_value]

#Testausgabe zur Anzeige aller Folgezustände für den Startzustand [4,4,1]
print("#Test:  Folgezustände von [4,4,1]:  ",gib_folgezustaende([4,4,1]))

###################################################################################

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
                    print("Lösung gefunden: ",res1,res2)
                    count = count + 1
                else:
                    return (res1,res2) # Just stop
        else:
            if debug == 2: print((level+1)*' '+"repeated",z)
    return (False,[])
#####################################################################################

#Jetzt suchen wir nur noch alle möglichen Lösungen
suche([4,4,1],[],all_solutions=True,debug=0)
print("Insgesamt wurden ",count, " Lösungen gefunden")

