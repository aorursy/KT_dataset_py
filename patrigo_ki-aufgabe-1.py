#Variation des Algorithmus, welcher erlaubt eine beliebige Bootsgröße und eine beliebige Größe der Personengruppe zu benutzen
max_value = 4 #Anzahl der Missionäre und Anzahl der Kanibalen
boot_groesse = 3 #Größe des Bootes
startzustand = [max_value,max_value,1]
zielzustand = [0,0,0]
solution_found = False;

# erzeugt alle Folgezustände für die angegebene boot_grösse
def gib_fzustaende(zustand):
    global boot_groesse
    boot_size = boot_groesse
    fzustaende = []
    m,k,b = zustand
    incr  = -1 if b else +1
    b_new = 0 if b else 1
    # geht durch alle Bootgrößen, da z.B mit einem Boot mit Platz für 3 Personen auch nur 2 Personen fahren können
    while boot_size >= 1:
        i = boot_size
        # fügt alle Folgezustände für eine mögliche Bootgröße bzw. Untergröße zu fzustaende hinzu
        while i >= 0:
            fzustaende.append([m+incr*(boot_size-i),k+incr*i,b_new])
            i = i - 1
        boot_size = boot_size - 1
        
    return fzustaende

def gib_folgezustand(zustand):
    global max_value
    global boot_groesse
    m,k,b = zustand
    incr = -1 if b else +1
    b_new = 0 if b else 1
    
    fzustaende = gib_fzustaende(zustand)
    
    return [[m,k,b] for m,k,b in fzustaende if k >= 0 and k <= max_value and m >= 0 and m <= max_value]

def is_valid(zustand):
    global max_value
    m,k,b = zustand
    if m < k and m > 0: return False
    if m > k and m < max_value: return False
    return True

def gib_valide_folgezustaende(zustand):
    return [z for z in gib_folgezustand(zustand) if is_valid(z)]

def suche(zustand, history, all_solutions=False,level=0, debug=1):
    global solution_found
    if debug: print(level*' ',zustand," ->",end="")
    
    if zustand == zielzustand: return(True, history+[zustand])  # Falls Zustand ein Zielzustand ist, stoppe dieses Pfad und gebe Verlauf zurück
    fzustaende = gib_valide_folgezustaende(zustand)
    
    if debug: print("  ",fzustaende)
    if not len(fzustaende): return (False, [])    #Falls Pfad zu Ende, gebe zurück, dass dies keine Lösung ist
    for z in fzustaende:
        if z not in history+zustand:
            res1,res2 = suche(z,history+[zustand],all_solutions,level+1,debug) #Falls es einen nicht untersuchten Zustand gibt, gehe diesen Pfad
            if res1:
                if all_solutions:
                    print("Solution found: ", res1, res2)
                    solution_found = True
                else:
                    return(res1, res2) #Gibt nur die erste gefundene Lösung zurück
        else:
            if debug == 2: print((level+1)*' '+"repeated",z)
    if solution_found and level == 0:
        return
    else:
        return(False,[])
suche(startzustand,[],all_solutions=True,debug=0)
