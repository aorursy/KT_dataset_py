max_value = 4
boot_groesse = max_value - 1
startzustand = [max_value,max_value,1]
zielzustand  = [0,0,0]


def gib_folgezustaende(zustand):
    global max_value
    global boot_groesse
    m,k,b = zustand
    incr  = -1 if b else +1
    b_new = 0 if b else 1
    
    fzustaende = []
    i = 1
    while i <= boot_groesse:
        im = i
        ik = 0
        while ik <= i:
            fzustaende.append([m+incr*im,k+incr*ik,b_new])
            ik += 1
            im -= 1
        i += 1
    # Entferne alle, die für m bzw. k kleiner 0 oder größer 3 sind
    return [[k,m,b] for k,m,b in fzustaende 
                if k >= 0 and k <= max_value and m >= 0 and m <= max_value]

def is_valid(zustand):
    global max_value
    m,k,b = zustand
    if m < k and m > 0: return False
    if m > k and m < max_value: return False
    return True

def gib_valide_folgezustaende(zustand):
    return [z for z in gib_folgezustaende(zustand) if is_valid(z)]

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

suche(startzustand,[],all_solutions=False,debug=0)

max_value = 2
boot_groesse = max_value - 1
startzustand = [max_value,max_value,1]
suche(startzustand,[],all_solutions=False,debug=0)
suche(startzustand,[],all_solutions=True,debug=0)
max_value = 3
boot_groesse = max_value - 1
startzustand = [max_value,max_value,1]
suche(startzustand,[],all_solutions=False,debug=0)
max_value = 5
boot_groesse = max_value - 1
startzustand = [max_value,max_value,1]
suche(startzustand,[],all_solutions=False,debug=0)
max_value = 6
boot_groesse = max_value - 1
startzustand = [max_value,max_value,1]
suche(startzustand,[],all_solutions=False,debug=0)
max_value = 7
boot_groesse = max_value - 1
startzustand = [max_value,max_value,1]
suche(startzustand,[],all_solutions=False,debug=0)
max_value = 100
boot_groesse = max_value - 1
startzustand = [max_value,max_value,1]
suche(startzustand,[],all_solutions=False,debug=0)