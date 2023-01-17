# Startzustand angepasst

startzustand = [4,4,1]

zielzustand  = [0,0,0]



def gib_folgezustaende(zustand):

    result = []

    m,k,b = zustand

    b_new = 0 if b else 1    



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

        [m+incr*3,k,b_new],

        [m,k+incr*3,b_new],

        [m+incr*2,k+incr,b_new],

        [m+incr,k+incr*2,b_new],

    ]

    return [[k,m,b] for k,m,b in fzustaende 

                if k >= 0 and k <= max_value and m >= 0 and m <= max_value]
max_value = 4

suche([4,4,1],[],all_solutions=True,debug=2)