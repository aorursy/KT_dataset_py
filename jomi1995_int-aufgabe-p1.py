# by JM

# nun 4 Kannibale und 4 Missionare

startzustand = [4,4,1]

zielzustand  = [0,0,0]



#Änderung des "naiven" Algo damit 3 leute in ein Boot passen



def gib_folgezustaende(zustand):

    result = []

    m,k,b = zustand

    b_new = 0 if b else 1    

    

    if b:

        #Boot auf der linken Reite

        if k > 0:

            if m > 0:

                result.append([m-1,k-1,b_new])

            if m > 1:

                result.append([m-2,k-1,b_new])

            if k > 2:

                result.append([m,k-3,b_new])

            if k > 1:

                result.append([m,k-2,b_new])    

            result.append([m,k-1,b_new])

        if m > 2:

            result.append([m-3,k,b_new])

        if m > 1:

            result.append([m-2,k,b_new])

        if m > 0:

            result.append([m-1,k,b_new]) 

    if not b:

        #Boot auf der rechten Seite

        if k < 4:

            if m < 4:

                result.append([m+1,k+1,b_new])    

            if m < 3:

                result.append([m+2,k+1,b_new])                    

            if k < 3:

                result.append([m,k+2,b_new])

            if k < 2:

                result.append([m,k+3,b_new])    

            result.append([m,k+1,b_new])       

        if m < 2:

                result.append([m+2,k,b_new])

        if m < 3:

                result.append([m+1,k,b_new])

            

    return result



## Some tests

print(gib_folgezustaende([4,4,1]))

print(gib_folgezustaende([3,3,0]))

print(gib_folgezustaende([3,3,1]))            

print(gib_folgezustaende([2,2,0]))

print(gib_folgezustaende([2,2,1]))

print(gib_folgezustaende([1,1,0]))

print(gib_folgezustaende([1,1,1]))
def is_valid(zustand):

    m,k,b = zustand

    # es gibt auf der linken Seite mehr Kannibalen, als Missionare

    if m < k and m > 0: return False 

    # es gibt auf der rechten Seite mehr Kannibalen, als Missionare

    if m > k and m < 4: return False

    return True



def gib_valide_folgezustaende(zustand):

    return [z for z in gib_folgezustaende(zustand) if is_valid(z)]



print(gib_valide_folgezustaende([4,4,1]))

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



suche(startzustand,[],debug=0)
