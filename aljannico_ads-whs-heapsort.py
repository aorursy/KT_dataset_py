## Top-Down Heapsort

class TD_Heapsort():

    def __init__(self,a,direction='aufsteigend',tiebreak='links'):        

        # Prepare the correct test

        if direction == 'absteigend':

            self.test = lambda x,y: x < y

        else: # aufsteigend

            self.test = lambda x,y: x > y

        # Tiebreaking: left or right

        self.left = True if tiebreak=='links' else False

        self.vgl = 0 # Zählen der Vergleiche

        # Now sort it

        self.phase1(a)

        self.phase2(a)

        if debug: print("\nVergleiche:",self.vgl,"\n")

        

    '''

    Min-Heap für absteigende Sortierung (Wurzel <= Kinder)

    Max-Heap für aufsteigende Sortierung (Wurzel >= Kinder)

    

    Rekursive Lösung (damit es leichter nachverfolgbar ist)

    '''

    def sift_in(self,a,i,anfang,ende,step=0):

        if debug: print(a,"|",anfang,ende,"|",i,end=' ')

        s_l = i*2   # linker Sohn

        s_r = i*2+1 # rechter Sohn

        # Keine Kinder mehr, done

        if s_l > ende: 

            if debug: print('-','-','-')

            return

        # Bestimme das richtige Kind

        if s_r > ende: # es gibt nur das linke Kind

            s = s_l

            if debug: s_r_str = '-'

        elif a[s_l-1] == a[s_r-1]:

            s = s_l if self.left else s_r

            if debug: s_r_str = str(s_r)

            self.vgl += 1

        else:

            self.vgl += 1

            s = s_l if self.test(a[s_l-1],a[s_r-1]) else s_r

            if debug: s_r_str = str(s_r)

        self.vgl += 1

        if self.test(a[s-1],a[i-1]):

            a[s-1],a[i-1] = a[i-1],a[s-1]

            if debug: print(s_l,s_r_str,s)

            self.sift_in(a,s,anfang,ende,step+1)

        elif debug: print(s_l,s_r_str,"-")

                

                         

    def phase1(self,a):

        ende = len(a)

        mitte = (ende // 2)

        for anfang in range(mitte,0,-1):

            self.sift_in(a,anfang,anfang,ende)

                

    def phase2(self,a):

        l = len(a)

        for ende in range(l,1,-1):

            a[ende-1],a[1-1] = a[1-1],a[ende-1]

            self.sift_in(a,1,1,ende-1)

            

debug = True # Erzeugt eine tabellarische Ausgabe, die 

             # der Tabelle aus Übung und Klausur entspricht



a = [17, 6, 11, 15, 5, 6, 1, 8]

b = a[:] # COPY of a

c = a[:] # Another COPY of a



print("Sortiere: ",a,"\n")

TD_Heapsort(a,tiebreak='rechts')

print("Aufsteigend sortiert, Tiebreak nach rechts: ",a,"\n\n")



print("Sortiere: ",b,"\n")

TD_Heapsort(b)

print("Aufsteigend sortiert, Tiebreak nach links: ",b,"\n\n")





print("Sortiere: ",c,"\n")

TD_Heapsort(c,'absteigend')

print("Absteigend sortiert, Tiebreak nach links: ",c)
## Top-Down Heapsort

class TD_Heapsort():

    def __init__(self,direction='aufsteigend',tiebreak='links'):        

        # Tiebreaking: left or right

        self.left = True if tiebreak=='links' else False

        # Prepare the correct tests

        if direction == 'absteigend': # absteigend, Min-Heap

            self.test_wurzel = lambda s,w: s < w

            if self.left: 

                self.test_sohn = lambda sl,sr: sl <= sr # Fall 3

            else:

                self.test_sohn = lambda sl,sr: sl < sr # Fall 4

        else: # aufsteigend, Max-Heap

            self.test_wurzel = lambda s,w: s > w

            if self.left: 

                self.test_sohn = lambda sl,sr: sl >= sr # Fall 1

            else:

                self.test_sohn = lambda sl,sr: sl > sr # Fall 2

        

    def sort(self,a):

        self.vgl = 0 # Zählen der Vergleiche

        # Now sort it

        self.phase1(a)

        self.phase2(a)

        return self.vgl

    

    '''

    Min-Heap für absteigende Sortierung (Wurzel <= Kinder)

    Max-Heap für aufsteigende Sortierung (Wurzel >= Kinder)

    

    Rekursive Lösung (damit es leichter nachverfolgbar ist)

    '''

    def sift_in(self,a,i,ende):

        s_l = i*2   # linker Sohn

        s_r = i*2+1 # rechter Sohn

        # Keine Kinder mehr, done

        if s_l > ende: return

        # Bestimme das richtige Kind

        if s_r > ende: # es gibt nur das linke Kind

            s = s_l

        else:

            self.vgl += 1

            s = s_l if self.test_sohn(a[s_l-1],a[s_r-1]) else s_r

        self.vgl += 1

        if self.test_wurzel(a[s-1],a[i-1]):

            a[s-1],a[i-1] = a[i-1],a[s-1]

            self.sift_in(a,s,ende)

                         

    def phase1(self,a):

        ende = len(a)

        mitte = (ende // 2)

        for anfang in range(mitte,0,-1):

            self.sift_in(a,anfang,ende)

                

    def phase2(self,a):

        l = len(a)

        for ende in range(l,1,-1):

            a[ende-1],a[1-1] = a[1-1],a[ende-1]

            self.sift_in(a,1,ende-1)



a = [17, 6, 11, 15, 5, 6, 1, 8]

b = a[:] # COPY of a

c = a[:] # Another COPY of a



h1 = TD_Heapsort(tiebreak="rechts")

vgl = h1.sort(a)

print("Aufsteigend sortiert: ",a,"\nVergleiche: ",vgl,"\n")



h2 = TD_Heapsort(tiebreak="links")

vgl = h2.sort(b)

print("Aufsteigend sortiert: ",b,"\nVergleiche: ",vgl,"\n")





h3 = TD_Heapsort('absteigend')

vgl = h3.sort(c)

print("Absteigend sortiert: ",c,"\nVergleiche: ",vgl,"\n")
a1 = [9,12,247,18,4,21,99,123,76]

a2 = [1,2,3,4,9,8,7,6,5,4,2,1]



# Create the Heapsorter for "absteigend" und "rechts"

h = TD_Heapsort('absteigend','rechts')



# Apply it to data

vgl = h.sort(a1)

print("Absteigend sortiert: ",a1,"\nVergleiche: ",vgl,"\n")



# ..and again

vgl = h.sort(a2)

print("Absteigend sortiert: ",a2,"\nVergleiche: ",vgl,"\n")



import math

n = len(a2)



print("Abschätzung der Vergleiche im O-Kalkül:", n*math.log(n,2))

# Die Magie der Objektorientierung...wir überschreiben eine der Methoden

class TD_Heapsort_Iterativ(TD_Heapsort):

    def sift_in(self,a,i,ende,step=0):

        tausch = True

        while i*2 <= ende and tausch: # Keine Kinder mehr: 

            s_l = i*2   # linker Sohn

            s_r = i*2+1 # rechter Sohn

            # Bestimme das richtige Kind

            if s_r > ende: # es gibt nur das linke Kind

                s = s_l

            else:

                self.vgl += 1

                s = s_l if self.test_sohn(a[s_l-1],a[s_r-1]) else s_r

            self.vgl += 1

            if self.test_wurzel(a[s-1],a[i-1]):

                a[s-1],a[i-1] = a[i-1],a[s-1]

                i = s

            else:

                tausch = False

                

a = [17, 6, 11, 15, 5, 6, 1, 8]

b = a[:] # COPY of a

c = a[:] # Another COPY of a



h1 = TD_Heapsort_Iterativ(tiebreak="rechts")

vgl = h1.sort(a)

print("Aufsteigend sortiert: ",a,"\nVergleiche: ",vgl,"\n")



h2 = TD_Heapsort_Iterativ(tiebreak="links")

vgl = h2.sort(b)

print("Aufsteigend sortiert: ",b,"\nVergleiche: ",vgl,"\n")



h3 = TD_Heapsort_Iterativ('absteigend')

vgl = h3.sort(c)

print("Absteigend sortiert: ",c,"\nVergleiche: ",vgl,"\n")

                
class BU_Heapsort(TD_Heapsort):

    '''

    Wir müssen nur das Einsinken anpassen!

    '''

    def sift_in(self,a,i,ende):

        vgl = 0

        if debug: print(a,i,ende,end=' ')

        start = i

        weg = [i]

        # Virtuellen Einsinkpfad bestimmen ("Weg" in der Tabelle)

        while i*2 <= ende: # Keine Kinder mehr: 

            s_l = i*2   # linker Sohn

            s_r = i*2+1 # rechter Sohn

            # Bestimme das richtige Kind

            if s_r > ende: # es gibt nur das linke Kind

                i = s_l

            else:

                vgl += 1

                i = s_l if self.test_sohn(a[s_l-1],a[s_r-1]) else s_r

            weg.append(i)

        if debug: print(weg,end=' ')

        # Einfügeposition finden (epos)

        ringtausch = []

        epos=start

        for pos in reversed(weg):

            # Look for

            if ringtausch == [] and pos != start:

                vgl += 1

                # if self.test_wurzel(a[pos-1],a[start-1]): # Würde bei Gleichstand zur

                # Wurzel weiter nach oben rücken, semantisch ok, aber nicht, was das

                # Lernsystem möchte!

                # Besser:

                # Epos gefunden, wenn s >= w für max_heap bzw. s <= w für min_heap

                # Getestet wird das mit not (w > s) und not (w < s), weil so

                # unsere Bedingungen oben gesetzt sind (für TD)

                if not self.test_wurzel(a[start-1],a[pos-1]): 

                    ringtausch = [start]

                    store = a[start-1]

                    epos = pos

            if ringtausch != []:

                ringtausch.append(pos)

                a[pos-1],store = store,a[pos-1]

        if debug: print("Epos:",epos, "Ringtausch:",ringtausch, "Vgl:",vgl)

        self.vgl += vgl # zählen der Vergleiche



debug = True          

            

a = [17, 6, 11, 15, 5, 6, 1, 8]

b = a[:] # COPY of a

c = a[:] # Another COPY of a



h1 = BU_Heapsort(tiebreak="rechts")

vgl = h1.sort(a)

print("Aufsteigend sortiert: ",a,"\nVergleiche: ",vgl,"\n")



h2 = BU_Heapsort(tiebreak="links")

vgl = h2.sort(b)

print("Aufsteigend sortiert: ",b,"\nVergleiche: ",vgl,"\n")





h3 = BU_Heapsort('absteigend')

vgl = h3.sort(c)

print("Absteigend sortiert: ",c,"\nVergleiche: ",vgl,"\n")
debug = True          

            

a = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]



h1 = BU_Heapsort(tiebreak="rechts")

vgl = h1.sort(a)

print("Aufsteigend sortiert: ",a,"\nVergleiche: ",vgl,"\n")
debug = True          

            

a = [13,4,13,17,14]



h1 = BU_Heapsort(tiebreak="rechts")

vgl = h1.sort(a)

print("Aufsteigend sortiert: ",a,"\nVergleiche: ",vgl,"\n")